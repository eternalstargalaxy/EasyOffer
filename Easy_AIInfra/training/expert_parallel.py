"""
【题目】手撕 Expert Parallel (EP) + All-to-All Dispatch

【背景】
MoE 模型的专家分布在不同 GPU 上，每个 GPU 只持有部分专家。
EP 的核心是 All-to-All 通信：dispatch token 到对应专家所在 GPU，
计算后再 combine 回来。

【输入/输出】
输入: tokens (batch, seq, dim), expert_assignments (batch, seq, top_k)
输出: tokens 经过专家处理后的结果

【考察点】
- All-to-All 通信模式（每 GPU 发送/接收不同数据）
- Expert 负载均衡（auxiliary loss）
- 与 TP/PP 的组合
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class ExpertParallelMoE(nn.Module):
    """
    简化版 Expert Parallel MoE
    假设: world_size 个 GPU, 每个 GPU 持有 n_experts/world_size 个专家
    """

    def __init__(self, dim: int, n_experts: int, n_local_experts: torch.Tensor, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.n_local = n_local_experts
        self.top_k = top_k
        # 当前 GPU 上的本地专家
        self.local_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
            for _ in range(n_local_experts)
        ])
        # Router (gate)
        self.gate = nn.Linear(dim, n_experts)

    def route(self, x: torch.Tensor) -> tuple:
        """路由: 决定每个 token 去哪个专家"""
        logits = self.gate(x)  # (..., n_experts)
        weights, indices = torch.topk(logits.softmax(dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # 归一化
        return weights, indices

    def all_to_all_dispatch(self, x: torch.Tensor, indices: list) -> dict:
        """
        All-to-All dispatch: 将 token 发送到对应专家所在 GPU
        简化模拟: 不做真实通信，只按 expert_id 分组
        """
        # 实际实现中用 dist.all_to_all_single
        # 这里模拟: 按 local_expert_id 分组
        local_expert_id = indices % self.n_local
        dispatched = {}
        for eid in range(self.n_local):
            mask = (local_expert_id == eid)
            if mask.any():
                dispatched[eid] = x[mask]
        return dispatched

    def all_to_all_combine(self, outputs: torch.Tensor, x_shape: torch.Tensor, weights: torch.Tensor, indices: list) -> torch.Tensor:
        """All-to-All combine: 将专家输出收集回来"""
        result = torch.zeros(x_shape, device=next(self.parameters()).device)
        local_expert_id = indices % self.n_local
        for eid, out in outputs.items():
            mask = (local_expert_id == eid)
            result[mask] += out * weights[mask].unsqueeze(-1)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, indices = self.route(x)
        # Dispatch
        dispatched = self.all_to_all_dispatch(x, indices)
        # 本地专家计算
        outputs = {}
        for eid, tokens in dispatched.items():
            outputs[eid] = self.local_experts[eid](tokens)
        # Combine
        result = self.all_to_all_combine(outputs, x.shape, weights, indices)
        return result

    def auxiliary_loss(self, x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
        """负载均衡 auxiliary loss"""
        logits = self.gate(x)
        probs = logits.softmax(dim=-1)
        # 每个专家被选中的比例
        freq = (probs > 0).float().mean(dim=0)  # 粗略
        # 每个专家的平均概率
        mean_prob = probs.mean(dim=0)
        # aux loss = n_experts * sum(freq_i * mean_prob_i)
        return alpha * self.n_experts * (freq * mean_prob).sum()


if __name__ == "__main__":
    torch.manual_seed(42)
    dim, n_experts, n_local, top_k = 64, 8, 4, 2
    moe = ExpertParallelMoE(dim, n_experts, n_local, top_k)
    x = torch.randn(2, 10, dim)
    out = moe(x)
    aux = moe.auxiliary_loss(x)
    assert out.shape == x.shape, f"输出形状应保持: {out.shape} vs {x.shape}"
    assert aux.item() >= 0, "aux loss 应非负"
    print(f"输入: {x.shape} → 输出: {out.shape}")
    print(f"Auxiliary loss: {aux.item():.4f}")
    print("✅ Expert Parallel MoE 验证通过")