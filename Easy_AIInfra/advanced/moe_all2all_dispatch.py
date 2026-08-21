"""
【题目】MoE All-to-All Dispatch

【背景】
MoE 模型中每个 token 被路由到不同 expert，expert 分布在多卡上。
All-to-All dispatch：各卡把本地 token 按 expert 归类后发给持有该 expert 的卡，
计算完再 All-to-All 发回。核心通信原语：dispatch (send) + combine (recv)。
优化：分组 All-to-All、overlap 通信与计算、zero-bubble dispatch。

【输入/输出】
- 输入：tokens [N, D], routing [N, expert_id], num_experts, num_workers
- 输出：dispatch 后各 expert 的 token 分组

【考察点】
- All-to-All 通信模式与 token 归类
- dispatch/combine 对称性
- 提示：按 expert_id 分组
"""
import torch
import torch.nn as nn
from collections import defaultdict


def moe_dispatch(tokens: torch.Tensor, routing: torch.Tensor, num_experts: int) -> tuple:
    """按 routing 把 token 分发到各 expert。"""
    expert_tokens = defaultdict(list)
    expert_indices = defaultdict(list)
    for i, expert_id in enumerate(routing.tolist()):
        expert_tokens[expert_id].append(tokens[i])
        expert_indices[expert_id].append(i)
    return expert_tokens, expert_indices


def moe_combine(expert_outputs: dict, expert_indices: dict, num_tokens: int, dim: int) -> torch.Tensor:
    """把各 expert 输出按原顺序合并。"""
    output = torch.zeros(num_tokens, dim)
    for expert_id, outs in expert_outputs.items():
        for j, out in enumerate(outs):
            orig_idx = expert_indices[expert_id][j]
            output[orig_idx] = out
    return output


class MoELayer(nn.Module):
    def __init__(self, dim: int, num_experts: torch.Tensor, capacity: int = 4):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity = capacity
        self.experts = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, tokens: torch.Tensor) -> tuple:
        gate_logits = self.gate(tokens)
        routing = torch.argmax(gate_logits, dim=-1)
        expert_tokens, expert_indices = moe_dispatch(tokens, routing, self.num_experts)
        expert_outputs = {}
        for eid, toks in expert_tokens.items():
            if toks:
                stacked = torch.stack(toks)
                expert_outputs[eid] = self.experts[eid](stacked).unbind(0)
        output = moe_combine(expert_outputs, expert_indices, tokens.shape[0], self.dim)
        return output, routing


def all_to_all_simulate(local_tokens: torch.Tensor, local_routing: torch.Tensor, num_workers: torch.Tensor) -> list:
    """模拟 All-to-All：各卡 token 按 expert 归类发到对应卡。"""
    dispatched = [[] for _ in range(num_workers)]
    for i, eid in enumerate(local_routing.tolist()):
        target_worker = eid % num_workers
        dispatched[target_worker].append(local_tokens[i])
    return dispatched


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    N, D, E = 8, 16, 4
    tokens = torch.randn(N, D)
    routing = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    et, ei = moe_dispatch(tokens, routing, E)
    assert sum(len(v) for v in et.values()) == N
    print(f"✅ Dispatch: {N} tokens -> {len(et)} experts")

    dummy_outs = {eid: [t * 2 for t in toks] for eid, toks in et.items()}
    combined = moe_combine(dummy_outs, ei, N, D)
    assert combined.shape == (N, D)
    assert torch.allclose(combined, tokens * 2)
    print("✅ Combine: 还原正确")

    moe = MoELayer(D, E)
    out, r = moe(tokens)
    assert out.shape == (N, D)
    assert r.shape == (N,)
    print(f"✅ MoELayer forward: {tokens.shape} -> {out.shape}")

    dispatched = all_to_all_simulate(tokens, routing, num_workers=4)
    assert sum(len(d) for d in dispatched) == N
    print("✅ All-to-All simulate: 4 workers")
    print("✅ 全部测试通过")
