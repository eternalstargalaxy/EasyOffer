"""
【题目】MoE 负载均衡

【背景】
MoE 训练中 token 路由不均衡会导致部分 expert 过载(capacity overflow)、
部分空闲，降低有效容量。负载均衡方法：
1. Auxiliary Loss：鼓励均匀路由，L_aux = alpha * sum(f_i * P_i)
2. Capacity Factor：动态调整每个 expert 的容量
3. Token Dropping：超容量 token 丢弃或传给次优 expert
4. DeepSeek 的 aux-loss-free 方法：直接在路由权重上加偏置

【输入/输出】
- 输入：routing weights [N, E], expert_capacity
- 输出：均衡损失 + 调整后的路由

【考察点】
- auxiliary loss 公式与梯度
- capacity overflow 处理
- 提示：f_i = 实际路由比例, P_i = 平均路由概率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def auxiliary_loss(routing_weights: torch.Tensor, routing_indices: torch.Tensor,
                   num_experts: int, alpha: float = 0.01) -> tuple:
    """
    Switch Transformer auxiliary loss。
    routing_weights: [N, E] softmax 概率
    routing_indices: [N] 实际路由到的 expert
    L_aux = alpha * E * sum(f_i * P_i)
    f_i = 该 expert 被选中的 token 比例
    P_i = 该 expert 的平均路由概率
    """
    N = routing_weights.shape[0]
    f = torch.zeros(num_experts)
    P = torch.zeros(num_experts)
    for i in range(num_experts):
        f[i] = (routing_indices == i).float().mean()
        P[i] = routing_weights[:, i].mean()
    loss = alpha * num_experts * (f * P).sum()
    return loss, f, P


def capacity_check(routing_indices: torch.Tensor, num_experts: int,
                   capacity: int) -> tuple:
    """检查各 expert 是否超容量。"""
    overflow = torch.zeros(num_experts, dtype=torch.bool)
    counts = torch.zeros(num_experts, dtype=torch.long)
    for i in range(num_experts):
        counts[i] = (routing_indices == i).sum()
        overflow[i] = counts[i] > capacity
    return overflow, counts


def token_dropping(routing_weights: torch.Tensor, routing_indices: torch.Tensor,
                   num_experts: int, capacity: int) -> torch.Tensor:
    """超容量 token 丢弃。"""
    expert_count = torch.zeros(num_experts, dtype=torch.long)
    keep_mask = torch.ones(routing_indices.shape[0], dtype=torch.bool)
    for i in range(routing_indices.shape[0]):
        eid = routing_indices[i].item()
        if expert_count[eid] >= capacity:
            keep_mask[i] = False
        else:
            expert_count[eid] += 1
    return keep_mask


def deepseek_balance_bias(routing_weights: torch.Tensor, expert_load: torch.Tensor,
                          bias_lr: float = 0.001) -> torch.Tensor:
    """DeepSeek aux-loss-free：用偏置调整路由。"""
    target_load = expert_load.mean()
    bias = bias_lr * (expert_load - target_load)
    return routing_weights + bias.unsqueeze(0)


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    N, E = 100, 4
    weights = F.softmax(torch.randn(N, E), dim=-1)
    indices = torch.randint(0, E, (N,))

    loss, f, P = auxiliary_loss(weights, indices, E)
    assert loss >= 0
    assert f.sum().item() == 1.0
    assert P.sum().item() == 1.0
    print(f"✅ Auxiliary loss: {loss.item():.6f}, f={f.tolist()}, P={P.tolist()}")

    overflow, counts = capacity_check(indices, E, capacity=30)
    print(f"✅ Capacity check: counts={counts.tolist()}, overflow={overflow.tolist()}")

    keep = token_dropping(weights, indices, E, capacity=30)
    kept = keep.sum().item()
    assert kept <= 30 * E
    print(f"✅ Token dropping: {N} -> {kept} (capacity=30*{E})")

    expert_load = torch.tensor([40.0, 10.0, 30.0, 20.0])
    adjusted = deepseek_balance_bias(weights, expert_load)
    assert adjusted.shape == weights.shape
    print("✅ DeepSeek balance bias: 调整路由权重")

    indices_balanced = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss_b, f_b, _ = auxiliary_loss(weights[:8], indices_balanced, E)
    assert f_b[0].item() == 0.25
    print("✅ 均衡路由: f_i = 0.25")
    print("✅ 全部测试通过")
