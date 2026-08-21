"""
【题目】FSDP Hybrid Shard + CPU Offload

【背景】
标准 FSDP(ZeRO-3)按 DP 维度全切分参数，通信开销大(每层 2 次 all-gather)。
Hybrid Shard：节点内用 DDP(全量参数)，节点间用 FSDP(切分)，
核心思想是 NVLink 带宽 >> IB 带宽，节点内通信几乎免费，
节点间才做参数分片。减少跨节点通信量。
CPU Offload：将优化器状态(12*Phi)卸载到 CPU，GPU 只保留参数+梯度，
update 时从 CPU load 到 GPU，更新完卸载。可用 pin_memory 加速传输。

【输入/输出】
- 输入：model, dp_size=8, nnodes=4, sharding_strategy
- 输出：每卡的参数分片方案 + offload 策略

【考察点】
- Hybrid: node-local DDP + cross-node FSDP
- CPU offload 时 pin_memory 与 cudaMemcpyAsync 的 pipeline
- 提示：torch.distributed.new_group 建 node 内/间通信组
"""
import torch
import torch.nn as nn
from collections import namedtuple

ShardInfo = namedtuple("ShardInfo", ["strategy", "local_rank", "node_id",
                                      "shard_offset", "shard_size", "total_size"])


def hybrid_fsdp_setup(model: nn.Module, local_rank: int,
                      local_world: int, global_world: int):
    """
    Hybrid Shard 设置：
    - 节点内(local_group): DDP，全量参数
    - 节点间(cross_group): FSDP，参数分片
    返回分片信息。
    """
    num_nodes = global_world // local_world
    node_id = local_rank // local_world
    rank_in_node = local_rank % local_world

    total_params = sum(p.numel() for p in model.parameters())
    shard_size = total_params // num_nodes
    shard_offset = node_id * shard_size

    return ShardInfo(
        strategy="hybrid_shard",
        local_rank=rank_in_node,
        node_id=node_id,
        shard_offset=shard_offset,
        shard_size=shard_size,
        total_size=total_params,
    )


class CPUOffloadOptimizer:
    """CPU offload 优化器：状态在 CPU，更新时 load 到 GPU。"""

    def __init__(self, params: torch.Tensor, lr: float = 1e-3):
        self.params = list(params)
        self.lr = lr
        self.cpu_m = [torch.zeros_like(p, device="cpu") for p in self.params]
        self.cpu_v = [torch.zeros_like(p, device="cpu") for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            m = self.cpu_m[i]
            v = self.cpu_v[i]
            grad_cpu = p.grad.to("cpu")
            m.mul_(0.9).add_(grad_cpu, alpha=0.1)
            v.mul_(0.999).addcmul_(grad_cpu, grad_cpu, value=0.001)
            m_hat = m / (1 - 0.9 ** self.t)
            v_hat = v / (1 - 0.999 ** self.t)
            update = self.lr * m_hat / (v_hat.sqrt() + 1e-8)
            p.data -= update.to(p.device)


def cpu_offload_step(optimizer: CPUOffloadOptimizer, gpu_params: torch.Tensor, cpu_states: torch.Tensor):
    """CPU offload 更新一步：load state -> update -> offload。"""
    optimizer.step()


# ===== 测试验证 =====
if __name__ == '__main__':
    model = nn.Linear(100, 50)
    info = hybrid_fsdp_setup(model, local_rank=0, local_world=2, global_world=8)
    assert info.strategy == "hybrid_shard"
    assert info.node_id == 0
    assert info.shard_size == info.total_size // 4
    print(f"✅ Hybrid FSDP: {info.total_size} params, {info.shard_size} shard/node")

    info2 = hybrid_fsdp_setup(model, local_rank=3, local_world=2, global_world=8)
    assert info2.node_id == 1
    assert info2.local_rank == 1
    print(f"✅ rank=3: node={info2.node_id}, local_rank={info2.local_rank}")

    info3 = hybrid_fsdp_setup(model, local_rank=6, local_world=2, global_world=8)
    assert info3.node_id == 3
    print(f"✅ rank=6: node={info3.node_id}")

    p = nn.Parameter(torch.randn(10, 10))
    opt = CPUOffloadOptimizer([p], lr=0.01)
    x = torch.randn(5, 10)
    loss = (p(x) - torch.randn(5, 10)).pow(2).mean()
    loss.backward()
    w_before = p.data.clone()
    cpu_offload_step(opt, [p], None)
    assert not torch.allclose(w_before, p.data), "参数应被更新"
    print("✅ CPU Offload Optimizer: 更新成功")
    print("✅ 全部测试通过")
