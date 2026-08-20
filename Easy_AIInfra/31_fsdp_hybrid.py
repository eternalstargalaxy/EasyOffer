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
import torch.distributed as dist


def hybrid_fsdp_setup(model: nn.Module, local_rank: int,
                       local_world: int, global_world: int):
    raise NotImplementedError


def cpu_offload_step(optimizer, gpu_params, cpu_states):
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    print('ℹ' + " Hybrid FSDP 需多节点分布式环境")
    print("验证：节点内 all_gather 走 NVLink，节点间走 IB")
