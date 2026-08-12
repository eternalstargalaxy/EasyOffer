"""
【题目】ZeRO-1 / ZeRO-2 / ZeRO-3 状态分片

【背景】
DDP 下每卡冗余持有全部 optimizer state / grad / param。ZeRO 依次把它们在数据并行组间分片：
- ZeRO-1：分片 optimizer state（Adam 的 m/v）→ 优化器状态 4x 降为 4x/N
- ZeRO-2：再分片梯度（reduce-scatter 后各卡只留自己那片）→ 再降 2x/N
- ZeRO-3：再分片参数（前向/反向前 all-gather 出完整参数，用完释放）→ 再降 param/N
更新只更新本 shard，ZeRO-3 更新后需 all-gather 把新权重同步给所有卡。

【输入/输出】
- 输入：model, optimizer, data_parallel_size=N, rank
- 输出：每卡只持有 param/grad/state 的 1/N，训练正常推进

【考察点】
- reduce-scatter / all-gather 与 step 的交错顺序
- ZeRO-3 前向也需 all-gather（参数不全算不了）
- 三种方式的显存公式与通信量 trade-off
"""
import torch
import torch.nn as nn
import torch.distributed as dist


class ShardedAdam:
    """ZeRO-1：只为本 param shard 维护 m/v"""
    def __init__(self, params_shard, lr=1e-3, betas=(0.9, 0.999)):
        # TODO: 只为传入的 param shard 分配 m/v
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


def zero2_step(grad_full):
    """reduce-scatter 梯度，返回本卡 shard（1/N）"""
    # TODO: dist.reduce_scatter
    raise NotImplementedError


class Zero3:
    def __init__(self, model: nn.Module, dp_size: int, rank: int):
        # TODO: 把所有 param 拼成 flat buffer，按 rank 分片，平时只持有本 shard
        raise NotImplementedError

    def gather_param(self):
        """前向/反向前 all-gather 出完整参数，挂回模块"""
        # TODO: dist.all_gather
        raise NotImplementedError

    def release_param(self):
        """计算完释放非本 shard 副本"""
        raise NotImplementedError

    def forward(self, *args):
        # gather -> forward -> release
        raise NotImplementedError

    def backward_step(self, loss):
        # reduce-scatter grad -> 更新本 shard -> all-gather 同步权重
        raise NotImplementedError


def mem_formula(N, param_cnt, grad_cnt, state_cnt):
    """返回 ZeRO-1/2/3 单卡显存（以元素数计）"""
    raise NotImplementedError
