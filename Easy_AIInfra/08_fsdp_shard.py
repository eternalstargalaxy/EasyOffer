"""
【题目】FSDP 参数分片（FlatShard + all-gather / reduce-scatter）

【背景】
FSDP 是 ZeRO-3 的 PyTorch 原生实现：把模块所有参数拼成一个 flat buffer，按 rank 分片持久存储，
平时只持有本 shard。前向/反向前 all-gather 拼出完整参数挂回模块，算完释放非本 shard 副本；
反向后 reduce-scatter 梯度，只更新本 shard，再 all-gather 把新权重同步给所有卡。
可加 pre-fetch（提前一层 all-gather）隐藏通信，可 CPU offload 把本 shard 放 CPU。

【输入/输出】
- 输入：module, dp_size=N, rank
- 输出：每卡持久只存 flat param 的 1/N，前向/反向按需 materialize/释放

【考察点】
- flat param 拼包/解包与原 param 的 .data 视图同步
- all-gather（前向/反向前） / reduce-scatter（反向后）交错
- pre-fetch 与 CPU offload
"""
import torch
import torch.nn as nn
import torch.distributed as dist


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, dp_size: int, rank: int,
                 prefetch: bool = False, cpu_offload: bool = False):
        super().__init__()
        # TODO:
        #   1. 收集 module 所有 param，拼成 flat_full
        #   2. 本卡只存 flat_shard = flat_full[rank::N]（可 offload 到 CPU）
        #   3. 记录每个原 param 在 flat 中的 (offset, size) 以便恢复视图
        #   4. 注册前向 pre-hook / post-hook
        raise NotImplementedError

    def _all_gather_full_param(self):
        """dist.all_gather 拼出完整 flat，按 offset 还原各 param.data 视图"""
        raise NotImplementedError

    def _release_non_shard(self):
        """释放非本 shard 的参数副本（置空或只留 shard）"""
        raise NotImplementedError

    def forward(self, *args):
        # all-gather -> forward -> release
        raise NotImplementedError

    def backward_step(self, loss):
        """
        loss.backward() 后：
          1. reduce-scatter 梯度得本 shard 梯度
          2. 用本 shard 梯度更新本 shard 参数
          3. all-gather 把更新后权重同步给所有卡
        """
        raise NotImplementedError
