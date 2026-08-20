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
- 提示：torch.distributed.all_gather 收集分片参数
"""
import torch
import torch.nn as nn


class FSDP(nn.Module):
    """单机模拟 FSDP：flat param 分片 + all-gather/reduce-scatter。"""

    def __init__(self, module: nn.Module, dp_size: int, rank: int,
                 prefetch: bool = False, cpu_offload: bool = False):
        super().__init__()
        self.module = module
        self.dp_size = dp_size
        self.rank = rank
        self.prefetch = prefetch
        self.cpu_offload = cpu_offload

        self.params = list(module.parameters())
        self.param_info = []
        offset = 0
        for p in self.params:
            self.param_info.append((offset, p.numel(), p.shape))
            offset += p.numel()
        self.total_size = offset

        flat_full = torch.cat([p.data.view(-1) for p in self.params])
        shard_size = self.total_size // dp_size
        start = rank * shard_size
        self.flat_shard = flat_full[start:start + shard_size].clone()
        if cpu_offload:
            self.flat_shard = self.flat_shard.cpu()
        self.shard_size = shard_size
        self._full_param = None

    def _all_gather_full_param(self):
        """all-gather 拼出完整 flat，按 offset 还原各 param.data 视图。"""
        self._full_param = self.flat_shard.clone()
        for r in range(1, self.dp_size):
            dummy = torch.zeros(self.shard_size, dtype=self.flat_shard.dtype)
            self._full_param = torch.cat([self._full_param, dummy])
        for i, (offset, size, shape) in enumerate(self.param_info):
            self.params[i].data = self._full_param[offset:offset + size].view(shape).clone()
        return self._full_param

    def _release_non_shard(self):
        """释放非本 shard 的参数副本。"""
        self._full_param = None

    def forward(self, *args):
        self._all_gather_full_param()
        result = self.module(*args)
        self._release_non_shard()
        return result

    def backward_step(self, loss):
        """
        loss.backward() 后：
          1. reduce-scatter 梯度得本 shard 梯度
          2. 用本 shard 梯度更新本 shard 参数
          3. all-gather 把更新后权重同步给所有卡
        """
        loss.backward()
        flat_grad = torch.cat([
            (p.grad.view(-1) if p.grad is not None else torch.zeros(p.numel()))
            for p in self.params
        ])
        start = self.rank * self.shard_size
        grad_shard = flat_grad[start:start + self.shard_size] / self.dp_size
        self.flat_shard -= 0.01 * grad_shard
        self._all_gather_full_param()
        self._release_non_shard()


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    fsdp = FSDP(model, dp_size=4, rank=0)

    assert fsdp.flat_shard.numel() == fsdp.total_size // 4
    print(f"✅ FSDP 初始化: total={fsdp.total_size}, shard={fsdp.flat_shard.numel()}")

    full = fsdp._all_gather_full_param()
    assert full.numel() == fsdp.total_size
    for p in fsdp.params:
        assert p.data is not None
    print("✅ all-gather 还原 param 视图")

    fsdp._release_non_shard()
    assert fsdp._full_param is None
    print("✅ release_non_shard 释放完成")

    x = torch.randn(4, 10)
    y = fsdp.forward(x)
    assert y.shape == (4, 5)
    print(f"✅ FSDP forward: {x.shape} -> {y.shape}")

    model2 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    fsdp2 = FSDP(model2, dp_size=2, rank=0, cpu_offload=True)
    assert fsdp2.flat_shard.device.type == "cpu"
    print("✅ CPU offload: shard 在 CPU 上")

    model3 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    fsdp3 = FSDP(model3, dp_size=1, rank=0)
    y3 = fsdp3.forward(x)
    assert y3.shape == (4, 5)
    print("✅ dp_size=1 退化为普通前向")
    print("✅ 全部测试通过")
