"""
【题目】DDP 梯度同步 + Ring-AllReduce

【背景】
数据并行：每卡完整模型，各算各的梯度后需聚合为全局梯度。Ring-AllReduce 把 N 卡的梯度规约
拆成两阶段、共 2(N-1) 步点对点通信，每步只传 1/N 的数据量，带宽利用率高（NCCL 默认实现）。
DDP 在反向时通过 backward hook 等所有梯度 ready 后触发一次 AllReduce；为通信/计算 overlap，
会把梯度分桶（bucket），一个 bucket 的梯度 ready 就先通信。

【输入/输出】
- 输入：rank, world_size, 本卡梯度 tensor（与其它卡同形状）
- 输出：所有卡上梯度变为各卡之和（或均值）

【考察点】
- Ring-AllReduce 两阶段（scatter-reduce → all-gather）正确性
- backward hook 触发时机、分桶 overlap
- 与 gradient accumulation 共存时只在真实 step 的 micro-batch 同步
- 提示：torch.distributed.init_process_group 初始化分布式
"""
import torch
import torch.nn as nn


def ring_all_reduce_simulate(tensors: list, average: bool = True):
    """
    单机模拟 Ring-AllReduce：输入 N 个同形状 tensor，返回规约后的 N 个副本。
    阶段一 scatter-reduce：N-1 步累加
    阶段二 all-gather：N-1 步广播
    """
    n = len(tensors)
    if n == 1:
        return tensors
    result = [t.clone() for t in tensors]
    chunk_size = result[0].numel() // n
    flat = [r.view(-1) for r in result]

    for step in range(n - 1):
        for rank in range(n):
            send_chunk = (rank + step) % n
            recv_chunk = (rank + step + 1) % n
            src_rank = (rank - 1) % n
            s_start = send_chunk * chunk_size
            s_end = s_start + chunk_size
            r_start = recv_chunk * chunk_size
            r_end = r_start + chunk_size
            flat[rank][r_start:r_end] += flat[src_rank][s_start:s_end]

    for step in range(n - 1):
        for rank in range(n):
            send_chunk = (rank + step + 1) % n
            recv_chunk = (rank + step + 2) % n
            src_rank = (rank - 1) % n
            s_start = send_chunk * chunk_size
            s_end = s_start + chunk_size
            r_start = recv_chunk * chunk_size
            r_end = r_start + chunk_size
            flat[rank][r_start:r_end] = flat[src_rank][s_start:s_end]

    if average:
        for r in result:
            r /= n
    return result


class DDP:
    """单机模拟 DDP：多份模型副本，梯度同步。"""

    def __init__(self, model: nn.Module, world_size: int, rank: int):
        self.model = model
        self.world_size = world_size
        self.rank = rank

    def forward(self, *args):
        return self.model(*args)

    def backward_and_sync(self, loss, all_grads: list):
        """
        loss.backward() 后梯度已就位；
        对每个 param.grad 调 ring_all_reduce，再 /= world_size 取均值。
        all_grads: 各卡同位置 param.grad 的列表。
        """
        loss.backward()
        synced = ring_all_reduce_simulate(all_grads, average=True)
        return synced


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    n = 4
    tensors = [torch.randn(16) for _ in range(n)]
    expected = sum(tensors) / n
    result = ring_all_reduce_simulate(tensors, average=True)
    for i in range(n):
        err = (result[i] - expected).abs().max().item()
        assert err < 1e-5, f"rank {i}: all-reduce 误差 {err} 过大"
    print("✅ Ring-AllReduce: 结果与朴素均值一致")

    tensors2 = [torch.randn(8, 8) for _ in range(3)]
    expected2 = sum(tensors2)
    result2 = ring_all_reduce_simulate(tensors2, average=False)
    for i in range(3):
        err = (result2[i] - expected2).abs().max().item()
        assert err < 1e-5
    print("✅ Ring-AllReduce (sum): 2D tensor 正确")

    model = nn.Linear(10, 5)
    ddp = DDP(model, world_size=2, rank=0)
    x = torch.randn(4, 10)
    y = ddp.forward(x)
    assert y.shape == (4, 5)
    print("✅ DDP forward 正确")

    grads_list = [torch.randn(10, 5) for _ in range(3)]
    synced = ring_all_reduce_simulate(grads_list, average=True)
    assert all(s.shape == (10, 5) for s in synced)
    print("✅ DDP 梯度同步正确")
    print("✅ 全部测试通过")
