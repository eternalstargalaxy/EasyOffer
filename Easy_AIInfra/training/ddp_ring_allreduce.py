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
- 提示：torch.distributed.init_process_group(backend, rank, world_size) 初始化分布式；torch.distributed.all_reduce 执行规约

"""
import torch
import torch.distributed as dist


def ring_all_reduce(tensor: torch.Tensor, rank: int, world_size: int):
    """
    就地把手卡 tensor 规约为所有卡之和。分 chunk 数 = world_size。
    阶段一 scatter-reduce：N-1 步，每步把本卡某 chunk 累加到右邻居对应 chunk
    阶段二 all-gather：  N-1 步，把已规约完的 chunk 广播给右邻居
    用 dist.send/ dist.recv 模拟（不直接调 dist.all_reduce）。
    """
    raise NotImplementedError


class DDP:
    def __init__(self, model: torch.nn.Module, world_size: int, rank: int):
        self.model = model
        # TODO: 注册 backward hook，梯度 ready 后触发 ring_all_reduce
        #       （可加分桶：按 param 累计字节数凑满 bucket_size 再通信）

    def forward(self, *args):
        return self.model(*args)

    def backward_and_sync(self, loss):
        """
        loss.backward() 后梯度已就位；
        对每个 param.grad 调 ring_all_reduce，再 /= world_size 取均值。
        """
        raise NotImplementedError

# ===== 测试验证 =====
if __name__ == "__main__":
    print("03_ddp_ring_allreduce.py 测试代码：")
    try:
        # TODO: 用户实现后可在此调用核心函数验证输出形状与性质
        pass
        print("✅ 待实现核心函数后运行验证")
    except NotImplementedError:
        print("ℹ 核心函数待实现，可先阅读文件头部背景理解原理")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
