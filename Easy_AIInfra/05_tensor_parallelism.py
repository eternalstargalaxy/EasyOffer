"""
【题目】张量并行（Tensor Parallelism）

【背景】
单层权重太大放不下一卡时，按维度切到多卡，卡间通信藏在层内。Megatron 两个基本积木：
- 列并行（ColumnParallel）：权重按输出维切，Y = X·W = [X·W1, X·W2]，各卡算自己那部分输出；
  后接逐元素非线性（GeLU）时无需通信，可与下一层行并行无缝衔接。
- 行并行（RowParallel）：权重按输入维切，Y = X·W = X1·W1 + X2·W2，各卡部分和需 all-reduce。
组合 ColumnParallel→GeLU→RowParallel（即 MLP）全程只在最后做一次 all-reduce。

【输入/输出】
- 输入：X: Tensor[B, in_dim]，分布在 TP 组各卡（行并行时 X 已按 in_dim 切分）
- 输出：Y: Tensor[B, out_dim]，列并行各卡持部分输出；行并行 all-reduce 后各卡持完整输出

【考察点】
- 切分维度选择与通信点最小化
- 行并行前向 all-reduce ↔ 反向 split 的对称性
- Embedding 按 vocab 维切 + all-reduce
- 提示：torch.distributed.all_reduce 用于行并行输出聚合

"""
import torch
import torch.nn as nn
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    """W 按输出维切：本卡持 W_i [in_dim, out_dim/N]，输出 [B, out_dim/N]"""
    def __init__(self, in_dim, out_dim, tp_size, rank):
        super().__init__()
        # TODO: 本地权重形状 [in_dim, out_dim//tp_size]
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 各卡独立算，无通信
        raise NotImplementedError


class RowParallelLinear(nn.Module):
    """W 按输入维切：本卡持 W_i [in_dim/N, out_dim]，输入 [B, in_dim/N]，输出需 all-reduce"""
    def __init__(self, in_dim, out_dim, tp_size, rank):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 部分和 -> dist.all_reduce -> 完整输出
        raise NotImplementedError


class TPMLP(nn.Module):
    """ColumnParallelLinear -> GeLU -> RowParallelLinear，全程一次 all-reduce"""
    def __init__(self, dim, hidden, tp_size, rank):
        super().__init__()
        # TODO: 组合上面两块
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class VocabParallelEmbedding(nn.Module):
    """按 vocab 维切 embedding，前向按 token 路由到持有该 vocab 段的卡，最后 all-reduce"""
    def __init__(self, vocab_size, dim, tp_size, rank):
        super().__init__()
        raise NotImplementedError

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# ===== 测试验证 =====
if __name__ == "__main__":
    print("05_tensor_parallelism.py 测试代码：")
    try:
        # TODO: 用户实现后可在此调用核心函数验证输出形状与性质
        pass
        print("✅ 待实现核心函数后运行验证")
    except NotImplementedError:
        print("ℹ 核心函数待实现，可先阅读文件头部背景理解原理")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
