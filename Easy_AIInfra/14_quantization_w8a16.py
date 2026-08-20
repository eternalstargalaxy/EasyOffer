"""
【题目】W8A16 线性层量化推理

【背景】
推理时把权重压成 int8（或 int4），激活保持 fp16，反量化后做 fp16 GEMM。
省显存、减权重带宽，对 decode（memory-bound，权重读一次只算 1 token）特别友好。
推理只量化权重不量化激活，因为激活逐 token 变化且对精度敏感，量化收益小风险大。
对称量化：W_int = round(W/scale)，scale = max(|W|)/127；非对称再加 zero-point。

【输入/输出】
- 输入：W: Tensor[out_dim, in_dim] (fp16/fp32), x: Tensor[B, in_dim] (fp16)
- 输出：W8A16Linear(x) ≈ x @ W^T + bias，权重以 int8 + scale 存储

【考察点】
- scale 维度选择（per-tensor/channel/group）对精度影响
- 反量化时机（是否融合进 GEMM epilogue）
- 对称 vs 非对称、int8 vs int4
- 提示：torch.quantize_per_tensor / per_channel 做动态量化

"""
import torch
import torch.nn as nn


def quantize_w8a16(W: torch.Tensor, granularity: str = "per_channel"):
    """
    W: [out_dim, in_dim]
    per_channel: scale[out_dim] = max(|W|, dim=1)/127
    per_tensor:  scale 标量
    per_group:   每 group_size 一组 scale
    返回 W_int8 (int8), scale
    """
    raise NotImplementedError


class W8A16Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, granularity: str = "per_channel"):
        # TODO: 注册 int8 权重 buffer + scale，不存 fp 权重
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """W = W_int8.to(fp16) * scale; return x @ W^T + bias"""
        raise NotImplementedError


def compare_error(W: torch.Tensor, x: torch.Tensor):
    """返回 per_tensor/per_channel/per_group 三种粒度下与 fp16 的最大误差与显存"""
    raise NotImplementedError

# ===== 测试验证 =====
if __name__ == "__main__":
    print("14_quantization_w8a16.py 测试代码：")
    try:
        # TODO: 用户实现后可在此调用核心函数验证输出形状与性质
        pass
        print("✅ 待实现核心函数后运行验证")
    except NotImplementedError:
        print("ℹ 核心函数待实现，可先阅读文件头部背景理解原理")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
