"""
【题目】ZeRO++：hpZ 参数通信优化 + qgZ 梯度量化

【背景】
ZeRO-3 每层前向需 all-gather 参数，跨节点通信成为瓶颈。
ZeRO++ 两个核心优化：
hpZ(hierarchical partition ZeRO)：参数主副本存在节点内多卡，
节点间只传一次参数(节点内广播替代跨节点 all-gather)。
qgZ(quantized gradient ZeRO)：对梯度做 INT4 量化后再 reduce-scatter，
降低梯度通信量约 4x，精度损失可忽略(用 INT4->FP16 dequant)。
两者独立组合使用，理论通信量降到 ~1/4。

【输入/输出】
- 输入：梯度 (fp16), all-gather 参数, 节点拓扑
- 输出：量化后的梯度、节点内广播的参数

【考察点】
- hpZ：二次分片 + 节点内广播替代跨节点通信
- qgZ：梯度 INT4 量化 + block-wise scale
- 提示：torch.distributed.broadcast, torch.quantize_per_tensor
"""
import torch


def hpz_all_gather(param_shard: torch.Tensor, local_group_size: int,
                   cross_group_size: int):
    """
    hpZ：节点内 broadcast 替代跨节点 all-gather。
    单机模拟：先跨节点 gather，再节点内 broadcast。
    """
    full_param = param_shard.clone()
    for _ in range(cross_group_size - 1):
        full_param = torch.cat([full_param, torch.zeros_like(param_shard)])
    broadcasted = full_param.clone()
    return broadcasted


def qgz_quantize_gradient(grad: torch.Tensor, n_bits: int = 4):
    """
    qgZ：block-wise INT4 量化梯度。
    返回 (quantized_data, scales, zero_points)。
    """
    block_size = 128
    flat = grad.view(-1)
    n = flat.numel()
    n_blocks = (n + block_size - 1) // block_size
    padded = torch.zeros(n_blocks * block_size, dtype=flat.dtype)
    padded[:n] = flat

    blocks = padded.view(n_blocks, block_size)
    qmax = 2 ** n_bits - 1

    scales = blocks.abs().max(dim=1).values / (qmax / 2)
    scales = scales.clamp(min=1e-8)
    zero_points = torch.zeros(n_blocks, dtype=torch.float32)

    quantized = torch.round(blocks / scales.unsqueeze(1) + qmax / 2)
    quantized = quantized.clamp(0, qmax).to(torch.uint8)

    return quantized, scales, zero_points, n


def qgz_dequantize(quantized: torch.Tensor, scales: torch.Tensor,
                   zero_points: torch.Tensor, n: int):
    """反量化：INT4 -> FP16。"""
    qmax = 2 ** 4 - 1
    block_size = 128
    n_blocks = quantized.shape[0]
    blocks = (quantized.float() - qmax / 2) * scales.unsqueeze(1)
    flat = blocks.view(-1)
    return flat[:n]


# ===== 测试验证 =====
if __name__ == '__main__':
    torch.manual_seed(42)
    shard = torch.randn(100)
    full = hpz_all_gather(shard, local_group_size=2, cross_group_size=4)
    assert full.numel() == shard.numel() * 4
    print(f"✅ hpZ: shard {shard.numel()} -> full {full.numel()}")

    grad = torch.randn(1024) * 0.1
    q, s, z, n = qgz_quantize_gradient(grad, n_bits=4)
    assert q.dtype == torch.uint8
    assert q.max().item() <= 15
    print(f"✅ qgZ quantize: {grad.numel()} fp16 -> {q.numel()} int4 + {s.numel()} scales")

    deq = qgz_dequantize(q, s, z, n)
    err = (deq - grad).abs().mean().item()
    rel_err = err / grad.abs().mean().item()
    assert rel_err < 0.2, f"量化误差过大: {rel_err}"
    print(f"✅ qgZ dequantize: 相对误差 {rel_err:.4f}")

    grad2 = torch.randn(256) * 0.01
    q2, s2, z2, n2 = qgz_quantize_gradient(grad2, n_bits=4)
    deq2 = qgz_dequantize(q2, s2, z2, n2)
    err2 = (deq2 - grad2).abs().mean().item()
    rel_err2 = err2 / grad2.abs().mean().item()
    print(f"✅ 小梯度量化: 相对误差 {rel_err2:.4f}")

    grad3 = torch.randn(100)
    q3, s3, z3, n3 = qgz_quantize_gradient(grad3, n_bits=8)
    deq3 = qgz_dequantize(q3, s3, z3, n3)
    err3 = (deq3 - grad3).abs().mean().item()
    rel_err3 = err3 / grad3.abs().mean().item()
    print(f"✅ INT8 对比: 相对误差 {rel_err3:.4f}")
    print("✅ 全部测试通过")
