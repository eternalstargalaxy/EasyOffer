"""
【题目】2:4 结构化稀疏

【背景】
NVIDIA Ampere+ GPU 支持 2:4 结构化稀疏：每 4 个连续值中恰好有 2 个非零。
稀疏化后存储减半，配合 Sparse Tensor Core 加速推理(理论 2x)。
做法：对权重矩阵每 4 个元素一组，保留绝对值最大的 2 个，其余置 0，
然后以压缩格式存储(CSC/CSR)。细粒度裁剪后需少量微调恢复精度。
应用：推理时 weight pruning -> sparse_matmul，训练时可用 STE 反向传播。

【输入/输出】
- 输入：weight [out,in]，sparsity_pattern
- 输出：sparse_weight(压缩格式)，mask [out,in]

【考察点】
- 2:4 pattern 约束 vs 非结构化稀疏的加速区别
- CSC/CSR 压缩格式存储
- 提示：torch.view 重排形状，torch.topk 选最大 2 个
"""
import torch


def prune_24_sparsity(weight: torch.Tensor) -> tuple:
    """每 4 个连续值保留 abs 最大的 2 个，其余置 0，返回 mask"""
    raise NotImplementedError


def compress_24_sparse(weight: torch.Tensor, mask: torch.Tensor):
    """压缩为 2:4 格式：(values, indices)，存储减半"""
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    W = torch.randn(32, 64)
    try:
        mask, sp_w = prune_24_sparsity(W)
        for r in range(4):
            for c in range(0, 64, 4):
                chunk = mask[r, c:c+4]
                assert chunk.sum().item() == 2, f"not 2:4 at [{r},{c}]"
        sparsity = 1 - mask.float().mean().item()
        assert abs(sparsity - 0.5) < 0.01, f"sparsity: {sparsity}"
        print(chr(0x2705) + " 2:4 稀疏化测试通过")
    except NotImplementedError:
        print(chr(0x2139) + " 待实现")
