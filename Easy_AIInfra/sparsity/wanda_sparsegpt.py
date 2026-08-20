"""
【题目】Wanda / SparseGPT：非结构化权重剪枝

【背景】
Wanda: 用 abs(W) * norm(X) 作为重要性，一次性剪掉最小 sparsity_ratio 权重，
无需梯度，仅需一次校准数据前向。剪枝后不微调也能保持精度。
SparseGPT: 二阶 Hessian 逐列剪枝并补偿误差，类似 GPTQ 量化框架。
每列保留 top-k，用逆 Hessian 修正剩余列。
对比：Wanda 快(一次前向)，SparseGPT 精度更高(二阶补偿)。

【输入/输出】
- 输入：weight [out,in], calibration X [B,in], sparsity_ratio
- 输出：pruned_weight, mask

【考察点】
- Wanda：importance = abs(W) * norm(X)，无需梯度
- SparseGPT：逐列 OBS 剪枝 + Hessian 补偿
- 提示：torch.linalg.cholesky_inverse 计算逆 Hessian, torch.topk 选 top-k
"""
import torch


def wanda_prune(weight: torch.Tensor, X_calib: torch.Tensor,
                sparsity: float = 0.5) -> tuple:
    raise NotImplementedError


def sparsegpt_prune(weight: torch.Tensor, X_calib: torch.Tensor,
                    sparsity: float = 0.5) -> tuple:
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    W = torch.randn(128, 256)
    X = torch.randn(4, 256)
    try:
        pW, mask = wanda_prune(W, X, 0.5)
        sp = 1 - mask.float().mean().item()
        assert abs(sp - 0.5) < 0.02
        print('✅' + f" Wanda sparse={sp:.2%}")
    except NotImplementedError:
        print('ℹ' + " 待实现")
