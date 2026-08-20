"""
【题目】混合架构：Jamba / Mamba-2 Hybrid (SSM + Attention + MoE)

【背景】
Jamba(AI21,2024)：交替堆叠 Transformer 和 Mamba 层。
Transformer 提供高精度 in-context reasoning，Mamba 提供高效长序列。
典型：每 1 Attn + 7 Mamba 为一 block，MLP 中引入 MoE(top-2/16)。
Mamba-2 hybrid: SSM 层间加 sliding-window attention，
利用 SSM 长程 + attention 精确召回，继承两者优势。

【输入/输出】
- 输入：x [B,L,D], config(每层是 mamba/attn)
- 输出：混合层处理后的序列特征

【考察点】
- 层比例与精度/速度 trade-off
- SSM 作为 attention 替代的作用
- 提示：nn.ModuleList 存异构层
"""
import torch; import torch.nn as nn


class HybridBlock(nn.Module):
    def __init__(self, dim: int, use_mamba: bool, use_moe: bool = False):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.is_mamba = use_mamba
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    B, L, D = 2, 64, 32
    try:
        b = HybridBlock(D, use_mamba=False)
        y = b(torch.randn(B, L, D))
        assert y.shape == (B, L, D)
        print('✅' + " Hybrid 测试通过")
    except NotImplementedError:
        print('ℹ' + " 待实现")
