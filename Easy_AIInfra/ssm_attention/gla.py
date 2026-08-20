"""
【题目】GLA：门控线性注意力 (Gated Linear Attention)

【背景】
GLA(2024)是线性注意力的升级：加 data-dependent gating 做遗忘控制。
核心：gate_t = sigmoid(W_g x_t + b_g)，累加时对历史做逐通道 gate：
S_t = gate_t * S_{t-1} + K_t^T V_t, O_t = Q_t S_t
优势：gate 由数据动态决定遗忘量，克服线性注意力"无遗忘"的局限。
与 Mamba-2 联系：GLA 的门控 SSM 核 = Mamba-2 的 semiseparable 核。
chunkwise 并行：每块内并行矩阵乘，块间递归传递 S，O(n^2/C) 显存。

【输入/输出】
- 输入：Q,K,V [B,L,D], chunk_size
- 输出：O [B,L,D]

【考察点】
- data-dependent gate vs Mamba selective SSM 统一视角
- chunkwise 递归 vs 全并行 trade-off
- 提示：torch.sigmoid 做 gate；chunk 内用矩阵乘，块间用 scan
"""
import torch; import torch.nn as nn


def gla_chunkwise(Q, K, V, gate, chunk_size: int = 64):
    raise NotImplementedError


class GALayer(nn.Module):
    def __init__(self, dim: int, head_dim: int = 64, chunk_size: int = 64):
        super().__init__()
        self.dim = dim; self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.gate_proj = nn.Linear(dim, self.num_heads, bias=True)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    B, L, D = 2, 128, 64
    try:
        m = GALayer(D)
        y = m(torch.randn(B, L, D))
        assert y.shape == (B, L, D)
        print('✅' + " GLA 测试通过")
    except NotImplementedError:
        print('ℹ' + " 待实现")
