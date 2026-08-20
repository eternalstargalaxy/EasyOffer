"""
【题目】Mamba (S6)：选择性状态空间模型

【背景】
Mamba(2023)首个在语言建模上匹敌 Transformer 的 SSM。
核心：Selective SSM——A,B,C 由输入 x 动态生成，非定常。
离散化(ZOH)：A_bar=exp(dt*A), B_bar=(A_bar-I)A^-1 B
并行 scan 替代 RNN 逐步计算，O(log n)训练。
SSM+gating：y = SSM(Conv1d(x)) * SiLU(gate)。

【输入/输出】
- 输入：x [B,L,D], A_log, dt_proj, x_proj
- 输出：y [B,L,D] SSM+门控后特征

【考察点】
- Selective SSM vs S4 定常 SSM 区别
- ZOH 离散化 + dt projection
- 并行 scan(Blelloch)加速原理
- 提示：torch.exp 做矩阵指数
"""
import torch; import torch.nn as nn; import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.dim = dim; self.d_state = d_state
        self.conv1d = nn.Conv1d(dim, dim, d_conv, groups=dim, padding=d_conv-1)
        self.x_proj = nn.Linear(dim, dim * 2, bias=False)
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        self.A_log = nn.Parameter(torch.randn(dim, d_state))
        self.D = nn.Parameter(torch.randn(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def selective_scan(self, u, delta, A, B, C):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    B, L, D = 2, 64, 32
    try:
        m = MambaBlock(D)
        y = m(torch.randn(B, L, D))
        assert y.shape == (B, L, D)
        print('✅' + " Mamba 测试通过")
    except NotImplementedError:
        print('ℹ' + " 待实现")
