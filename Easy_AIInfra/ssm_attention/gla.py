"""
【题目】GLA：Gated Linear Attention

【背景】
GLA 在 Linear Attention 基础上引入门控机制：
h_t = G_t ⊙ h_{t-1} + K_t^T V_t, y_t = Q_t @ h_t
G_t 是输入相关的门控（forget gate），控制历史信息的遗忘。
优势：比普通 linear attention 更强（有选择性遗忘）、比 Mamba 更简单（无 SSM 离散化）。

【输入/输出】
- 输入：x [B, L, D]
- 输出：y [B, L, D]

【考察点】
- 门控 G 的输入依赖
- 递推式与并行式
- 提示：sigmoid 门控
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLALayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.g_proj = nn.Linear(d_model, num_heads)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        H, Hd = self.num_heads, self.head_dim
        Q = self.q_proj(x).view(B, L, H, Hd)
        K = self.k_proj(x).view(B, L, H, Hd)
        V = self.v_proj(x).view(B, L, H, Hd)
        G = torch.sigmoid(self.g_proj(x))
        S = torch.zeros(B, H, Hd, Hd, device=x.device)
        outputs = []
        for t in range(L):
            g_t = G[:, t, :].view(B, H, 1, 1)
            S = g_t * S + torch.einsum("bhd,bhe->bhde", K[:, t, :], V[:, t, :])
            y_t = torch.einsum("bhd,bhde->bhe", Q[:, t, :], S)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1).view(B, L, D)
        return self.out_proj(y)


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    model = GLALayer(d_model=32, num_heads=4)
    x = torch.randn(2, 10, 32)
    y = model(x)
    assert y.shape == (2, 10, 32)
    print(f"✅ GLA: {x.shape} -> {y.shape}")

    x2 = torch.randn(4, 20, 32)
    y2 = model(x2)
    assert y2.shape == (4, 20, 32)
    print("✅ 不同 batch/seq 正确")

    model2 = GLALayer(d_model=64, num_heads=8)
    x3 = torch.randn(2, 8, 64)
    y3 = model2(x3)
    assert y3.shape == (2, 8, 64)
    print("✅ 不同维度正确")
    print("✅ 全部测试通过")
