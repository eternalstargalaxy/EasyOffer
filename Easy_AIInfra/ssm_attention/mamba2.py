"""
【题目】Mamba-2：SSD（State Space Duality）

【背景】
Mamba-2 把 SSM 与 attention 统一：SSD (Selective State Space Duality)。
核心 insight：SSM 的递推可写成矩阵形式 y = M @ (K^T V)，M 是下三角结构矩阵。
与 attention 的 y = softmax(QK^T)V 对偶，M 替代 softmax attention matrix。
优势：可用 attention 的 tiling/parallel 算法加速、理论统一。

【输入/输出】
- 输入：x [B, L, D]
- 输出：y [B, L, D]

【考察点】
- SSD 的矩阵形式
- 与 attention 的对偶关系
- 提示：构造下三角 M 矩阵
"""
import torch
import torch.nn as nn


class Mamba2SSD(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, headdim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.n_heads = d_model // headdim
        self.proj_A = nn.Linear(d_model, self.n_heads, bias=False)
        self.proj_B = nn.Linear(d_model, self.n_heads * d_state, bias=False)
        self.proj_C = nn.Linear(d_model, self.n_heads * d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        H, N = self.n_heads, self.d_state
        A = -torch.exp(self.proj_A(x))
        B_in = self.proj_B(x).view(B, L, H, N)
        C = self.proj_C(x).view(B, L, H, N)
        x_heads = x.view(B, L, H, self.headdim)
        outputs = []
        for t in range(L):
            y_t = torch.zeros(B, H, self.headdim, device=x.device)
            for s in range(t + 1):
                decay = torch.exp(A[:, t, :] + A[:, s, :]) * (0.5 if t != s else 1.0)
                kv = (B_in[:, s, :, :] * x_heads[:, s, :, :].unsqueeze(-1)).sum(dim=-1)
                q = C[:, t, :, :]
                y_t += (q * decay.unsqueeze(-1) * kv.unsqueeze(1)).sum(dim=-1).unsqueeze(-1)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1).view(B, L, D)
        return self.out_proj(y)


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    model = Mamba2SSD(d_model=32, d_state=8, headdim=16)
    x = torch.randn(2, 8, 32)
    y = model(x)
    assert y.shape == (2, 8, 32)
    print(f"✅ Mamba-2 SSD: {x.shape} -> {y.shape}")

    model2 = Mamba2SSD(d_model=64, d_state=16, headdim=32)
    x2 = torch.randn(3, 16, 64)
    y2 = model2(x2)
    assert y2.shape == (3, 16, 64)
    print("✅ 不同维度正确")
    print("✅ 全部测试通过")
