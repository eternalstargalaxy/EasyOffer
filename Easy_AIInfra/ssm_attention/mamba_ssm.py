"""
【题目】Mamba SSM：选择性状态空间模型

【背景】
Mamba 用选择性 SSM 替代 attention：h_t = A * h_{t-1} + B * x_t, y_t = C * h_t。
选择性体现在 A/B/C 是输入相关的（由 Linear(x) 算出），而非固定参数。
优势：线性复杂度 O(n)、常量推理显存（不存 KV cache）、长序列建模强。
离散化：A_bar = exp(A * delta), B_bar = B * delta。

【输入/输出】
- 输入：x [B, L, D]
- 输出：y [B, L, D]

【考察点】
- 选择性参数 A/B/C 的输入依赖
- 离散化与递推
- 提示：逐步递推 h_t
"""
import torch
import torch.nn as nn
import math


class MambaSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.proj_delta = nn.Linear(d_model, 1, bias=True)
        self.proj_A = nn.Linear(d_model, d_state, bias=False)
        self.proj_B = nn.Linear(d_model, d_state, bias=False)
        self.proj_C = nn.Linear(d_model, d_state, bias=False)
        self.proj_D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        N = self.d_state
        delta = torch.softplus(self.proj_delta(x))
        A = -torch.exp(self.proj_A(x))
        B_in = self.proj_B(x)
        C = self.proj_C(x)
        A_bar = torch.exp(A * delta)
        B_bar = B_in * delta
        h = torch.zeros(B, N, device=x.device)
        outputs = []
        for t in range(L):
            h = A_bar[:, t, :] * h + B_bar[:, t, :].unsqueeze(1) * x[:, t, :].unsqueeze(1)
            y_t = (C[:, t, :].unsqueeze(1) * h).sum(dim=-1)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y + x * self.proj_D
        return self.out_proj(y)


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    model = MambaSSM(d_model=32, d_state=16)
    x = torch.randn(2, 10, 32)
    y = model(x)
    assert y.shape == (2, 10, 32)
    print(f"✅ Mamba SSM: {x.shape} -> {y.shape}")

    x2 = torch.randn(4, 20, 32)
    y2 = model(x2)
    assert y2.shape == (4, 20, 32)
    print("✅ 不同 batch/seq 正确")

    y3 = model(x)
    assert torch.allclose(y, y3, atol=1e-5), "同输入应确定性输出"
    print("✅ 确定性输出")
    print("✅ 全部测试通过")
