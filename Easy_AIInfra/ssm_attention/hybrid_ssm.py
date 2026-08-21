"""
【题目】Hybrid SSM-Attention 架构

【背景】
纯 SSM 擅长长序列但短序列/精确检索不如 attention；纯 attention 擅长检索但长序列 O(n²)。
Hybrid 架构交替使用 SSM 和 attention 层（如 Jamba: 1 attention + 7 SSM）：
- SSM 层处理长距离依赖，线性复杂度
- Attention 层处理精确检索，少量即可
优势：兼顾长序列效率与检索能力。

【输入/输出】
- 输入：x [B, L, D]
- 输出：y [B, L, D]

【考察点】
- SSM/Attention 交替比例
- 各层的作用互补
- 提示：nn.ModuleList 交替
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.ln(x), self.ln(x), self.ln(x))
        return x + h


class SSMLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.proj_in = nn.Linear(d_model, d_model)
        self.proj_A = nn.Linear(d_model, d_state)
        self.proj_B = nn.Linear(d_model, d_state)
        self.proj_C = nn.Linear(d_model, d_state)
        self.proj_out = nn.Linear(d_model, d_model)
        self.d_state = d_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = self.d_state
        h = self.proj_in(x)
        A = -torch.exp(self.proj_A(h))
        B_in = self.proj_B(h)
        C = self.proj_C(h)
        state = torch.zeros(B, N, device=x.device)
        outputs = []
        for t in range(L):
            state = A[:, t, :] * state + B_in[:, t, :] * h[:, t, :].unsqueeze(-1)
            y_t = (C[:, t, :].unsqueeze(-1) * state).sum(dim=1)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        return x + self.proj_out(y)


class HybridSSMAttention(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 8, ssm_ratio: float = 0.75, num_heads: int = 4):
        super().__init__()
        num_ssm = int(num_layers * ssm_ratio)
        num_attn = num_layers - num_ssm
        layers = []
        ssm_idx, attn_idx = 0, 0
        for i in range(num_layers):
            if (i + 1) % (num_layers // max(num_attn, 1)) == 0 and attn_idx < num_attn:
                layers.append(AttentionLayer(d_model, num_heads))
                attn_idx += 1
            else:
                layers.append(SSMLayer(d_model))
                ssm_idx += 1
        self.layers = nn.ModuleList(layers)
        self.num_ssm = ssm_idx
        self.num_attn = attn_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    model = HybridSSMAttention(d_model=32, num_layers=8, ssm_ratio=0.75)
    x = torch.randn(2, 10, 32)
    y = model(x)
    assert y.shape == (2, 10, 32)
    print(f"✅ Hybrid: {x.shape} -> {y.shape}, SSM={model.num_ssm}, Attn={model.num_attn}")

    assert model.num_ssm + model.num_attn == 8
    assert model.num_attn >= 1
    print("✅ 层分配正确")

    model2 = HybridSSMAttention(d_model=64, num_layers=4, ssm_ratio=0.5)
    x2 = torch.randn(3, 8, 64)
    y2 = model2(x2)
    assert y2.shape == (3, 8, 64)
    print(f"✅ 50/50: SSM={model2.num_ssm}, Attn={model2.num_attn}")
    print("✅ 全部测试通过")
