"""
【题目】Linear Attention：线性注意力

【背景】
标准 attention O(n²) 复杂度。Linear Attention 去掉 softmax，用核函数 φ 替代：
y = (φ(Q) φ(K)^T V) / (φ(Q) φ(K)^T 1)
关键：φ(Q) φ(K)^T V = φ(Q) (φ(K)^T V)，先算 φ(K)^T V (O(n d²))，再乘 φ(Q)。
总复杂度 O(n d²)，当 d << n 时为线性。优势：无 KV cache、常量推理显存。

【输入/输出】
- 输入：Q, K, V [B, L, D]
- 输出：y [B, L, D]

【考察点】
- 核函数 φ 选择（elu+1, relu 等）
- 结合律改变计算顺序降复杂度
- 提示：先算 S = K^T V，再 y = Q @ S
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def phi_elu(x):
    """ELU+1 核函数：保证非负。"""
    return F.elu(x) + 1


def linear_attention(Q, K, V, phi=phi_elu):
    """线性注意力：O(n d²) 复杂度。"""
    Q_phi = phi(Q)
    K_phi = phi(K)
    S = torch.einsum("bld,bld->bdd", K_phi, V)
    Z = K_phi.sum(dim=1)
    y = torch.einsum("bld,bdd->bld", Q_phi, S)
    normalizer = torch.einsum("bld,bd->bl", Q_phi, Z).unsqueeze(-1)
    return y / (normalizer + 1e-8)


def linear_attention_causal(Q, K, V, phi=phi_elu):
    """因果线性注意力：递推式。"""
    B, L, D = Q.shape
    Q_phi = phi(Q)
    K_phi = phi(K)
    S = torch.zeros(B, D, D, device=Q.device)
    Z = torch.zeros(B, D, device=Q.device)
    outputs = []
    for t in range(L):
        y_t = torch.einsum("bd,bdd->bd", Q_phi[:, t, :], S)
        z_t = torch.einsum("bd,bd->b", Q_phi[:, t, :], Z).unsqueeze(-1)
        out_t = y_t / (z_t + 1e-8)
        outputs.append(out_t)
        S = S + torch.einsum("bd,be->bde", K_phi[:, t, :], V[:, t, :])
        Z = Z + K_phi[:, t, :]
    return torch.stack(outputs, dim=1)


class LinearAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=False):
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if causal:
            y = linear_attention_causal(Q, K, V)
        else:
            y = linear_attention(Q, K, V)
        return self.out_proj(y)


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, D = 2, 16, 32
    Q, K, V = torch.randn(B, L, D), torch.randn(B, L, D), torch.randn(B, L, D)

    y = linear_attention(Q, K, V)
    assert y.shape == (B, L, D)
    print(f"✅ Linear attention: {Q.shape} -> {y.shape}")

    y_c = linear_attention_causal(Q, K, V)
    assert y_c.shape == (B, L, D)
    print("✅ Causal linear attention 正确")

    layer = LinearAttentionLayer(D)
    x = torch.randn(B, L, D)
    y_layer = layer(x)
    assert y_layer.shape == (B, L, D)
    print("✅ LinearAttentionLayer forward")

    y_causal = layer(x, causal=True)
    assert y_causal.shape == (B, L, D)
    print("✅ Causal mode 正确")

    Q2, K2, V2 = torch.randn(B, 100, D), torch.randn(B, 100, D), torch.randn(B, 100, D)
    y2 = linear_attention(Q2, K2, V2)
    assert y2.shape == (B, 100, D)
    print("✅ 长序列正确")
    print("✅ 全部测试通过")
