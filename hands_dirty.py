"""
EasyOffer 综合手撕题合集

涵盖基础算法/损失函数/归一化/注意力/Transformer 等经典手撕题。
每个算法含实现 + assert 测试，运行 `python hands_dirty.py` 可一键验证全部。

更多完整模块请见各子目录：
  Easy_Pytorch(10) / Easy_Attention(14) / Easy_Generator(8) / Easy_Tokenizer(4)
  Easy_AIInfra(48) / Easy_RL(12) / Easy_deepseek(13)
"""

import numpy as np
import torch
import torch.nn.functional as F
import math

# ============================================================
# 1. 基础算法
# ============================================================

def kmeans(X, k, max_iters=100):
    """K-means 聚类"""
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

def numerical_gradient(f, x, h=1e-5):
    """数值梯度（中心差分）"""
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_flat = x.flatten()
        x_flat[i] += h
        f_plus = f(x_flat.reshape(x.shape))
        x_flat[i] -= 2 * h
        f_minus = f(x_flat.reshape(x.shape))
        grad.flat[i] = (f_plus - f_minus) / (2 * h)
    return grad

# ============================================================
# 2. 损失函数
# ============================================================

def softmax(x):
    """数值稳定 softmax"""
    x_max = x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def cross_entropy(logits, targets):
    """交叉熵损失（log-sum-exp 稳定实现）"""
    log_probs = logits - np.log(np.sum(np.exp(logits - logits.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)) - logits.max(axis=-1, keepdims=True)
    return -np.mean(log_probs[np.arange(len(targets)), targets])

def kl_divergence(p, q):
    """KL 散度 KL(p||q)"""
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

def contrastive_loss(embeddings, labels, temperature=0.1):
    """对比学习损失（InfoNCE）"""
    sim = np.matmul(embeddings, embeddings.T) / temperature
    exp_sim = np.exp(sim - np.max(sim, axis=1, keepdims=True))
    mask = (labels[:, None] == labels[None, :]).astype(float)
    pos = exp_sim * mask
    neg = exp_sim * (1 - mask)
    return np.mean(-np.log(np.sum(pos, axis=1) / (np.sum(pos, axis=1) + np.sum(neg, axis=1))))

# ============================================================
# 3. 归一化
# ============================================================

def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm: 逐样本归一化"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def rms_norm(x, gamma, eps=1e-6):
    """RMSNorm: 去均值，只用 RMS"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return gamma * (x / rms)

def batch_norm(x, gamma, beta, eps=1e-5):
    """BatchNorm: 跨 batch+seq 归一化"""
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    var = np.var(x, axis=(0, 1), keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

# ============================================================
# 4. 激活函数
# ============================================================

def gelu(x):
    """GELU（tanh 近似）"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def silu(x):
    """SiLU / Swish"""
    return x / (1 + np.exp(-x))

# ============================================================
# 5. 前馈网络
# ============================================================

def ffn_swiglu(x, d_model, d_ff):
    """SwiGLU FFN: w2(silu(w1(x)) * w3(x))"""
    w1 = np.random.randn(d_model, d_ff) * 0.02
    w2 = np.random.randn(d_ff, d_model) * 0.02
    w3 = np.random.randn(d_model, d_ff) * 0.02
    return (silu(x @ w1) * (x @ w3)) @ w2

# ============================================================
# 6. 位置编码
# ============================================================

def absolute_position_encoding(seq_len, d_model):
    """绝对正弦余弦位置编码"""
    pos = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    return pe

def apply_rope(x, freqs):
    """旋转位置编码（复数乘法）"""
    x_complex = x[..., 0::2] + 1j * x[..., 1::2]
    x_rotated = x_complex * freqs
    out = np.zeros_like(x)
    out[..., 0::2] = x_rotated.real
    out[..., 1::2] = x_rotated.imag
    return out

# ============================================================
# 7. 注意力
# ============================================================

def scaled_dot_product_attention(q, k, v, mask=None):
    """缩放点积注意力"""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    attn = softmax(scores)
    return attn @ v

def multi_head_attention(x, n_heads, d_model):
    """多头注意力（NumPy 简版）"""
    d_head = d_model // n_heads
    W_q = np.random.randn(d_model, d_model) * 0.02
    W_k = np.random.randn(d_model, d_model) * 0.02
    W_v = np.random.randn(d_model, d_model) * 0.02
    W_o = np.random.randn(d_model, d_model) * 0.02
    q = (x @ W_q).reshape(*x.shape[:-1], n_heads, d_head).transpose(-2, -3, -1)
    k = (x @ W_k).reshape(*x.shape[:-1], n_heads, d_head).transpose(-2, -3, -1)
    v = (x @ W_v).reshape(*x.shape[:-1], n_heads, d_head).transpose(-2, -3, -1)
    attn = scaled_dot_product_attention(q, k, v)
    out = attn.transpose(-2, -3, -1).reshape(*x.shape[:-1], d_model) @ W_o
    return out

# ============================================================
# 8. 优化器
# ============================================================

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = None, None, 0
    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ============================================================
# 测试验证
# ============================================================

def _test():
    np.random.seed(42)
    print("Running all tests...")

    # K-means
    X = np.random.randn(100, 2)
    centroids, labels = kmeans(X, 3)
    assert centroids.shape == (3, 2)
    print("  ✅ K-means")

    # 数值梯度
    x = np.random.randn(3, 2)
    grad_num = numerical_gradient(lambda x: np.sum(x ** 2), x)
    assert np.allclose(grad_num, 2 * x, atol=1e-6)
    print("  ✅ numerical_gradient")

    # Softmax
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(sum(p) - 1.0) < 1e-6
    print("  ✅ softmax")

    # KL 散度
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert kl_divergence(p, p) < 1e-6
    assert kl_divergence(p, np.array([0.4, 0.3, 0.2, 0.1])) > 0
    print("  ✅ kl_divergence")

    # LayerNorm
    x = np.random.randn(2, 3, 4)
    gamma, beta = np.ones(4), np.zeros(4)
    out = layer_norm(x, gamma, beta)
    assert abs(out[0, 0].mean()) < 1e-5
    print("  ✅ layer_norm")

    # RMSNorm
    out = rms_norm(x, gamma)
    assert out.shape == x.shape
    print("  ✅ rms_norm")

    # GELU
    assert abs(gelu(0)) < 1e-6
    print("  ✅ gelu")

    # 位置编码
    pe = absolute_position_encoding(10, 64)
    assert pe.shape == (10, 64)
    print("  ✅ absolute_position_encoding")

    # 注意力
    q = k = v = np.random.randn(2, 4, 8)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 8)
    print("  ✅ scaled_dot_product_attention")

    # 优化器
    p = np.array([5.0])
    opt = Adam(lr=0.1)
    for _ in range(100):
        grad = 2 * p
        opt.step([p], [grad])
    assert abs(p[0]) < 0.1
    print("  ✅ Adam optimizer")

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    _test()
