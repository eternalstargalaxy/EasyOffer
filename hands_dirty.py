"""EasyOffer 综合手撕题合集

涵盖基础算法/损失函数/归一化/注意力/Transformer 等。
更多完整模块请见各子目录。
"""

import numpy as np
import torch

# # EasyOffer 综合手撕题索引
#
# ## 项目模块索引
# | 模块 | 内容 | 题数 |
# |------|------|------|
# | Easy_Pytorch/ | Softmax/CrossEntropy/KL/Backprop/SGD | 5 |
# | Easy_Attention/ | MHA/FlashAttn/SparseAttn/GQA_MLA/RoPE/LoRA/FFN + activation | 13 |
# | Easy_Generator/ | Greedy/TopK/TopP/Temperature/BeamSearch | 5 |
# | Easy_AIInfra/ | training(13)/inference(10)/quant(4)/sparse(2)/ssm(5)/advanced(11) | 45 |
# | Easy_deepseek/ | DeepSeek 完整实现(MoE/MLA/MTP) | 15 |
# | Easy_RL/ | REINFORCE/PPO/GRPO/GSPO/DPO/SimPO/RM/RLHF | 9 |
# | AIInfra面经/ | 字节/阿里/腾讯 等面经 + 专题 | 12 |
# | 大厂常见思维题/ | 小红书/字节 + AIInfra 计算题 | 8 |
#
# ## 本文件内容
# 以下为经典大模型手撕题实现，涵盖基础算法/损失函数/归一化/注意力/Transformer/RAG。
# 更多完整模块请见上述目录。
#

# ## 1. 基础算法
#
# ### K-means 算法

def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# 示例
X = np.random.randn(100, 2)
centroids, labels = kmeans(X, 3)
print("K-means centroids shape:", centroids.shape)

# ### NumPy 实现 MLP 反向传播

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, x, y, output, lr=0.01):
        m = x.shape[0]
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

# 示例
mlp = SimpleMLP(10, 5, 1)
x = np.random.randn(4, 10)
y = np.random.randn(4, 1)
output = mlp.forward(x)
mlp.backward(x, y, output)
print("MLP backward done")

# ### 手写梯度计算

def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_flat = x.flatten()
        x_flat[i] += h
        f_plus = f(x_flat.reshape(x.shape))
        x_flat[i] -= 2*h
        f_minus = f(x_flat.reshape(x.shape))
        grad.flat[i] = (f_plus - f_minus) / (2*h)
    return grad

# 示例
def f(x):
    return np.sum(x**2)

x = np.random.randn(3, 2)
grad_num = numerical_gradient(f, x)
grad_ana = 2 * x
print("Numerical grad close to analytical:", np.allclose(grad_num, grad_ana))

# ### 反向传播推导

# 反向传播推导示例: 矩阵乘法 y = x @ W, loss = 0.5 * (y - t)^2
# dy/dy = y - t
# dW = x.T @ dy
# dx = dy @ W.T

def matmul_backward(x, W, dy):
    dW = x.T @ dy
    dx = dy @ W.T
    return dx, dW

# 示例
x = np.random.randn(4, 3)
W = np.random.randn(3, 2)
y = x @ W
t = np.random.randn(4, 2)
dy = y - t
dx, dW = matmul_backward(x, W, dy)
print("dx shape:", dx.shape, "dW shape:", dW.shape)

# ### SGD/Adam 优化器

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.lr * grad

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# 示例
params = [np.random.randn(2, 3)]
grads = [np.random.randn(2, 3)]
sgd = SGD()
sgd.step(params, grads)
adam = Adam()
adam.step(params, grads)
print("Optimizers updated")

# ## 2. 信息论度量
#
# ### 熵 (Entropy)

def entropy(p):
    p = p / np.sum(p)
    return -np.sum(p * np.log(p + 1e-10))

# 示例
p = np.array([0.1, 0.2, 0.3, 0.4])
ent = entropy(p)
print("Entropy:", ent)

# ### KL散度 (KL Divergence)

def kl_divergence(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

# 示例
p = np.array([0.1, 0.2, 0.3, 0.4])
q = np.array([0.2, 0.2, 0.3, 0.3])
kl = kl_divergence(p, q)
print("KL Divergence:", kl)

# ## 3. 损失函数
#
# ### 交叉熵损失

def cross_entropy_loss(logits, targets):
    # logits: (batch, seq, vocab), targets: (batch, seq)
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
    loss = -np.mean(log_probs[np.arange(logits.shape[0])[:, None], np.arange(logits.shape[1]), targets])
    return loss

# 示例
logits = np.random.randn(1, 10, 1000)
targets = np.random.randint(0, 1000, (1, 10))
loss_ce = cross_entropy_loss(logits, targets)
print("Cross Entropy Loss:", loss_ce)

# ### 对比学习损失 (Contrastive Loss)

def contrastive_loss(embeddings, labels, temperature=0.1):
    # embeddings: (batch, dim), labels: (batch,)
    sim = np.matmul(embeddings, embeddings.T) / temperature
    exp_sim = np.exp(sim - np.max(sim, axis=1, keepdims=True))
    mask = (labels[:, None] == labels[None, :]).astype(float)
    pos_sim = exp_sim * mask
    neg_sim = exp_sim * (1 - mask)
    loss = -np.log(np.sum(pos_sim, axis=1) / (np.sum(pos_sim, axis=1) + np.sum(neg_sim, axis=1)))
    return np.mean(loss)

# 示例
emb = np.random.randn(4, 128)
labels = np.array([0, 0, 1, 1])
loss_con = contrastive_loss(emb, labels)
print("Contrastive Loss:", loss_con)

# ## 4. 归一化技术
#
# ### LayerNorm

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

# 示例
gamma = np.ones(512)
beta = np.zeros(512)
x_norm = layer_norm(x_ffn, gamma, beta)
print("LayerNorm output shape:", x_norm.shape)

# ### RMSNorm

def rms_norm(x, gamma, eps=1e-5):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return gamma * (x / rms)

# 示例
x_rms = rms_norm(x_ffn, gamma)
print("RMSNorm output shape:", x_rms.shape)

# ### BatchNorm

def batch_norm(x, gamma, beta, eps=1e-5, momentum=0.9):
    # For simplicity, assume x is (batch, seq, d_model), normalize over batch and seq
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    var = np.var(x, axis=(0, 1), keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

# 示例
x_batch = batch_norm(x_ffn, gamma, beta)
print("BatchNorm output shape:", x_batch.shape)

# ## 5. 前馈网络 (FFN)
#
# ### FFN with ReLU

def ffn_relu(x, d_model, d_ff):
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)
    
    h = np.maximum(0, np.matmul(x, W1) + b1)  # ReLU
    output = np.matmul(h, W2) + b2
    return output

# 示例
x_ffn = np.random.randn(1, 10, 512)
output_ffn = ffn_relu(x_ffn, 512, 2048)
print("FFN ReLU output shape:", output_ffn.shape)

# ### FFN with GeLU

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def ffn_gelu(x, d_model, d_ff):
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)
    
    h = gelu(np.matmul(x, W1) + b1)
    output = np.matmul(h, W2) + b2
    return output

# 示例
output_gelu = ffn_gelu(x_ffn, 512, 2048)
print("FFN GeLU output shape:", output_gelu.shape)

# ### SwiGLU

def swish(x):
    return x * (1 / (1 + np.exp(-x)))

def swiglu(x, d_model, d_ff):
    W_gate = np.random.randn(d_model, d_ff)
    b_gate = np.random.randn(d_ff)
    W_up = np.random.randn(d_model, d_ff)
    b_up = np.random.randn(d_ff)
    W_out = np.random.randn(d_ff, d_model)
    b_out = np.random.randn(d_model)
    
    gate = swish(np.matmul(x, W_gate) + b_gate)
    up = np.matmul(x, W_up) + b_up
    h = gate * up
    output = np.matmul(h, W_out) + b_out
    return output

# 示例
output_swiglu = swiglu(x_ffn, 512, 2048)
print("SwiGLU output shape:", output_swiglu.shape)

# ## 6. 位置编码
#
# ### 绝对位置编码

def absolute_position_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

# 示例
pe = absolute_position_encoding(10, 512)
print("Absolute PE shape:", pe.shape)

# ### 旋转位置编码 (RoPE)

def rope_position_encoding(seq_len, d_model, base=10000):
    position = np.arange(seq_len)
    dim = np.arange(d_model // 2)
    theta = 1.0 / (base ** (2 * dim / d_model))
    angles = position[:, np.newaxis] * theta[np.newaxis, :]
    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin

def apply_rope(x, cos, sin):
    # x shape: (seq_len, d_model)
    x1 = x[:, :x.shape[1]//2]
    x2 = x[:, x.shape[1]//2:]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x2 * cos + x1 * sin
    return np.concatenate([rotated_x1, rotated_x2], axis=-1)

# 示例
cos, sin = rope_position_encoding(10, 512)
x = np.random.randn(10, 512)
x_rope = apply_rope(x, cos, sin)
print("RoPE applied shape:", x_rope.shape)

# ### 长度外推技术

def rope_with_extrapolation(seq_len, d_model, base=10000, scale=1.0):
    # Simple scaling for extrapolation
    position = np.arange(seq_len)
    dim = np.arange(d_model // 2)
    theta = scale / (base ** (2 * dim / d_model))
    angles = position[:, np.newaxis] * theta[np.newaxis, :]
    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin

# 示例 with scaling
cos_ext, sin_ext = rope_with_extrapolation(10, 512, scale=2.0)
print("Extrapolated RoPE shape:", cos_ext.shape)

# ## 7. 注意力机制
#
# ### Multi-Head Attention (MHA)

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores += mask * -1e9
    attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn /= np.sum(attn, axis=-1, keepdims=True)
    output = np.matmul(attn, v)
    return output, attn

def multi_head_attention(x, num_heads, d_model):
    d_k = d_v = d_model // num_heads
    batch_size, seq_len, _ = x.shape
    
    # Linear projections
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)
    W_o = np.random.randn(d_model, d_model)
    
    q = np.matmul(x, W_q).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    k = np.matmul(x, W_k).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    v = np.matmul(x, W_v).reshape(batch_size, seq_len, num_heads, d_v).transpose(0, 2, 1, 3)
    
    # Attention
    attn_output, _ = scaled_dot_product_attention(q, k, v)
    
    # Concat and output projection
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    output = np.matmul(attn_output, W_o)
    return output

# 示例
x = np.random.randn(1, 10, 512)
output = multi_head_attention(x, 8, 512)
print("MHA output shape:", output.shape)

# ### Multi-Query Attention (MQA)

def multi_query_attention(x, num_heads, d_model):
    d_k = d_v = d_model // num_heads
    batch_size, seq_len, _ = x.shape
    
    # Linear projections: multiple Q, single K, V
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_k)
    W_v = np.random.randn(d_model, d_v)
    W_o = np.random.randn(d_model, d_model)
    
    q = np.matmul(x, W_q).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    k = np.matmul(x, W_k).reshape(batch_size, seq_len, 1, d_k).transpose(0, 2, 1, 3).repeat(num_heads, axis=1)
    v = np.matmul(x, W_v).reshape(batch_size, seq_len, 1, d_v).transpose(0, 2, 1, 3).repeat(num_heads, axis=1)
    
    # Attention
    attn_output, _ = scaled_dot_product_attention(q, k, v)
    
    # Concat and output projection
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    output = np.matmul(attn_output, W_o)
    return output

# 示例
output_mqa = multi_query_attention(x, 8, 512)
print("MQA output shape:", output_mqa.shape)

# ### Group Query Attention (GQA)
#
# GQA 将heads分组，每组共享K和V。
#
# ### Multi Head Latent Attention (MLA)
#
# MLA 使用latent space来压缩K和V。

# ## 8. Tokenizer: 从零实现 BPE (Byte Pair Encoding)

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.inverse_vocab = {}

    def get_stats(self, ids):
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text):
        # Pre-tokenize into bytes
        tokens = list(text.encode("utf-8"))
        ids = list(tokens)

        # Initial vocab
        self.vocab = {i: bytes([i]) for i in range(256)}
        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        # Build inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

# 示例使用
tokenizer = BPETokenizer(vocab_size=300)
text = "Hello world! This is a test for BPE tokenizer."
tokenizer.train(text)
encoded = tokenizer.encode("Hello")
decoded = tokenizer.decode(encoded)
print("Encoded:", encoded)
print("Decoded:", decoded)

# ## 9. 手推Transformer
#
# ### PyTorch MHA

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

# 示例
mha = MultiHeadAttention(512, 8)
q = torch.randn(1, 10, 512)
output = mha(q, q, q)
print("PyTorch MHA output shape:", output.shape)

# ### PyTorch LayerNorm

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta

# 示例
ln = LayerNorm(512)
x = torch.randn(1, 10, 512)
output = ln(x)
print("LayerNorm output shape:", output.shape)

# ### PyTorch RoPE

class RoPE(nn.Module):
    def __init__(self, d_model, base=10000):
        super().__init__()
        self.d_model = d_model
        self.base = base

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device) * -(torch.log(torch.tensor(self.base)) / self.d_model))
        angles = position * div_term
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        x1 = x[..., :self.d_model//2]
        x2 = x[..., self.d_model//2:]
        x_rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return x_rotated

# 示例
rope = RoPE(512)
x = torch.randn(1, 10, 512)
output = rope(x)
print("RoPE output shape:", output.shape)

# ### PyTorch CrossEntropyLoss (with shift)

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # For language modeling, shift targets
        # logits: (batch, seq, vocab), targets: (batch, seq)
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)
        return self.criterion(logits, targets)

# 示例
ce_loss = CrossEntropyLoss()
logits = torch.randn(1, 10, 1000)
targets = torch.randint(0, 1000, (1, 10))
loss = ce_loss(logits, targets)
print("CrossEntropyLoss:", loss.item())

# ## 10. RAG常见手撕题
#
# ### 向量相似度计算

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 示例
a = np.random.randn(128)
b = np.random.randn(128)
sim = cosine_similarity(a, b)
print("Cosine similarity:", sim)
