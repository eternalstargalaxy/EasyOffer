"""
【题目】线性注意力：Performer vs Linear Transformer vs RetNet

【背景】
标准 softmax attention O(n^2) 复杂度，线性注意力用 kernel trick 降为 O(n)。
Linear Transformer: softmax -> phi(Q) phi(K)^T V，phi=elu(x)+1 保证非负。
Performer(FAVOR+): 用随机 Fourier 特征近似 softmax kernel，
用 ortho-random projection 降低近似方差。
RetNet: 多尺度指数衰减 mask + chunkwise 递归，训练并行(O(1)每chunk)，
推理变 RNN(O(1)每step)。三种方法对比：精度/速度/显存。

【输入/输出】
- 输入：Q,K,V [B,H,L,D]
- 输出：O [B,H,L,D]，计算过程不创建 LxL 矩阵

【考察点】
- kernel trick 与 softmax 的误差分析
- Performer 正交 random projection 降方差
- RetNet 的 multi-scale decay + chunkwise 并行
- 提示：torch.randn 生成正交矩阵用 torch.linalg.qr
"""
import torch; import torch.nn.functional as F


def linear_attention_eltu(Q, K, V):
    """Linear Transformer: phi(x)=elu(x)+1, O = phi(Q)(phi(K)^T V)"""
    raise NotImplementedError


def performer_attention(Q, K, V, num_features: int = 64):
    """Performer: 用 orthogonal random features 近似 softmax"""
    raise NotImplementedError


def retnet_chunk_attention(Q, K, V, gamma: float = 0.9):
    """RetNet: chunkwise 递归 + exponential decay mask"""
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    B, H, L, D = 2, 4, 128, 32
    Q = torch.randn(B, H, L, D)
    K = torch.randn(B, H, L, D)
    V = torch.randn(B, H, L, D)
    std = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    try:
        o1 = linear_attention_eltu(Q, K, V)
        assert o1.shape == (B, H, L, D)
        print('✅' + f" Linear Attn 通过, 误差={(o1-std).abs().mean():.4f}")
    except NotImplementedError:
        print('ℹ' + " 待实现")
