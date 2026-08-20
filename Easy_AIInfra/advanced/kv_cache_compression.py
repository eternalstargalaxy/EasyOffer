"""
【题目】KV Cache 压缩

【背景】
长上下文推理时 KV Cache 显存占用 O(n*d) 线性增长。压缩方法：
1. 量化压缩：KV 从 fp16 压到 int8/int4，2-4x 压缩
2. 驱逐策略：LRU/H2O 丢弃不重要的 KV，保留 top-k
3. 低秩压缩：KV = U @ V，U/V 维度远小于 n
4. 滑动窗口：只保留最近 W 个 token 的 KV

【输入/输出】
- 输入：KV Cache, 压缩策略, 参数
- 输出：压缩后的 KV Cache

【考察点】
- 各策略的压缩率与精度 trade-off
- 驱逐策略的 attention 等价性
- 提示：torch.topk 选择重要 KV
"""
import torch
import torch.nn.functional as F


def quantize_kv(kv: torch.Tensor, bits: int = 8):
    """KV Cache 量化压缩。"""
    qmax = 2 ** (bits - 1) - 1
    scale = kv.abs().max() / qmax
    scale = scale.clamp(min=1e-8)
    kv_int = torch.round(kv / scale).clamp(-qmax, qmax).to(torch.int8)
    return kv_int, scale


def dequantize_kv(kv_int: torch.Tensor, scale: torch.Tensor):
    return kv_int.float() * scale


def sliding_window_kv(kv: torch.Tensor, window: int):
    """滑动窗口：只保留最近 window 个 token。"""
    return kv[-window:]


def lru_evict_kv(kv: torch.Tensor, keep: int):
    """LRU 驱逐：保留最近 keep 个。"""
    return kv[-keep:]


def h2o_evict_kv(kv: torch.Tensor, attention_scores: torch.Tensor, keep: int):
    """H2O：根据 attention score 保留 top-k 重要 KV。"""
    importance = attention_scores.sum(dim=0)
    topk_idx = importance.topk(keep).indices.sort().values
    return kv[topk_idx]


def low_rank_compress(kv: torch.Tensor, rank: int):
    """低秩压缩：SVD 分解取 top-rank。"""
    U, S, Vh = torch.linalg.svd(kv.float(), full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    compressed = U_r @ torch.diag(S_r) @ Vh_r
    return compressed, (U_r, S_r, Vh_r)


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    kv = torch.randn(100, 64)

    kv_int, scale = quantize_kv(kv, bits=8)
    assert kv_int.dtype == torch.int8
    kv_deq = dequantize_kv(kv_int, scale)
    err = (kv - kv_deq).abs().mean().item()
    assert err < 0.01
    print(f"✅ 量化压缩: int8, MAE={err:.6f}, 压缩 2x")

    kv_win = sliding_window_kv(kv, window=20)
    assert kv_win.shape == (20, 64)
    print("✅ 滑动窗口: 100 -> 20")

    kv_lru = lru_evict_kv(kv, keep=30)
    assert kv_lru.shape == (30, 64)
    print("✅ LRU 驱逐: 100 -> 30")

    scores = torch.rand(10, 100)
    kv_h2o = h2o_evict_kv(kv, scores, keep=25)
    assert kv_h2o.shape == (25, 64)
    print("✅ H2O 驱逐: 100 -> 25")

    kv_lr, (U, S, Vh) = low_rank_compress(kv, rank=16)
    assert kv_lr.shape == kv.shape
    err_lr = (kv - kv_lr).abs().mean().item()
    print(f"✅ 低秩压缩: rank=16, MAE={err_lr:.6f}")
    print("✅ 全部测试通过")
