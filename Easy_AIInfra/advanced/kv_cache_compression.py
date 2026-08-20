"""
【题目】KV Cache 压缩：H2O / SnapKV / StreamingLLM

【背景】
长序列推理时 KV Cache 显存 O(L*n_layers*n_heads*d_head) 成为瓶颈。
H2O(Heavy Hitter Oracle)：保留累积 attention 分数最高的 heavy hitter token，
丢弃其他，实现近无损压缩。heavy hitter 由累积 attention score sum 度量。
SnapKV：观察 prompt 末尾几个 token 对前面 token 的 attention pattern，
选出重要位置进行压缩(仅保留被注意力高的 KV 对)。
StreamingLLM：保留开头的 attention sink token(前4个) + 最近 window，
丢弃中间，实现无限长序列推理。
三者对比：H2O 压缩比高(80-90%)，SnapKV 适配好(观察驱动)，StreamingLLM 简单。

【输入/输出】
- 输入：KV cache [B,H,L,D], 压缩比 ratio
- 输出：压缩后 KV cache，建议保留的重要 token 索引列表

【考察点】
- attention sink 现象(开头 token 总被关注)
- heavy hitter 基于累积分数的选择策略
- 提示：torch.topk 选 top-k heavy hitters，torch.gather 做索引压缩
"""
import torch


def h2o_compress(K, V, attn_scores_sum, keep_ratio: float = 0.1):
    raise NotImplementedError


def snapkv_compress(K, V, window_size: int = 32, keep_ratio: float = 0.2):
    raise NotImplementedError


def streaming_llm_compress(K, V, num_sinks: int = 4, window_size: int = 256):
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    B, H, L, D = 1, 4, 512, 64
    K = torch.randn(B, H, L, D)
    V = torch.randn(B, H, L, D)
    scores = torch.randn(B, H, L).abs()
    try:
        kept = h2o_compress(K, V, scores, 0.2)
        print('✅' + f" H2O 压缩通过")
    except NotImplementedError:
        print('ℹ' + " 待实现")
