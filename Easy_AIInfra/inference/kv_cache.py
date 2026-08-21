"""
【题目】KV Cache 增量推理

【背景】
自回归解码每步只新增一个 token，若每步重算全部历史 K/V 则复杂度 O(n²)。
缓存历史 K/V，每步只算新 token 的 q/k/v 并 append，复杂度降为 O(n)。
prefill 阶段一次性算 prompt 的 K/V 填入缓存（可并行）；decode 阶段每步 1 token。
GQA 下 KV head 数 < Q head 数，需把 KV 复制/广播到对应 Q head；MLA 缓存的是低秩 latent，更省。

【输入/输出】
- 输入：q,k,v: Tensor[num_heads, d_head]（decode）或 [seq_len, num_heads, d_head]（prefill）
- 输出：attention 输出 Tensor[num_heads, d_head]；KV cache 状态被更新

【考察点】
- prefill（[S,H,d] 一次算）vs decode（[1,H,d] 逐 token）形状差异
- GQA 下 KV head → Q head 的映射（repeat_interleave）
- 缓存按 (layer, seq) 组织与 append
- 提示：torch.cat 拼接 KV；torch.tril 生成因果 mask
"""
import torch
import torch.nn.functional as F


class KVCache:
    """预分配式多层 KV Cache。"""

    def __init__(self, num_layers: int, num_kv_heads: int, d_head: int,
                 max_seq_len: int, dtype: torch.dtype = torch.float32):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.k_buf = torch.zeros(
            num_layers, max_seq_len, num_kv_heads, d_head, dtype=dtype
        )
        self.v_buf = torch.zeros_like(self.k_buf)
        self.seq_len = 0

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """把新 k/v 写入 buffer 末尾，更新 seq_len。"""
        seq_new = k.shape[0]
        end = self.seq_len + seq_new
        assert end <= self.max_seq_len, f"cache 满: {end} > {self.max_seq_len}"
        self.k_buf[layer_idx, self.seq_len:end] = k
        self.v_buf[layer_idx, self.seq_len:end] = v
        if layer_idx == self.num_layers - 1:
            self.seq_len = end

    def get(self, layer_idx: int) -> tuple:
        """返回该层已写入部分的 K, V: [cur_len, num_kv_heads, d_head]"""
        return (
            self.k_buf[layer_idx, :self.seq_len],
            self.v_buf[layer_idx, :self.seq_len],
        )


def attention_step(q_new: torch.Tensor, kv_cache: KVCache, layer_idx: int,
                   num_q_heads: int, scale: float,
                   k_new: torch.Tensor = None, v_new: torch.Tensor = None) -> torch.Tensor:
    """
    q_new: [num_q_heads, d_head]（decode 单 token）
    1. 取 K,V = kv_cache.get(layer_idx)
    2. GQA: 把 KV head 复制到 num_q_heads
    3. attn = softmax(q @ K^T * scale) @ V
    4. 把新 token 的 k,v append 进 cache
    5. return attn: [num_q_heads, d_head]
    """
    if k_new is not None and v_new is not None:
        kv_cache.append(layer_idx, k_new, v_new)
    K, V = kv_cache.get(layer_idx)
    num_kv_heads = kv_cache.num_kv_heads
    group_size = num_q_heads // num_kv_heads
    K_exp = K.unsqueeze(2).expand(-1, -1, group_size, -1).reshape(
        K.shape[0], num_q_heads, -1
    )
    V_exp = V.unsqueeze(2).expand(-1, -1, group_size, -1).reshape(
        V.shape[0], num_q_heads, -1
    )
    scores = torch.einsum("hd,shd->hs", q_new, K_exp) * scale
    attn_w = F.softmax(scores, dim=-1)
    out = torch.einsum("hs,shd->hd", attn_w, V_exp)
    return out


def prefill(prompt_ids: torch.Tensor, model: nn.Module, kv_cache: KVCache) -> torch.Tensor:
    """一次性算 prompt 各层 K/V 填入 cache，返回最后一个 token 的 logits"""
    with torch.no_grad():
        logits = model(prompt_ids, kv_cache=kv_cache)
    return logits[-1:]


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    num_layers, num_kv_heads, d_head, max_len = 2, 2, 16, 64
    cache = KVCache(num_layers, num_kv_heads, d_head, max_len)

    k_prefill = torch.randn(5, num_kv_heads, d_head)
    v_prefill = torch.randn(5, num_kv_heads, d_head)
    cache.append(0, k_prefill, v_prefill)
    cache.append(1, k_prefill, v_prefill)
    assert cache.seq_len == 5
    K, V = cache.get(0)
    assert K.shape == (5, num_kv_heads, d_head)
    assert torch.allclose(K, k_prefill), "prefill 内容不一致"
    print("✅ prefill: 一次写入 5 个 token")

    k_dec = torch.randn(1, num_kv_heads, d_head)
    v_dec = torch.randn(1, num_kv_heads, d_head)
    cache.append(0, k_dec, v_dec)
    cache.append(1, k_dec, v_dec)
    assert cache.seq_len == 6
    print("✅ decode: 逐 token append")

    num_q_heads = 4
    q = torch.randn(num_q_heads, d_head)
    out = attention_step(q, cache, layer_idx=0, num_q_heads=num_q_heads,
                         scale=1.0 / (d_head ** 0.5))
    assert out.shape == (num_q_heads, d_head), f"输出形状错误: {out.shape}"
    print(f"✅ attention_step (GQA {num_q_heads}/{num_kv_heads}): 输出 {out.shape}")

    cache2 = KVCache(num_layers, num_kv_heads, d_head, max_len)
    k1 = torch.randn(3, num_kv_heads, d_head)
    v1 = torch.randn(3, num_kv_heads, d_head)
    cache2.append(0, k1, v1)
    cache2.append(1, k1, v1)
    q2 = torch.randn(num_q_heads, d_head)
    k2 = torch.randn(1, num_kv_heads, d_head)
    v2 = torch.randn(1, num_kv_heads, d_head)
    out2 = attention_step(q2, cache2, 0, num_q_heads, 1.0 / (d_head ** 0.5),
                          k_new=k2, v_new=v2)
    assert cache2.seq_len == 4, "append 应在 attention_step 内执行"
    print("✅ attention_step 内 append 新 k/v 正确")
    print("✅ 全部测试通过")
