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
"""
import torch


class KVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, d_head: int,
                 max_seq_len: int, dtype=torch.float16):
        # TODO: 预分配 [num_layers, max_seq_len, num_kv_heads, d_head] 的 k/v buffer
        #       维护已写入长度 seq_len
        raise NotImplementedError

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """把新 k/v 写入 buffer 末尾，更新 seq_len。支持 prefill（多 token）与 decode（1 token）"""
        raise NotImplementedError

    def get(self, layer_idx: int):
        """返回该层已写入部分的 K, V: [cur_len, num_kv_heads, d_head]"""
        raise NotImplementedError


def attention_step(q_new: torch.Tensor, kv_cache: KVCache, layer_idx: int,
                   num_q_heads: int, scale: float):
    """
    q_new: [num_q_heads, d_head]（decode 单 token）
    1. 取 K,V = kv_cache.get(layer_idx)
    2. GQA: 把 KV head 复制到 num_q_heads
    3. attn = softmax(q @ K^T * scale) @ V
    4. 把新 token 的 k,v append 进 cache
    5. return attn: [num_q_heads, d_head]
    """
    raise NotImplementedError


def prefill(prompt_ids, model, kv_cache: KVCache):
    """一次性算 prompt 各层 K/V 填入 cache，返回最后一个 token 的 logits"""
    raise NotImplementedError
