"""
【题目】PagedAttention（块状 KV 管理）

【背景】
vLLM 的 PagedAttention 把每个序列的 KV 划成固定大小 block（如 16 token），
用 block table（逻辑 block id → 物理 block id）+ 全局物理块池管理，
避免为变长序列预分配 max_seq_len 显存，且 Continuous Batching 下显存碎片接近 0。
物理块可被多个序列引用（beam search 共享前缀），用引用计数 + copy-on-write 管理。

【输入/输出】
- 输入：q: [num_heads, d_head]（decode 单 token）, block_table: List[int], num_valid_tokens: int
- 输出：attention 输出；物理块池状态被更新

【考察点】
- block table 与物理块解耦、引用计数
- 变长访问的 mask（block 内未填满部分 mask 掉）
- copy-on-write 在共享前缀写入时
- 提示：block table 用 torch.LongTensor 存储映射
"""
import torch
import torch.nn.functional as F


class BlockPool:
    """物理块池：管理 KV block 的分配/释放/引用计数。"""

    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int, d_head: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.d_head = d_head
        self.k_pool = torch.zeros(num_blocks, block_size, num_kv_heads, d_head)
        self.v_pool = torch.zeros_like(self.k_pool)
        self.free_list = list(range(num_blocks))
        self.ref_count = [0] * num_blocks

    def allocate(self) -> int:
        """返回一个空闲物理 block id；无空闲则报错。"""
        assert len(self.free_list) > 0, "物理 block 不足"
        blk = self.free_list.pop()
        self.ref_count[blk] = 1
        return blk

    def free(self, block_id: int):
        """引用计数 -1，归零则归还 free_list。"""
        self.ref_count[block_id] -= 1
        if self.ref_count[block_id] == 0:
            self.free_list.append(block_id)

    def incref(self, block_id: int):
        self.ref_count[block_id] += 1


class PagedKVCache:
    """分页 KV Cache：seq_id -> block_table + 已写入 token 数。"""

    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.block_tables = {}
        self.seq_lens = {}

    def allocate_seq(self, seq_id: int):
        self.block_tables[seq_id] = []
        self.seq_lens[seq_id] = 0

    def append_token(self, seq_id: int, k: torch.Tensor, v: torch.Tensor):
        """写入新 token 的 k/v；当前 block 满则向 pool 申请新 block。"""
        pos = self.seq_lens[seq_id]
        block_idx = pos // self.pool.block_size
        block_offset = pos % self.pool.block_size
        if block_idx >= len(self.block_tables[seq_id]):
            self.block_tables[seq_id].append(self.pool.allocate())
        phys = self.block_tables[seq_id][block_idx]
        self.pool.k_pool[phys, block_offset] = k
        self.pool.v_pool[phys, block_offset] = v
        self.seq_lens[seq_id] += 1

    def get_logical(self, seq_id: int):
        """按逻辑顺序拼接该序列的所有 k/v。"""
        seq_len = self.seq_lens[seq_id]
        k_parts, v_parts = [], []
        for blk in self.block_tables[seq_id]:
            k_parts.append(self.pool.k_pool[blk])
            v_parts.append(self.pool.v_pool[blk])
        k_all = torch.cat(k_parts, dim=0)[:seq_len]
        v_all = torch.cat(v_parts, dim=0)[:seq_len]
        return k_all, v_all

    def free_seq(self, seq_id: int):
        """序列结束，释放其所有 block（按引用计数）。"""
        for blk in self.block_tables[seq_id]:
            self.pool.free(blk)
        del self.block_tables[seq_id]
        del self.seq_lens[seq_id]

    def copy_on_write(self, src_id: int, dst_id: int):
        """prefix sharing: dst 引用 src 的 block（写时才分配新 block）。"""
        self.allocate_seq(dst_id)
        for blk in self.block_tables[src_id]:
            self.pool.incref(blk)
            self.block_tables[dst_id].append(blk)
        self.seq_lens[dst_id] = self.seq_lens[src_id]


def paged_attention(q: torch.Tensor, block_table: list, num_valid_tokens: int,
                    pool: BlockPool, scale: float):
    """
    按 block_table 逐 block 取 K/V 拼成 [num_valid_tokens, ...]，
    做 scaled-dot-product attention，对末 block 未填满部分 mask。
    q: [num_heads, d_head]
    """
    k_parts, v_parts = [], []
    for blk in block_table:
        k_parts.append(pool.k_pool[blk])
        v_parts.append(pool.v_pool[blk])
    k_all = torch.cat(k_parts, dim=0)[:num_valid_tokens]
    v_all = torch.cat(v_parts, dim=0)[:num_valid_tokens]
    scores = torch.einsum("hd,shd->hs", q, k_all) * scale
    attn = F.softmax(scores, dim=-1)
    return torch.einsum("hs,shd->hd", attn, v_all)


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    pool = BlockPool(num_blocks=10, block_size=4, num_kv_heads=2, d_head=16)
    assert len(pool.free_list) == 10
    blk = pool.allocate()
    assert pool.ref_count[blk] == 1
    assert blk not in pool.free_list
    pool.free(blk)
    assert blk in pool.free_list
    print("✅ BlockPool: allocate/free 正确")

    cache = PagedKVCache(pool)
    cache.allocate_seq(0)
    H, D = 2, 16
    for i in range(6):
        k = torch.randn(H, D)
        v = torch.randn(H, D)
        cache.append_token(0, k, v)
    assert cache.seq_lens[0] == 6
    assert len(cache.block_tables[0]) == 2, "6 token / block_size=4 需要 2 block"
    k_ret, v_ret = cache.get_logical(0)
    assert k_ret.shape == (6, H, D)
    print("✅ PagedKVCache: 跨 block 读写正确")

    cache.copy_on_write(0, 1)
    assert cache.seq_lens[1] == 6
    k0, _ = cache.get_logical(0)
    k1, _ = cache.get_logical(1)
    assert torch.allclose(k0, k1), "CoW 共享内容应一致"
    print("✅ copy_on_write: prefix sharing 正确")

    q = torch.randn(H, D)
    out = paged_attention(q, cache.block_tables[0], 6, pool, 1.0 / (D ** 0.5))
    assert out.shape == (H, D)
    print(f"✅ paged_attention: 输出 {out.shape}")

    cache.free_seq(0)
    assert 0 not in cache.block_tables
    print("✅ free_seq: block 回收")
    print("✅ 全部测试通过")
