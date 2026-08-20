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


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int, d_head: int):
        # TODO: 预分配物理块 [num_blocks, block_size, num_kv_heads, d_head] 的 k/v
        #       free_list + ref_count
        raise NotImplementedError

    def allocate(self) -> int:
        """返回一个空闲物理 block id；无空闲则报错或触发淘汰"""
        raise NotImplementedError

    def free(self, block_id: int):
        """引用计数 -1，归零则归还 free_list"""
        raise NotImplementedError

    def incref(self, block_id: int):
        raise NotImplementedError


class PagedKVCache:
    def __init__(self, pool: BlockPool):
        # TODO: seq_id -> block_table(List[int]) + 已写入 token 数
        raise NotImplementedError

    def append_token(self, seq_id: int, k: torch.Tensor, v: torch.Tensor):
        """写入新 token 的 k/v；当前 block 满则向 pool 申请新 block 挂到 block_table"""
        raise NotImplementedError

    def free_seq(self, seq_id: int):
        """序列结束，释放其所有 block（按引用计数）"""
        raise NotImplementedError


def paged_attention(q: torch.Tensor, block_table, num_valid_tokens: int,
                    pool: BlockPool, layer_idx: int, scale: float):
    """
    按 block_table 逐 block 取 K/V 拼成 [num_valid_tokens, ...]，
    做 scaled-dot-product attention，对末 block 未填满部分 mask。
    """
    raise NotImplementedError

# ===== 测试验证 =====
if __name__ == "__main__":
    print("10_paged_attention.py 测试代码：")
    try:
        # TODO: 用户实现后可在此调用核心函数验证输出形状与性质
        pass
        print("✅ 待实现核心函数后运行验证")
    except NotImplementedError:
        print("ℹ 核心函数待实现，可先阅读文件头部背景理解原理")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
