"""
【题目】Prefix Caching / RadixAttention

【背景】
多请求共享相同 system prompt / few-shot 前缀时，重复 prefill 浪费算力。
把已算过的前缀 KV 按基数树（radix tree）缓存：按 token 序列分段建树，节点存对应 KV block 引用。
新请求沿树匹配最长公共前缀，直接复用其 KV，从分歧点开始 prefill 剩余 suffix 并挂到树上。
按 token 序列建树而非 hash 整段，是因为不同请求可在任意公共前缀处命中（部分共享也能复用）。
LRU 淘汰需配合引用计数：被多个在跑序列引用的节点不可驱逐。

【输入/输出】
- 输入：新请求的 token 序列
- 输出：命中前缀的 KV 引用 + 需 prefill 的 suffix；树被更新

【考察点】
- radix tree 的插入/匹配/分裂/删除
- KV block 与树节点的引用计数
- LRU 淘汰与引用安全
"""
from dataclasses import dataclass


@dataclass
class RadixNode:
    tokens: tuple            # 该节点对应的 token 段
    children: dict           # token -> RadixNode
    kv_blocks: list = None   # 对应 KV 的物理 block 引用
    ref: int = 0             # 引用计数
    last_used: int = 0       # LRU 时间戳


class RadixTree:
    def __init__(self, capacity_blocks: int):
        # TODO: root 节点 + LRU 顺序结构
        raise NotImplementedError

    def match(self, tokens: list):
        """
        从 root 沿子节点匹配，返回 (matched_node_path, matched_len, remaining_suffix)
        命中节点的 KV 引用计数 +1
        """
        raise NotImplementedError

    def insert(self, tokens: list, kv_blocks: list):
        """把新 suffix 挂到匹配终点，必要时分裂现有节点（部分前缀重合）"""
        raise NotImplementedError

    def release(self, node: RadixNode):
        """序列结束：沿路径 ref -1，ref 归零的节点进 LRU 候选"""
        raise NotImplementedError

    def evict(self, need_blocks: int):
        """按 LRU 淘汰 ref==0 的节点，释放其 KV block，注意不能破坏有引用的祖先链"""
        raise NotImplementedError


def serve_request(tree: RadixTree, tokens: list, model, block_pool):
    """match -> 复用 KV -> prefill suffix -> insert -> decode"""
    raise NotImplementedError
