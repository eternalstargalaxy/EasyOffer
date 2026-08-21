"""
【题目】Prefix Caching：共享前缀 KV Cache 复用

【背景】
多请求共享相同 system prompt 时，重复 prefill 浪费算力。
Prefix Caching 把 system prompt 的 KV Cache 缓存，新请求复用：
1. 检测新请求前缀是否匹配已缓存 prefix
2. 匹配则直接从 cache 末尾开始 decode，跳过 prefill
3. 不匹配则 prefill 并缓存新 prefix KV
vLLM 用 hash(prefix) 做 key，自动匹配复用。

【输入/输出】
- 输入：请求 prompt, prefix_cache
- 输出：复用或新建 KV Cache

【考察点】
- prefix 匹配与 hash
- KV Cache 引用计数与淘汰
- 提示：dict 存 prefix_hash -> kv_cache
"""
import torch
import torch.nn as nn
import hashlib
from collections import OrderedDict


class PrefixCache:
    """LRU prefix KV Cache。"""

    def __init__(self, max_entries: int = 16):
        self.cache = OrderedDict()
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _hash(tokens: list) -> str:
        return hashlib.md5(str(tokens).encode()).hexdigest()

    def get(self, prefix: list) -> torch.Tensor:
        key = self._hash(prefix)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, prefix: list, kv_cache: torch.Tensor) -> None:
        key = self._hash(prefix)
        self.cache[key] = kv_cache
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": hit_rate}


class PrefixCachingServer:
    def __init__(self, model: nn.Module, max_cache: int = 16):
        self.model = model
        self.prefix_cache = PrefixCache(max_cache)

    def generate(self, prompt: torch.Tensor, max_new: int = 5) -> list:
        """带 prefix caching 的生成。"""
        cached = self.prefix_cache.get(prompt)
        if cached is not None:
            tokens = list(prompt) + cached["output"]
        else:
            tokens = list(prompt)
            x = torch.tensor([tokens], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(x)
            output = []
            for _ in range(max_new):
                x = torch.tensor([tokens], dtype=torch.long)
                with torch.no_grad():
                    logits = self.model(x)
                nxt = torch.argmax(logits[0, -1]).item()
                tokens.append(nxt)
                output.append(nxt)
            self.prefix_cache.put(prompt, {"output": output})
        return tokens


class TinyLM(nn.Module):
    def __init__(self, vocab: int, hidden: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.rnn(self.embed(x))[0])


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    vocab = 20
    model = TinyLM(vocab)
    server = PrefixCachingServer(model, max_cache=4)

    prompt1 = [1, 2, 3]
    result1 = server.generate(prompt1, max_new=3)
    assert server.prefix_cache.misses == 1
    print("✅ 首次请求: miss")

    result2 = server.generate(prompt1, max_new=3)
    assert server.prefix_cache.hits == 1
    assert result2 == result1
    print("✅ 相同前缀: hit, 结果复用")

    prompt3 = [4, 5, 6]
    server.generate(prompt3, max_new=3)
    assert server.prefix_cache.misses == 2
    print("✅ 不同前缀: miss")

    for i in range(10):
        server.generate([i, i+1], max_new=2)
    stats = server.prefix_cache.stats()
    assert stats["hit_rate"] > 0
    print(f"✅ 统计: {stats}")

    for i in range(20):
        server.generate([i, i+1, i+2], max_new=1)
    assert len(server.prefix_cache.cache) <= 4
    print(f"✅ LRU 淘汰: cache size <= {server.prefix_cache.max_entries}")
    print("✅ 全部测试通过")
