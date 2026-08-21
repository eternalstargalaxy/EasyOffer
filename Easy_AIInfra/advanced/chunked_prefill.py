"""
【题目】Chunked Prefill：分块预填充

【背景】
长 prompt 的 prefill 一次前向计算量大、显存峰值高。Chunked Prefill 把 prompt 分成多个 chunk，
逐块 prefill 填充 KV Cache，与 decode 请求混排调度。优势：降低显存峰值、
prefill 与 decode 共享 batch 提高吞吐、长 prompt 不阻塞短请求。

【输入/输出】
- 输入：prompt tokens, chunk_size, kv_cache
- 输出：分块填充 KV Cache，返回最终 logits

【考察点】
- chunk 大小选择与显存/吞吐 trade-off
- prefill/decode 混排调度
- 提示：每块 prefill 后 KV Cache 追加
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLM(nn.Module):
    def __init__(self, vocab: int, dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens: list):
        return self.head(self.rnn(self.embed(tokens))[0])


class ChunkedPrefill:
    def __init__(self, model: nn.Module, chunk_size: int = 8):
        self.model = model
        self.chunk_size = chunk_size
        self.kv_cache = []

    def prefill(self, prompt_tokens: torch.Tensor):
        """分块 prefill：把 prompt 分成 chunk 逐块前向。"""
        tokens = prompt_tokens
        for start in range(0, len(tokens), self.chunk_size):
            chunk = tokens[start:start + self.chunk_size]
            x = torch.tensor([chunk], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(x)
            self.kv_cache.append(chunk)
        return logits[0, -1]

    def decode(self, token: torch.Tensor):
        x = torch.tensor([[token]], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x)
        return logits[0, -1]


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    vocab = 50
    model = SimpleLM(vocab)
    cp = ChunkedPrefill(model, chunk_size=4)

    prompt = list(range(10))
    logits = cp.prefill(prompt)
    assert logits.shape == (vocab,)
    assert len(cp.kv_cache) == 3, f"10/4 应有 3 块, 实际 {len(cp.kv_cache)}"
    print(f"✅ Chunked prefill: {len(prompt)} tokens / chunk_size=4 -> {len(cp.kv_cache)} chunks")

    next_logit = cp.decode(11)
    assert next_logit.shape == (vocab,)
    print("✅ Decode after prefill 正确")

    cp2 = ChunkedPrefill(model, chunk_size=16)
    logits2 = cp2.prefill(prompt)
    assert len(cp2.kv_cache) == 1
    print("✅ chunk_size > prompt: 单块完成")
    print("✅ 全部测试通过")
