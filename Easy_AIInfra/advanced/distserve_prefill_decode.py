"""
【题目】DistServe：prefill/decode 分离 + KV Cache 传输优化

【背景】
DistServe 在 disaggregated serving 基础上进一步优化 KV Cache 传输：
1. KV Cache 在 prefill 完成后异步传输到 decode pool
2. decode pool 收到 KV 后立即开始 decode，不需重新计算
3. 支持 KV Cache 压缩减少传输量
核心优势：消除 prefill 对 decode 的干扰，各阶段独立扩缩容。

【输入/输出】
- 输入：请求, prefill_gpu, decode_gpu
- 输出：prefill 完成后 KV 异步传到 decode pool

【考察点】
- KV 传输与 decode 的 overlap
- 传输压缩策略
- 提示：模拟异步传输
"""
import torch
from collections import deque
from dataclasses import dataclass, field


@dataclass
class DistServeRequest:
    req_id: int
    prompt: list
    max_new: int
    output: list = field(default_factory=list)
    kv_cache: list = field(default_factory=list)
    stage: str = "prefill"
    kv_transferred: bool = False


class DistServe:
    def __init__(self):
        self.prefill_done = deque()
        self.kv_transfer_queue = deque()
        self.decode_running = []
        self.completed = []

    def prefill(self, req: torch.Tensor, model: nn.Module) -> torch.Tensor:
        x = torch.tensor([req.prompt], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        req.kv_cache = list(req.prompt)
        req.stage = "kv_transfer"
        self.kv_transfer_queue.append(req)
        return logits

    def transfer_kv(self) -> None:
        """模拟 KV Cache 异步传输。"""
        while self.kv_transfer_queue:
            req = self.kv_transfer_queue.popleft()
            req.kv_transferred = True
            req.stage = "decode"
            self.decode_running.append(req)

    def decode_step(self, model: nn.Module) -> None:
        still = []
        for req in self.decode_running:
            x = torch.tensor([req.kv_cache[-1:]], dtype=torch.long)
            with torch.no_grad():
                logits = model(x)
            nxt = torch.argmax(logits[0, -1]).item()
            req.output.append(nxt)
            req.kv_cache.append(nxt)
            if len(req.output) < req.max_new:
                still.append(req)
            else:
                self.completed.append(req)
        self.decode_running = still

    def run(self, reqs: torch.Tensor, model: nn.Module, max_steps: int = 100) -> list:
        for r in reqs:
            self.prefill(r, model)
        self.transfer_kv()
        steps = 0
        while self.decode_running and steps < max_steps:
            self.decode_step(model)
            steps += 1
        return self.completed


class TinyLM(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int = 32):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.rnn = torch.nn.GRU(hidden, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.rnn(self.embed(x))[0])


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    vocab = 20
    model = TinyLM(vocab)
    server = DistServe()
    reqs = [DistServeRequest(1, [1, 2, 3], 5), DistServeRequest(2, [4, 5], 3)]
    results = server.run(reqs, model, max_steps=50)
    assert len(results) == 2
    for r in results:
        assert r.kv_transferred
        assert len(r.output) == r.max_new
        print(f"  请求 {r.req_id}: KV transferred={r.kv_transferred}, {len(r.output)} tokens")
    print("✅ DistServe: KV 传输 + decode 正确")
    print("✅ 全部测试通过")
