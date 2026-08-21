"""
【题目】Disaggregated Serving：prefill/decode 分离部署

【背景】
传统 serving 中 prefill(计算密集)和 decode(访存密集)在同一 GPU 资源池，
互相干扰。Disaggregated Serving 把两者分到不同 GPU 池：
prefill pool 专注高算力，decode pool 专注高带宽。
KV Cache 通过网络从 prefill 传到 decode。优势：各池独立优化、消除干扰、提高吞吐。

【输入/输出】
- 输入：请求流, prefill_pool_size, decode_pool_size
- 输出：请求在 prefill pool 完成后转 decode pool

【考察点】
- KV Cache 传输开销与 overlap
- 负载均衡：prefill/decode 池配比
- 提示：模拟两池调度
"""
import torch
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Request:
    req_id: int
    prompt: list
    max_new: int
    output: list = field(default_factory=list)
    kv_cache: object = None
    stage: str = "prefill"


class DisaggregatedServing:
    def __init__(self, prefill_pool_size: int = 2, decode_pool_size: int = 4):
        self.prefill_pool = [None] * prefill_pool_size
        self.decode_pool = [None] * decode_pool_size
        self.prefill_queue = deque()
        self.decode_queue = deque()
        self.completed = []

    def add(self, req: torch.Tensor) -> None:
        self.prefill_queue.append(req)

    def step_prefill(self, model: nn.Module) -> int:
        """prefill pool 处理请求。"""
        for i in range(len(self.prefill_pool)):
            if self.prefill_pool[i] is None and self.prefill_queue:
                req = self.prefill_queue.popleft()
                x = torch.tensor([req.prompt], dtype=torch.long)
                with torch.no_grad():
                    logits = model(x)
                req.kv_cache = req.prompt.copy()
                req.stage = "decode"
                self.decode_queue.append(req)
                self.prefill_pool[i] = None
        return len(self.decode_queue)

    def step_decode(self, model: nn.Module) -> None:
        """decode pool 逐 token 生成。"""
        for i in range(len(self.decode_pool)):
            if self.decode_pool[i] is None and self.decode_queue:
                req = self.decode_queue.popleft()
                self.decode_pool[i] = req
            if self.decode_pool[i] is not None:
                req = self.decode_pool[i]
                x = torch.tensor([req.kv_cache[-1:]], dtype=torch.long)
                with torch.no_grad():
                    logits = model(x)
                nxt = torch.argmax(logits[0, -1]).item()
                req.output.append(nxt)
                req.kv_cache.append(nxt)
                if len(req.output) >= req.max_new:
                    self.completed.append(req)
                    self.decode_pool[i] = None

    def run(self, model: nn.Module, max_steps: int = 100) -> list:
        steps = 0
        while (self.prefill_queue or self.decode_queue or
               any(x is not None for x in self.decode_pool)) and steps < max_steps:
            self.step_prefill(model)
            self.step_decode(model)
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
    server = DisaggregatedServing(prefill_pool_size=2, decode_pool_size=2)
    server.add(Request(1, [1, 2, 3], max_new=5))
    server.add(Request(2, [4, 5], max_new=3))
    results = server.run(model, max_steps=50)
    assert len(results) == 2
    for r in results:
        assert len(r.output) == r.max_new
        print(f"  请求 {r.req_id}: {r.max_new} tokens")
    print("✅ Disaggregated serving: 2 请求完成")
    print("✅ 全部测试通过")
