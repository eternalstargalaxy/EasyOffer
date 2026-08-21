"""
【题目】Continuous Batching（动态批调度 / in-flight batching）

【背景】
静态 batch 下序列长度不齐，短序列结束后空等长序列，GPU 利用率低。
Continuous Batching 在每步 iteration 粒度动态拼批：新请求 prefill 完即加入 decode 队列，
完成的请求随时踢出，batch 内序列动态进出。显存预算通常以 KV block 数衡量，
调度器每步在预算内决定拉入哪些 waiting、是否 preempt（换出）哪些 running。

【输入/输出】
- 输入：请求流（prompt + max_tokens），token_budget / kv_block 预算
- 输出：每步动态拼批前向，各请求独立采样、独立结束

【考察点】
- 显存预算（KV block 数）与调度/preempt 决策
- 变长 batch 的 padding/unpadding 与结果按原 idx 回填
- prefill/decode 混排的优先级（prefill 算力大，会拖慢 decode）
- 提示：torch.cat 拼接不等长序列做 batch attention
"""
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Request:
    req_id: int
    prompt_ids: list
    max_tokens: int
    output_ids: list = field(default_factory=list)
    stage: str = "waiting"

    @property
    def all_tokens(self) -> list:
        return self.prompt_ids + self.output_ids

    @property
    def is_finished(self) -> bool:
        return len(self.output_ids) >= self.max_tokens


class Scheduler:
    """连续批处理调度器：waiting -> running -> done。"""

    def __init__(self, max_batch_size: int, eos_token: int = -1):
        self.max_batch_size = max_batch_size
        self.eos = eos_token
        self.waiting = deque()
        self.running = []
        self.completed = []

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    def schedule(self) -> list:
        """从 waiting 填入 running 空位，返回当前 batch。"""
        while len(self.running) < self.max_batch_size and self.waiting:
            req = self.waiting.popleft()
            req.stage = "running"
            self.running.append(req)
        return self.running

    def run_step(self, batch: list, model: nn.Module) -> None:
        """拼 padded batch 一次前向，各请求独立采样。"""
        if not batch:
            return
        tokens_list = [req.all_tokens for req in batch]
        max_len = max(len(t) for t in tokens_list)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, t in enumerate(tokens_list):
            padded[i, :len(t)] = torch.tensor(t)
        with torch.no_grad():
            logits = model(padded)
        still_running = []
        for i, req in enumerate(batch):
            nxt = torch.argmax(logits[i, len(req.all_tokens) - 1]).item()
            req.output_ids.append(nxt)
            if req.is_finished or nxt == self.eos:
                req.stage = "done"
                self.completed.append(req)
            else:
                still_running.append(req)
        self.running = still_running

    def run(self, model: nn.Module, max_steps: int = 1000) -> list:
        steps = 0
        while (self.waiting or self.running) and steps < max_steps:
            batch = self.schedule()
            self.run_step(batch, model)
            steps += 1
        return self.completed


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, tokens: list) -> torch.Tensor:
        return self.head(self.rnn(self.embed(tokens))[0])


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    vocab = 20
    model = TinyLM(vocab).eval()

    scheduler = Scheduler(max_batch_size=2, eos_token=0)
    scheduler.add_request(Request(1, [1, 2, 3], max_tokens=5))
    scheduler.add_request(Request(2, [4, 5], max_tokens=3))
    scheduler.add_request(Request(3, [6, 7, 8, 9], max_tokens=4))

    results = scheduler.run(model, max_steps=50)
    assert len(results) == 3, f"应完成 3 个请求，实际 {len(results)}"
    for req in results:
        assert req.is_finished or req.stage == "done"
        print(f"  请求 {req.req_id}: 输出 {len(req.output_ids)}/{req.max_tokens} tokens")
    print("✅ 3 个不等长请求全部完成")

    scheduler2 = Scheduler(max_batch_size=1, eos_token=0)
    scheduler2.add_request(Request(10, [1, 2], max_tokens=3))
    results2 = scheduler2.run(model, max_steps=20)
    assert len(results2) == 1
    print("✅ batch_size=1 正常工作")

    scheduler3 = Scheduler(max_batch_size=2, eos_token=-1)
    for i in range(5):
        scheduler3.add_request(Request(i, [1, 2], max_tokens=3))
    results3 = scheduler3.run(model, max_steps=50)
    assert len(results3) == 5, f"排队测试失败: {len(results3)}"
    print("✅ 请求数 > max_batch_size 时排队等待正确")
    print("✅ 全部测试通过")
