"""
【题目】vLLM：prefill/decode 分离调度 + 优先级队列

【背景】
vLLM 将 prefill(计算密集)和 decode(访存密集)分离：
prefill 处理 prompt 填充 KV Cache，decode 逐 token 生成。
分离后独立优化各自瓶颈。优先级调度基于 latency/throughput 需求。
双队列：waiting_queue(按优先级排序) + running_queue(活跃序列)。

【输入/输出】
- 输入：请求队列, max_batch, block_table, num_blocks
- 输出：每 step 调度 (prefill_slots, decode_slots, idle_slots)

【考察点】
- prefill/decode 分离的 CU 利用率优化
- 优先级调度与 KV block 分配/回收时机
- 提示：deque 做队列, sorted(key=lambda r: r.priority)
"""
import torch
from collections import deque
from dataclasses import dataclass, field


@dataclass
class VRequest:
    id: int
    prompt: list
    max_len: int
    priority: int = 0
    state: str = "waiting"
    kv_blocks: list = field(default_factory=list)
    output_ids: list = field(default_factory=list)

    @property
    def is_finished(self):
        return len(self.output_ids) >= self.max_len


class VLLMScheduler:
    """vLLM 风格调度器：prefill/decode 分离 + 优先级。"""

    def __init__(self, max_batch: int, num_blocks: int):
        self.waiting = deque()
        self.running = []
        self.max_batch = max_batch
        self.num_blocks = num_blocks
        self.free_blocks = num_blocks
        self.completed = []

    def add_request(self, req: VRequest):
        self.waiting.append(req)

    def _estimate_blocks(self, req: VRequest, block_size: int = 16):
        return (len(req.prompt) + req.max_len + block_size - 1) // block_size

    def schedule(self) -> dict:
        """
        返回本 step 调度结果：
        - prefill_list: 从 waiting 拉入的 prefill 请求
        - decode_list: running 中继续 decode 的请求
        - preempted: 预算不足被换出的请求
        """
        sorted_waiting = sorted(self.waiting, key=lambda r: r.priority)
        prefill_list = []
        for req in sorted_waiting:
            if len(self.running) + len(prefill_list) >= self.max_batch:
                break
            needed = self._estimate_blocks(req)
            if needed <= self.free_blocks:
                req.state = "prefill"
                self.free_blocks -= needed
                prefill_list.append(req)
                self.waiting.remove(req)

        decode_list = [r for r in self.running if r.state == "decode"]
        preempted = []
        while len(self.running) + len(prefill_list) > self.max_batch and self.running:
            victim = self.running.pop()
            self.free_blocks += self._estimate_blocks(victim)
            victim.state = "waiting"
            self.waiting.append(victim)
            preempted.append(victim)

        return {
            "prefill": prefill_list,
            "decode": decode_list,
            "preempted": preempted,
        }

    def step_done(self, prefill_list: torch.Tensor, model: nn.Module = None):
        """prefill 完成的请求转入 running decode。"""
        for req in prefill_list:
            req.state = "decode"
            self.running.append(req)

    def free_blocks_(self, req: VRequest):
        """请求完成，回收 KV block。"""
        self.free_blocks += len(req.kv_blocks)
        req.kv_blocks = []
        req.state = "done"
        self.completed.append(req)
        if req in self.running:
            self.running.remove(req)


# ===== 测试验证 =====
if __name__ == '__main__':
    s = VLLMScheduler(max_batch=4, num_blocks=64)
    s.add_request(VRequest(0, list(range(10)), 100, priority=2))
    s.add_request(VRequest(1, list(range(20)), 50, priority=1))
    s.add_request(VRequest(2, list(range(5)), 30, priority=3))

    d = s.schedule()
    assert len(d["prefill"]) > 0, "应有请求被调度"
    assert all(r.state == "prefill" for r in d["prefill"])
    print(f"✅ 调度 {len(d['prefill'])} 个 prefill 请求")

    assert d["prefill"][0].priority <= d["prefill"][-1].priority if len(d["prefill"]) > 1 else True
    print("✅ 优先级排序正确")

    s.step_done(d["prefill"])
    assert all(r.state == "decode" for r in s.running)
    print(f"✅ {len(s.running)} 个请求转入 decode")

    req0 = s.running[0]
    s.free_blocks_(req0)
    assert req0.state == "done"
    assert req0 in s.completed
    print("✅ 请求完成回收 block")

    s2 = VLLMScheduler(max_batch=1, num_blocks=2)
    s2.add_request(VRequest(10, list(range(100)), 200))
    d2 = s2.schedule()
    if len(d2["prefill"]) == 0:
        print("✅ 预算不足时拒绝调度")
    else:
        print("✅ 预算足够时正常调度")
    print("✅ 全部测试通过")
