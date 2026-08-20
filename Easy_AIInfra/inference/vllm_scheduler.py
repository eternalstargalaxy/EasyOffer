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
    id: int; prompt: list; max_len: int
    priority: int = 0
    state: str = "waiting"
    kv_blocks: list = field(default_factory=list)


class VLLMScheduler:
    def __init__(self, max_batch: int, num_blocks: int):
        self.waiting = deque(); self.running = []
        self.max_batch = max_batch; self.num_blocks = num_blocks

    def add_request(self, req: VRequest):
        raise NotImplementedError

    def schedule(self) -> dict:
        raise NotImplementedError

    def free_blocks(self, req: VRequest):
        raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    s = VLLMScheduler(4, 64)
    s.add_request(VRequest(0, list(range(10)), 100, priority=2))
    s.add_request(VRequest(1, list(range(20)), 50, priority=1))
    try:
        d = s.schedule()
        assert len(d) > 0
        print('✅' + " 调度测试通过")
    except NotImplementedError:
        print('ℹ' + " 待实现")
