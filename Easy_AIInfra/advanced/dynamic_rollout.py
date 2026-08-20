"""
【题目】动态扩容 Rollout + In-flight Batching 调度

【背景】
LLM 推理服务中，请求到达时间不均、生成长度不同。动态 rollout 指：
不停服务、按需动态调整 batch 中序列的生成策略(如变长采样、提前终止)。
In-flight batching: 每步检查 batch 中是否有序列生成完(EOS/stop)，
立即移出并填入新请求，batch 大小动态变化。
核心数据结构：请求队列 + 活跃 batch 管理器 + slot 分配表。
调度策略：FCFS / SRPT(最短剩余处理时间优先) / 优先级队列。

【输入/输出】
- 输入：请求队列(Request(id,prompt,max_len,priority)), slot_table
- 输出：每 step 的调度决策(哪些 slot 继续生成，哪些释放)

【考察点】
- slot 管理：分配/释放/复用
- batch 拼接：不等长序列的 attention(variable-length flash attn)
- prefill 和 decode 混合调度
- 提示：collections.deque 做队列, torch.cat 拼接变长序列
"""
import torch
from collections import deque
from dataclasses import dataclass


@dataclass
class Request:
    id: int
    prompt: list
    max_len: int
    priority: int = 0
    generated: list = None

    def __post_init__(self):
        self.generated = self.generated or []


class DynamicBatchScheduler:
    def __init__(self, max_batch_size: int, max_seq_len: int):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.waiting = deque()
        self.slots = [None] * max_batch_size

    def add_request(self, req: Request):
        raise NotImplementedError

    def step(self) -> list:
        raise NotImplementedError

    def remove_finished(self, slot_idx: int):
        raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    scheduler = DynamicBatchScheduler(4, 256)
    for i in range(6):
        scheduler.add_request(Request(i, list(range(10, 10+i)), 128))
    try:
        active = scheduler.step()
        assert 0 < len(active) <= 4
        print('✅' + f" DynamicBatch 首步激活 {len(active)} slots")
    except NotImplementedError:
        print('ℹ' + " 待实现")
