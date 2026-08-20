"""
【题目】Disaggregated Serving：prefill/decode 分离部署

【背景】
prefill(计算密集)和 decode(访存密集)瓶颈不同。
Disaggregated serving：将 prefill 和 decode 部署在不同 GPU/节点，
prefill node 算完 prompt 后把 KV Cache 通过网络传给 decode node。
优势：prefill 和 decode 独立扩缩容(取决于流量比例)，
资源利用率更高(不用在一台机器上预留 peer GPU)。
Splitwise：按 GPU 计算+显存的 Pareto 前端分配比例。
Mooncake(KV Cache 传输)：RDMA+NVMe 先缓存 KV 再传给 decode。
DistServe：用 profiling 决定 prefill/decode GPU 的最优分配。

【输入/输出】
- 输入：请求流, prefill_nodes, decode_nodes, 网络拓扑
- 输出：每请求的 prefill/decode 路由决策 + KV 传输调度

【考察点】
- prefill/decode 节点比例求解(基于 latency SLO)
- KV Cache 跨节点传输效率(RDMA vs NVLink)
- 提示：torch.distributed.send/recv 传 KV tensor
"""
import torch; from collections import deque


class DisaggregatedScheduler:
    def __init__(self, n_prefill: int, n_decode: int):
        self.n_prefill = n_prefill; self.n_decode = n_decode
        self.prefill_queue = deque()
        self.decode_slots = [None] * n_decode

    def dispatch_prefill(self, request) -> int:
        raise NotImplementedError

    def transfer_kv(self, kv_cache, from_node: int, to_node: int):
        raise NotImplementedError

    def schedule_step(self) -> dict:
        raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    print('ℹ' + " Disaggregated Serving 需多节点环境")
    print("验证：prefill/decode 节点数随流量动态变化")
    print("验证：KV Cache 传输延迟 < 推理延迟的 20%")
