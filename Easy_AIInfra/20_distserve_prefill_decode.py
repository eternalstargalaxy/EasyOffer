"""
【题目】Prefill/Decode 分离部署（Disaggregated Serving / DistServe）

【背景】
prefill（compute-bound、长 prompt）与 decode（memory-bound、逐 token）资源画像相反，
混部时 prefill 的大算力突发会挤占 decode，造成 decode 的 TPOT 抖动（长尾）。
分离部署把 prefill 与 decode 放到不同实例组：prefill 实例算完一个请求产出 KV，
把 KV 通过 RDMA/共享存储迁到某个 decode 实例，decode 实例接续解码并参与 continuous batching。
迁移开销与 decode 可重叠（迁下一条的同时 decode 当前条）；但若请求短、KV 迁移开销占比大，分离反而变差。

【输入/输出】
- 输入：请求流（prompt + max_tokens），prefill/decode 实例数
- 输出：prefill 实例产出 KV → 迁移 → decode 实例接续解码 → 返回 token

【考察点】
- prefill/decode 资源画像与隔离收益
- KV 迁移开销与重叠（迁移与 decode 并行）
- 实例间路由与负载均衡、迁移时机
"""
from dataclasses import dataclass
from queue import Queue


@dataclass
class Request:
    req_id: int
    prompt_ids: list
    max_tokens: int
    kv_handle: object = None   # 迁移用的 KV 句柄
    output: list = None


class PrefillWorker:
    def __init__(self, worker_id: int, kv_store):
        raise NotImplementedError

    def prefill(self, req: Request, model):
        """算 prompt 的 KV，存到 kv_handle，发往某 decode worker"""
        raise NotImplementedError


class KVTransfer:
    def __init__(self, bandwidth: float):
        # TODO: 模拟 RDMA/共享存储，按 KV 大小算迁移时延
        raise NotImplementedError

    def send(self, kv_handle, dst_decode_id: int):
        raise NotImplementedError

    def recv(self) -> object:
        """返回收到的 kv_handle"""
        raise NotImplementedError


class DecodeWorker:
    def __init__(self, worker_id: int, kv_store):
        # TODO: 维护 running 队列做 continuous batching
        raise NotImplementedError

    def load_kv(self, kv_handle):
        """把迁来的 KV 装进本地 cache，加入 running 队列"""
        raise NotImplementedError

    def run_step(self, model):
        """continuous batching decode 一步"""
        raise NotImplementedError


class DistServeController:
    def __init__(self, prefill_workers: list, decode_workers: list, transfer: KVTransfer):
        raise NotImplementedError

    def route(self, req: Request):
        """选一个 prefill worker；prefill 完后选一个 decode worker 迁 KV"""
        raise NotImplementedError

    def run(self, request_stream):
        raise NotImplementedError


def compare_colocated_vs_disaggregated(num_req: int):
    """返回混部 vs 分离在 P99 TTFT / P99 TPOT / 吞吐 上的对比"""
    raise NotImplementedError
