"""
【题目】Chunked Prefill 与 decode 混排

【背景】
prefill 是 compute-bound（长 prompt 一次算），decode 是 memory-bound（每步 1 token）。
同 batch 混入 prefill 会拖慢 decode 的 token 速率（prefill 占算力大）。
Chunked Prefill 把长 prefill 切成固定 chunk（如 512 token），与 decode 共享 step，
既复用算力又限制 prefill 对 decode 的干扰。chunk 间 KV 需连续：前一 chunk 末尾的 KV 是后一 chunk 的前缀。
chunk_size 大 → prefill 吞吐高但 decode TPOT 抖动大；小 → 抖动小但 prefill 调度开销大。

【输入/输出】
- 输入：长 prompt 请求，chunk_size，token budget
- 输出：prompt 被切成多 chunk 逐 step 执行，与 decode 混排，最终产出 token

【考察点】
- prefill chunk 与 decode 的优先级与 budget 分配
- chunk 边界 KV 衔接（前 chunk 末态作为后 chunk 前缀）
- 对 TTFT（首 token 延迟）与 TPOT（逐 token 延迟）的影响
- 提示：FlashAttention 分块处理；把长 prompt 拆成 chunk 逐步 prefill

"""
from dataclasses import dataclass


@dataclass
class PrefillChunk:
    req_id: int
    tokens: list            # 该 chunk 的 token
    prefix_kv_len: int      # 前 chunk 已积累的 KV 长度（衔接用）
    is_last: bool


class ChunkedScheduler:
    def __init__(self, chunk_size: int, token_budget: int):
        # TODO: decode 队列 + prefill chunk 队列
        raise NotImplementedError

    def split_prefill(self, req_id: int, prompt: list):
        """把 prompt 切成多个 PrefillChunk 入队，记录前缀 KV 长度"""
        raise NotImplementedError

    def schedule(self) -> list:
        """
        每步在 token_budget 内混排：
          - decode 请求优先（保 TPOT）
          - 剩余预算塞 prefill chunk（受 prefill 占比上限）
        """
        raise NotImplementedError

    def run_step(self, batch: list, model, kv_cache):
        """
        prefill chunk: 用 prefix_kv_len 接续 KV，算该 chunk KV 并 append，最后一个 chunk 产出首 token
        decode: 逐 token 推进
        """
        raise NotImplementedError


def ttft_tpot_vs_chunk_size(chunk_sizes: list):
    """返回不同 chunk_size 下的 TTFT/TPOT 曲线"""
    raise NotImplementedError

# ===== 测试验证 =====
if __name__ == "__main__":
    print("17_chunked_prefill.py 测试代码：")
    try:
        # TODO: 用户实现后可在此调用核心函数验证输出形状与性质
        pass
        print("✅ 待实现核心函数后运行验证")
    except NotImplementedError:
        print("ℹ 核心函数待实现，可先阅读文件头部背景理解原理")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
