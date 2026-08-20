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
from dataclasses import dataclass


@dataclass
class Request:
    req_id: int
    prompt_ids: list
    max_tokens: int
    output_ids: list = None
    stage: str = "waiting"   # waiting(prefill) / running(decode) / done


class Scheduler:
    def __init__(self, kv_block_budget: int, block_size: int):
        # TODO: waiting deque, running list, 已用 kv block 计数
        raise NotImplementedError

    def add_request(self, req: Request):
        raise NotImplementedError

    def schedule(self) -> list:
        """
        返回本 step 要执行的 batch：
          1. 优先把 waiting 中 prefill 拉入（受预算限制）
          2. 再把 running decode 拼入
          3. 预算不足时 preempt 末尾 running（换回 waiting）
        """
        raise NotImplementedError

    def run_step(self, batch: list, model, kv_cache):
        """
        1. 把 batch 拼成 padded tensor 一次前向
        2. 按各 req 的 sequence_len 取对应 logit，各自采样
        3. 写回 KV cache，append token；完成的标 done 并释放 KV
        """
        raise NotImplementedError

# ===== 测试验证 =====
if __name__ == "__main__":
    print("12_continuous_batching.py 测试代码：")
    try:
        # TODO: 用户实现后可在此调用核心函数验证输出形状与性质
        pass
        print("✅ 待实现核心函数后运行验证")
    except NotImplementedError:
        print("ℹ 核心函数待实现，可先阅读文件头部背景理解原理")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
