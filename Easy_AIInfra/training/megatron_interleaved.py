"""
【题目】Megatron：Interleaved 1F1B 流水调度

【背景】
标准 1F1B 每个 pipeline stage 跑连续层(L/PP)，bubble=PP-1。
Interleaved 1F1B 把每个 stage 的 L 层拆成 V 个 chunk(交错排布)，
如 stage0 跑 layer0,4,8; stage1 跑 layer1,5,9; stage2 跑 layer2,6,10...
bubble 降为 (PP-1)/(V*M+PP-1)，V 越大 bubble 越小，但通信次数 *V。
Megatron-LM 默认 V=PP，bubble 为 PP 拆分前的一半。

【输入/输出】
- 输入：PP, num_microbatches, num_model_chunks=V
- 输出：每 step 各 stage 的调度指令 [(stage, micro, F/B), ...]

【考察点】
- V 与 bubble/通信的 trade-off
- interleaved 模型 chunk 分配算法
- 提示：循环调度，同 micro 不同 stage 间 F/B 流式传递
"""
from collections import deque


def interleaved_1f1b_schedule(pp_size: int, num_micro: int,
                               num_chunks: int, rank: int) -> list:
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    pp, M, V = 4, 8, 2
    for r in range(pp):
        schedule = interleaved_1f1b_schedule(pp, M, V, r)
        assert len(schedule) > 0
    try:
        s = interleaved_1f1b_schedule(4, 8, 2, 0)
        f_count = sum(1 for x in s if x[2] == "F")
        b_count = sum(1 for x in s if x[2] == "B")
        assert f_count == b_count + pp - 1
        print('✅' + " Interleaved 1F1B 测试通过")
    except NotImplementedError:
        print('ℹ' + " 待实现")
