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
    """
    Interleaved 1F1B 调度。
    返回 [(chunk_id, micro_id, op_type), ...]
    """
    total_micro = num_micro * num_chunks
    warmup = (pp_size - rank - 1) * num_chunks
    warmup = min(warmup, total_micro)

    schedule = []
    fwd_count = 0
    bwd_count = 0

    for i in range(warmup):
        chunk = i // num_micro
        micro = i % num_micro
        schedule.append((chunk, micro, "F"))
        fwd_count += 1

    steady = total_micro - warmup
    for i in range(steady):
        f_chunk = (warmup + i) // num_micro
        f_micro = (warmup + i) % num_micro
        schedule.append((f_chunk, f_micro, "F"))
        fwd_count += 1

        b_chunk = i // num_micro
        b_micro = i % num_micro
        schedule.append((b_chunk, b_micro, "B"))
        bwd_count += 1

    for i in range(steady, total_micro):
        b_chunk = i // num_micro
        b_micro = i % num_micro
        schedule.append((b_chunk, b_micro, "B"))
        bwd_count += 1

    return schedule


def compute_bubble(pp_size: int, num_micro: int, num_chunks: int) -> float:
    """计算 bubble 占比。"""
    ideal = num_micro * num_chunks
    bubble = (pp_size - 1) / (num_chunks * num_micro + pp_size - 1)
    return bubble


# ===== 测试验证 =====
if __name__ == '__main__':
    pp, M, V = 4, 8, 2

    for r in range(pp):
        schedule = interleaved_1f1b_schedule(pp, M, V, r)
        f_count = sum(1 for x in schedule if x[2] == "F")
        b_count = sum(1 for x in schedule if x[2] == "B")
        assert f_count == M * V, f"rank {r}: F={f_count}, 期望 {M*V}"
        assert b_count == M * V, f"rank {r}: B={b_count}, 期望 {M*V}"
        print(f"  rank {r}: F={f_count}, B={b_count}, ops={len(schedule)}")
    print("✅ Interleaved 1F1B: 各 stage F/B 数量正确")

    s0 = interleaved_1f1b_schedule(pp, M, V, 0)
    assert s0[0][2] == "F", "第一个操作应为 F"
    assert s0[-1][2] == "B", "最后一个操作应为 B"
    print("✅ 调度顺序: F 开头 B 结尾")

    b1 = compute_bubble(pp, M, 1)
    b2 = compute_bubble(pp, M, 2)
    b4 = compute_bubble(pp, M, 4)
    assert b4 < b2 < b1, "V 越大 bubble 应越小"
    print(f"✅ Bubble: V=1 {b1:.3f}, V=2 {b2:.3f}, V=4 {b4:.3f}")

    s_simple = interleaved_1f1b_schedule(2, 4, 1, 0)
    assert len(s_simple) == 8
    print(f"✅ 简单配置 (PP=2,M=4,V=1): {len(s_simple)} ops")
    print("✅ 全部测试通过")
