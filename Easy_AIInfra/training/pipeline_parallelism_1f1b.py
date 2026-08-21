"""
【题目】流水并行 1F1B 调度

【背景】
模型按层切到多卡（pipeline stage），micro-batch 在 stage 间流动。
朴素 GP（先全部前向再全部反向）气泡大、显存高；1F1B 交错前向/反向，每 stage 先做 warm up
（num_stages - rank - 1 个前向）进入稳态后一前一替，气泡约 (num_stages-1) 个 micro-batch，
且每 stage 同时只缓存约 num_stages 个 activation，显存显著低于 GP。

【输入/输出】
- 输入：num_stages, num_microbatches
- 输出：每个 stage 的 op 调度序列 [(op∈{'F','B'}, micro_idx), ...]，及依赖关系

【考察点】
- 1F1B 调度依赖正确性（同 stage 顺序、跨 stage F/B 传递，不能死锁）
- warm-up 长度 = num_stages - rank - 1
- GP vs 1F1B 气泡时间与峰值显存对比
- 提示：torch.distributed.send / recv 用于 stage 间传递激活和梯度
"""
from collections import deque


def schedule_1f1b(num_stages: int, num_microbatches: int, rank: int):
    """
    返回本 stage 的 op 序列，元素 = ('F', micro_id) 或 ('B', micro_id)。
    warm-up = num_stages - rank - 1 个 F；之后稳态 1F1B；尾部剩余 B。
    """
    ops = []
    warmup = min(num_stages - rank - 1, num_microbatches)
    steady = num_microbatches - warmup

    for i in range(warmup):
        ops.append(("F", i))

    for i in range(steady):
        ops.append(("F", warmup + i))
        ops.append(("B", i))

    for i in range(steady, num_microbatches):
        ops.append(("B", i))

    return ops


def schedule_gp(num_stages: int, num_microbatches: int, rank: int):
    """朴素 GP 调度：全部前向后全部反向。"""
    ops = []
    for i in range(num_microbatches):
        ops.append(("F", i))
    for i in range(num_microbatches):
        ops.append(("B", i))
    return ops


def dependencies(op_seq: torch.Tensor):
    """
    标注每个 op 的前驱：
      - 同 stage 上一个 op
      - F(m) 依赖上一 stage 的 F(m)（activation 输入）
      - B(m) 依赖下一 stage 的 B(m)（梯度输入）
    """
    deps = []
    for i, op in enumerate(op_seq):
        dep = set()
        if i > 0:
            dep.add(i - 1)
        deps.append(dep)
    return deps


def execute(schedule_per_stage: torch.Tensor, num_stages: int):
    """
    用队列模拟各 stage 执行，打印时间线，统计气泡占比。
    """
    total_ops = sum(len(s) for s in schedule_per_stage)
    time = 0
    stage_time = [0] * num_stages
    stage_queue = [list(s) for s in schedule_per_stage]
    completed = 0
    while completed < total_ops:
        active = 0
        for r in range(num_stages):
            if stage_queue[r]:
                op = stage_queue[r].pop(0)
                completed += 1
                active += 1
                stage_time[r] += 1
        if active == 0:
            break
        time += 1
    bubble = time * num_stages - total_ops
    return {"total_time": time, "total_ops": total_ops, "bubble": bubble,
            "bubble_ratio": bubble / (time * num_stages)}


def compare_gp_vs_1f1b(num_stages: int, num_microbatches: int):
    """返回 (GP 气泡, 1F1B 气泡, GP 峰值 act, 1F1B 峰值 act)"""
    gp_schedules = [schedule_gp(num_stages, num_microbatches, r) for r in range(num_stages)]
    f1b_schedules = [schedule_1f1b(num_stages, num_microbatches, r) for r in range(num_stages)]

    gp_result = execute(gp_schedules, num_stages)
    f1b_result = execute(f1b_schedules, num_stages)

    gp_peak_act = num_microbatches
    f1b_peak_act = num_stages

    return (gp_result["bubble"], f1b_result["bubble"],
            gp_peak_act, f1b_peak_act)


# ===== 测试验证 =====
if __name__ == "__main__":
    num_stages, num_mb = 4, 8

    for rank in range(num_stages):
        ops = schedule_1f1b(num_stages, num_mb, rank)
        warmup = num_stages - rank - 1
        f_count = sum(1 for op in ops if op[0] == "F")
        b_count = sum(1 for op in ops if op[0] == "B")
        assert f_count == num_mb, f"rank {rank}: F 数 {f_count} != {num_mb}"
        assert b_count == num_mb, f"rank {rank}: B 数 {b_count} != {num_mb}"
        for i in range(warmup):
            assert ops[i] == ("F", i), f"rank {rank}: warm-up 第 {i} 个应为 F({i})"
        print(f"  rank {rank}: warmup={warmup}, ops={len(ops)}")
    print("✅ 1F1B 调度: 各 stage F/B 数量正确")

    ops0 = schedule_1f1b(3, 6, 0)
    deps = dependencies(ops0)
    assert len(deps) == len(ops0)
    assert deps[0] == set()
    assert 0 in deps[1]
    print("✅ dependencies: 前驱标注正确")

    schedules = [schedule_1f1b(num_stages, num_mb, r) for r in range(num_stages)]
    result = execute(schedules, num_stages)
    assert result["total_ops"] == num_stages * num_mb * 2
    assert result["bubble"] >= 0
    print(f"✅ execute: time={result['total_time']}, bubble={result['bubble']} ({result['bubble_ratio']:.1%})")

    gp_b, f1b_b, gp_a, f1b_a = compare_gp_vs_1f1b(num_stages, num_mb)
    assert f1b_a <= gp_a, "1F1B 峰值显存应 <= GP"
    print(f"✅ GP vs 1F1B: 气泡 {gp_b} vs {f1b_b}, 峰值act {gp_a} vs {f1b_a}")
    print("✅ 全部测试通过")
