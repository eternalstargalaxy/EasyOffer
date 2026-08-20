"""
【题目】流水并行 1F1B 调度

【背景】
模型按层切到多卡（pipeline stage），micro-batch 在 stage 间流动。
朴素 GP（先全部前向再全部反向）气泡大、显存高；1F1B 交错前向/反向，每 stage 先做 warm-up
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
    # TODO
    raise NotImplementedError


def dependencies(op_seq):
    """
    标注每个 op 的前驱：
      - 同 stage 上一个 op
      - F(m) 依赖上一 stage 的 F(m)（activation 输入）
      - B(m) 依赖下一 stage 的 B(m)（梯度输入）
    """
    raise NotImplementedError


def execute(schedule_per_stage, num_stages):
    """
    用队列模拟各 stage 执行：
      F: 算 activation，发往下一 stage（最后一 stage 启动 B）
      B: 算梯度，发往上一 stage（第一 stage 完成 micro）
    打印时间线，统计气泡占比。
    """
    raise NotImplementedError


def compare_gp_vs_1f1b(num_stages: int, num_microbatches: int):
    """返回 (GP 气泡, 1F1B 气泡, GP 峰值 act, 1F1B 峰值 act)"""
    raise NotImplementedError

# ===== 测试验证 =====
if __name__ == "__main__":
    print("06_pipeline_parallelism_1f1b.py 测试代码：")
    try:
        # TODO: 用户实现后可在此调用核心函数验证输出形状与性质
        pass
        print("✅ 待实现核心函数后运行验证")
    except NotImplementedError:
        print("ℹ 核心函数待实现，可先阅读文件头部背景理解原理")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
