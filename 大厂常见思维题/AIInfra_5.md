## 题目：MoE 专家负载计算

训练一个 MoE 模型：16 experts, top-2 routing, batch=4, seq_len=2048, token 总数 = 8192。
每个 expert 处理 token 数期望是多少？如果 capacity_factor=1.25，每个 expert 最多处理多少 token？
超限 token 怎么办？

## 解析

### 1. 期望负载
总 token: B * S = 4 * 2048 = 8192
总调度次数: token * topk = 8192 * 2 = 16384
每 expert 平均: 16384 / 16 = 1024 个 token

### 2. Capacity Factor
容量 = avg_tokens * capacity_factor = 1024 * 1.25 = 1280
即每 expert 最多处理 1280 个 token

### 3. 超限处理
- 超限 token 被 dropout（丢弃），不参与该 expert 计算
- 在 auxiliary loss 中会对 overload expert 增加惩罚
- 极端情况：capacity_factor=1.0 会丢弃 10-20% 的 token

### 4. 关键结论
- topk 越大，总调度越多，负载越均匀
- 容量因子越大，丢弃越少，但 expert 利用率越低
- 实践：cap_factor=1.25 在多数场景平衡良好