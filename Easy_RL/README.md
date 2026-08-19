# Easy_RL — 强化学习对齐算法手撕

> 覆盖从基础策略梯度到最新对齐算法的完整演进链，每个文件头部含**题目背景 + 原理 + 考察点**，末尾含**测试验证 cell**。

## 演进路线

```
REINFORCE → A2C(加 critic) → PPO(加 clip+多epoch) → GRPO(组内相对,省 critic) → GSPO(序列级重要性)
                                                                      ↘
DPO(偏好对闭式解,绕过 RM) → SimPO(去 reference,长度归一化)
```

## 文件索引

| 文件 | 算法 | 核心思想 | 来源 |
|------|------|----------|------|
| `REINFORCE.ipynb` | REINFORCE | 基础策略梯度，无 critic/clip，严格 on-policy | Williams 1992 |
| `PPO.ipynb` | PPO | actor-critic、clip 目标、GAE、KL 罚，RLHF 经典 | Schulman 2017 |
| `GRPO.ipynb` | GRPO | 省 critic，组内相对优势 | DeepSeek-Math/R1 2024 |
| `GSPO.ipynb` | GSPO | 序列级重要性采样，缓解 token 级 clip 长度偏置 | Qwen3 2025 |
| `DPO.ipynb` | DPO | 绕过 RM，偏好对直接训练 | Rafailov 2023 |
| `SimPO.ipynb` | SimPO | 去掉 reference，长度归一化隐式奖励 | Meng 2024 |

## 算法对比

| 维度 | REINFORCE | PPO | GRPO | GSPO | DPO | SimPO |
|------|-----------|-----|------|------|-----|-------|
| 范式 | online RL | online RL | online RL | online RL | offline SL | offline SL |
| 模型数 | 1 | 4 | 2 | 2 | 2 | 1 |
| critic | 无 | 有 | 无 | 无 | 无 | 无 |
| reference | 无 | 有 | 有 | 有 | 有 | 无 |
| clip | 无 | token 级 | token 级 | 序列级 | 无 | 无 |
| 组采样 | 否 | 否 | 是 | 是 | 否 | 否 |
| 代表 | Williams 1992 | InstructGPT | DeepSeek-R1 | Qwen3 | Rafailov 2023 | Meng 2024 |
