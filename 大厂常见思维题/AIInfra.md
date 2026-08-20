# AI Infra 思维计算题

---

## 题目1

## 💭 题目

训练一个 13B 参数的模型，使用 Adam 优化器，混合精度（fp16 计算 + fp32 master），batch_size=1，seq_len=2048。
**问题：至少需要多少显存？给出各项分解。**

## ✏️ 解析

### 1. 模型相关显存（与 batch/seq 无关）

| 项目 | 公式 | 大小 |
|------|------|------|
| fp16 参数 | $2\Phi$ | 26 GB |
| fp16 梯度 | $2\Phi$ | 26 GB |
| fp32 master weight | $4\Phi$ | 52 GB |
| fp32 momentum | $4\Phi$ | 52 GB |
| fp32 variance | $4\Phi$ | 52 GB |
| **小计** | $16\Phi$ | **208 GB** |

### 2. 激活显存（与 batch/seq 相关）

13B 模型约 40 层，hidden_size=5120，注意力头数=40。
- 每层激活约 $\text{batch} \times \text{seq} \times \text{hidden} \times (34 + 5 \times \text{seq} / \text{hidden})$ bytes
- $\approx 1 \times 2048 \times 5120 \times 34 \times 2 \text{bytes} \approx 0.7$ GB/层
- 40 层 $\approx 28$ GB

### 3. 总计

$$\text{总显存} \approx 208 + 28 = 236 \text{ GB}$$

### 4. 结论
- 单卡 A100-80G 放不下（需 236 GB）。
- **ZeRO-3 / FSDP** 4 卡: 模型部分 $208/4 = 52$ GB + 激活 28 GB = 80 GB → 刚好 1 张 A100-80G。
- **ZeRO-3 + 激活重计算** 4 卡: $52 + 28/\sqrt{40} \approx 52 + 4.4 = 56.4$ GB → 有余量。

### 关键公式记忆
- 训练显存 $\approx 16\Phi + \text{激活}$（Adam + 混合精度）
- 推理显存 $\approx 2\Phi + \text{KV Cache}$（fp16）
- int4 推理 $\approx 0.5\Phi + \text{KV Cache}$


---

## 题目2

## 💭 题目

4 张 GPU 做 DDP 训练，模型大小 1 GB（fp32），网络带宽 25 GB/s。
**问题：Ring-AllReduce 一次需要多少时间？对比 PS 架构。**

## ✏️ 解析

### 1. Ring-AllReduce

Ring-AllReduce = ReduceScatter + AllGather，每卡通信量 $\frac{2(N-1)}{N} \cdot M$。

- $N = 4$, $M = 1$ GB
- 每卡通信量: $\frac{2 \times 3}{4} \times 1 = 1.5$ GB
- 时间: $\frac{1.5}{25} = 0.06$ 秒 = **60 ms**

### 2. PS (Parameter Server) 架构

PS 聚合所有梯度: PS 节点接收 $N-1$ 份梯度，广播 1 份结果。

- PS 接收: $(N-1) \times M = 3$ GB
- PS 广播: $N \times M = 4$ GB（发给所有 worker）
- PS 总通信: $7$ GB，时间 $\frac{7}{25} = 280$ ms
- **瓶颈**: PS 节点带宽打满，其他节点空闲。

### 3. 对比

| 维度 | Ring-AllReduce | PS |
|------|---------------|-----|
| 每卡通信量 | 1.5 GB | 不均（PS 7GB，worker 1GB） |
| 时间 | 60 ms | 280 ms |
| 瓶颈 | 无（均匀） | PS 节点 |
| 扩展性 | 好（通信量与 N 无关） | 差（PS 随 N 增长） |

### 关键结论
- Ring-AllReduce 通信量 $\approx 2M$，与 $N$ 无关 → 可扩展。
- PS 架构通信量 $\approx NM$，随 $N$ 线性增长 → 不可扩展。
- 这就是为什么 DDP 用 Ring-AllReduce 而非 PS。


---

## 题目3

## 💭 题目

用 7B 模型（32 层，32 头，head_dim=128）做推理，batch=4，max_seq_len=4096，fp16。
**问题：KV Cache 占多少显存？如果用 PagedAttention（block_size=16），最多需要多少个 block？**

## ✏️ 解析

### 1. KV Cache 显存

$$\text{KV Cache} = 2 \times n_{layer} \times n_{head} \times d_{head} \times seq \_len \times batch \times \text{dtype\_size}$$

- 2: K 和 V 两个缓存
- $n_{layer} = 32$, $n_{head} = 32$, $d_{head} = 128$
- $seq\_len = 4096$, $batch = 4$
- dtype_size = 2 bytes (fp16)

$$= 2 \times 32 \times 32 \times 128 \times 4096 \times 4 \times 2 = 8 \text{ GB}$$

### 2. PagedAttention 的 block 数

- block_size = 16 token/block
- 每序列最多 $4096 / 16 = 256$ 个 block
- batch=4，最多 $4 \times 256 = 1024$ 个 block
- 每 block 显存: $2 \times 32 \times 32 \times 128 \times 16 \times 2 = 0.0078$ GB = 8 MB
- 总 block 显存: $1024 \times 8 \text{MB} = 8$ GB（与连续分配相同）

### 3. PagedAttention 的优势

| 维度 | 连续分配 | PagedAttention |
|------|----------|----------------|
| 预分配 | 4 × 4096 × per_token = 8 GB | 按需分配 |
| 实际用（avg_seq=512） | 仍 8 GB（预分配 max） | 4 × 512 × per_token = 1 GB |
| 碎片 | 严重（87.5% 浪费） | 无（按需分配 block） |
| 可超 max_seq_len | 否（预分配固定） | 是（动态分配新 block） |

### 关键结论
- KV Cache 显存 $\propto n_{layer} \times n_{head} \times d_{head} \times seq \times batch$。
- PagedAttention 消除内部碎片，实际显存 $\propto$ **实际生成长度**而非 max_seq_len。
- block_size 越小碎片越少但 block table 越大，典型值 16。


---

## 题目4

## 💭 题目

8 卡 A100-80G（单节点 NVLink），训 70B 模型。
**问题：如何设计并行策略？给出 TP/PP/DP 的选择和理由。**

## ✏️ 解析

### 1. 显存估算

- 70B 训练显存 $\approx 16\Phi = 16 \times 70 = 1120$ GB（Adam + 混合精度）
- 8 卡总显存 $= 8 \times 80 = 640$ GB → **放不下**（1120 > 640）

### 2. 必须用 ZeRO-3/FSDP 切分参数

- ZeRO-3 后模型部分: $1120 / 8 = 140$ GB/卡 → 仍超 80G（还有激活）
- 需额外用 **激活重计算** 把激活从 ~200G 降到 ~40G
- $140 + 40 = 180$ GB → 仍超 → **需 TP 进一步切分单层参数**

### 3. 方案: TP=4 + DP=2 + ZeRO-1

| 并行 | 配置 | 理由 |
|------|------|------|
| TP=4 | 节点内 4 卡 | 70B 单层参数大，TP 切分单层，NVLink 低延迟 |
| DP=2 | 2 组数据并行 | 剩余 4 卡做 DP，提升吞吐 |
| ZeRO-1 | DP 组内切优化器状态 | 优化器状态占 12Φ=840G 最大，ZeRO-1 切到 2 卡 |

显存核算（每卡）:
- TP 切分后单层参数: $2\Phi/4 = 35$ GB（fp16 参数+梯度）
- ZeRO-1 切优化器: $12\Phi / (4 \times 2) = 105$ GB → 仍太大

### 4. 更优方案: TP=8 + ZeRO-1 + 激活重计算

- TP=8: 单层参数 $2\Phi/8 = 17.5$ GB
- ZeRO-1: 优化器 $12\Phi/8 = 105$ GB → 仍超

### 5. 最终方案: FSDP=8 + 激活重计算 + CPU offload

- FSDP=8: 参数+梯度+优化器全切 $16\Phi/8 = 140$ GB
- 激活重计算: $\approx 40$ GB
- 140 + 40 = 180 GB → 仍超 80G
- **CPU offload 优化器状态**: 优化器 $12\Phi/8 = 105$ GB offload 到 CPU
- 显存: $(16\Phi - 12\Phi)/8 + 40 = 4\Phi/8 + 40 = 35 + 40 = 75$ GB → **可行！**

### 6. 结论

| 配置 | 显存/卡 | 可行 |
|------|---------|------|
| 纯 DDP | 1120+200=1320 | ❌ |
| ZeRO-3=8 | 140+200=340 | ❌ |
| ZeRO-3=8 + 重计算 | 140+40=180 | ❌ |
| FSDP=8 + 重计算 + CPU offload | 35+40=75 | ✅ |

### 关键思路
1. 先算总显存，判断是否需要切分。
2. 优先 FSDP/ZeRO-3（通信开销小，通用性好）。
3. 显存仍超 → 加激活重计算（时间换显存）。
4. 仍超 → CPU offload（通信换显存）。
5. TP 只在单层参数放不下时用（通信频繁，适合 NVLink）。


---

## 题目5

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
