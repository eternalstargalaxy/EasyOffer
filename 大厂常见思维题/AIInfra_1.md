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
