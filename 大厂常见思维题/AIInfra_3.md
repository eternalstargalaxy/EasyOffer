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
