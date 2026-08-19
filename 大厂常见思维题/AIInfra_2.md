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
