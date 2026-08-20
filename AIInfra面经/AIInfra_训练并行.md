# AI Infra 面经：训练并行

> 覆盖数据并行(DP/DDP)、ZeRO 1/2/3、张量并行(TP)、流水并行(PP)、FSDP、序列并行等高频考点。

## 一、数据并行 (DDP)

### Q1: DDP 和 DP 的区别？为什么 DDP 用 Ring-AllReduce？
- **DP (DataParallel)**: 单进程多线程，受 GIL 限制，参数聚合用 PS 架构（一个 GPU 做 reduce），通信量 $O(N)$ 在 PS 上成为瓶颈。
- **DDP (DistributedDataParallel)**: 多进程，每张卡一个进程，无 GIL。用 **Ring-AllReduce** 把通信量均摊到所有卡上，每卡通信量 $O(2(N-1)/N \cdot \text{model\_size})$，无瓶颈。

### Q2: Ring-AllReduce 的通信量和时间复杂度？
- 模型大小 $M$，$N$ 张卡。
- **通信量**: 每卡发送 $\frac{2(N-1)}{N} \cdot M$，约为 $2M$（与 $N$ 无关）。
- **时间**: $2 \cdot \frac{N-1}{N} \cdot \frac{M}{\text{bandwidth}}$，带宽利用率接近 100%。
- 对比 PS 架构: PS 节点通信量 $N \cdot M$，成为瓶颈。

### Q3: DDP 中为什么需要 gradient hook？
- DDP 在反向传播时自动注册 gradient hook：当某个参数的梯度计算完成时，立即触发该参数的 AllReduce，与反向传播**重叠**，隐藏通信延迟。
- 如果等所有梯度算完再 AllReduce，通信无法与计算重叠，训练变慢。

## 二、ZeRO 优化

### Q4: ZeRO-1/2/3 分别切分什么？显存节省多少？
以 7B 模型、$N$ 张卡为例，混合精度训练显存构成：
- 参数 (fp16): $2\Phi = 14$ GB
- 梯度 (fp16): $2\Phi = 14$ GB
- 优化器状态 (fp32 momentum + fp32 variance + fp32 master): $12\Phi = 84$ GB
- 总计: $16\Phi = 112$ GB（$\Phi$ = 参数量）

| 阶段 | 切分对象 | 每卡显存 | 通信开销 |
|------|----------|----------|----------|
| ZeRO-1 | 优化器状态 | $2\Phi + 2\Phi/N + 12\Phi/N$ | 同 DDP |
| ZeRO-2 | 优化器状态 + 梯度 | $2\Phi + 4\Phi/N$ | 同 DDP（reduce-scatter 替代 all-reduce） |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | $16\Phi/N$ | 前向+反向各一次 all-gather，约 2x DDP |

### Q5: ZeRO-3 为什么通信量比 DDP 大？
- DDP: 反向后一次 AllReduce（reduce-scatter + all-gather），通信量 $2M$。
- ZeRO-3: 前向前 all-gather 参数 $M$，反向前 all-gather 参数 $M$，反向后 reduce-scatter 梯度 $M$，总计 $\approx 3M$，约为 DDP 的 1.5 倍。
- **权衡**: 用 1.5x 通信换 $N$ 倍参数显存节省。

### Q6: ZeRO-3 和 FSDP 的关系？
- FSDP (Fully Sharded Data Parallel) 是 PyTorch 原生实现的 ZeRO-3。
- 核心相同：参数按 $1/N$ 分片，前向前 all-gather 拿全参数，算完即丢弃。
- FSDP 额外特性：支持 **CPU offload**（参数/梯度卸载到 CPU）、**reshard_after_forward**（前向后立即释放）、嵌套 FSDP（不同层不同分片策略）。

## 三、张量并行 (TP)

### Q7: TP 的切分方式？列并行 vs 行并行？
- **列并行 (Column Parallel)**: $Y = XW$，把 $W$ 按列切 $W = [W_1, W_2, ..., W_N]$，每卡算 $Y_i = X W_i$，输出 $Y = [Y_1, ..., Y_N]$（沿特征维切分）。**无需通信**。
- **行并行 (Row Parallel)**: $W$ 按行切 $W = [W_1; W_2; ...; W_N]$，每卡算 $Y_i = X_i W_i$，输出 $Y = \sum Y_i$，需 **AllReduce**。
- 典型 MLP: `ColumnParallel → activation → RowParallel → AllReduce`，只需一次 AllReduce。

### Q8: TP 中为什么 MLP 用 "列并行 + 行并行" 组合？
- 列并行后输出沿特征维切分，正好作为行并行的输入（每卡已有对应的 $X_i$），无需中间通信。
- 行并行后需 AllReduce 聚合，整个 MLP 只需 **一次 AllReduce**。
- 如果两次都用列并行，输出需 AllGather 拼回，通信更多。

### Q9: TP 的通信量？为什么适合 intra-node（节点内）？
- 每层 MLP 一次 AllReduce，通信量 $2(N-1)/N \cdot d_{hidden}$，量小但**频率高**（每层都要）。
- AllReduce 对**延迟**敏感，适合 NVLink（低延迟、高带宽）的节点内，跨节点用 IB 延迟太高。
- 经验: TP 一般不超过 8（单节点 GPU 数）。

## 四、流水并行 (PP)

### Q10: 朴素 PP 的 bubble 占比？1F1B 如何改善？
- **朴素 PP**: $N$ 个 stage，$M$ 个 micro-batch。总时间 $= (N + M - 1) \cdot t$，理想时间 $= M \cdot t$，bubble 占比 $= (N-1)/(N+M-1)$。
- **1F1B**: 每个 stage 交替做 1 个 forward + 1 个 backward，pipeline 中同时有 $N$ 个 micro-batch 在飞，bubble 仍是 $(N-1)/(N+M-1)$，但**显存从 $M$ 降到 $N$**（只需缓存 $N$ 个活跃 micro-batch 的激活）。
- **Interleaved 1F1B**: 把每个 stage 的 $L$ 层拆成 $V$ 个 chunk，bubble 降到 $(N-1)/(V \cdot M + N - 1)$，代价是更多通信。

### Q11: PP 中如何处理 batch norm 和 layer norm？
- PP 切分的是**层**（不同 stage 跑不同层），同一条数据流过所有 stage。
- LayerNorm: 无跨样本统计，各 stage 独立计算，无问题。
- BatchNorm: 需跨样本统计 mean/var，PP 中不同 stage 看到不同 micro-batch，统计不一致 → **PP 通常配合 LayerNorm 使用**，避免 BN。

## 五、混合并行

### Q12: 3D 并行 (TP + PP + DP) 如何组合？
- **TP**: 节点内（NVLink），切分单层参数，通信频繁低延迟。
- **PP**: 节点间，切分层，通信少但有大 bubble，适合跨节点。
- **DP**: 剩余 GPU，切分数据，通信用 Ring-AllReduce。
- 组合: $N_{total} = N_{TP} \times N_{PP} \times N_{DP}$，先 TP 再 PP 再 DP。
- 通信: TP intra-node NVLink，PP inter-node IB，DP inter-node IB。

### Q13: 序列并行 (SP) 解决什么问题？
- 长序列（如 128K context）时，Attention 和 LN 的激活随 seq_len 线性增长，单卡放不下。
- SP 把序列维度切分到多卡：
  - LN: 沿 seq 维切分，各卡算局部统计后 AllReduce 聚合全局 mean/var。
  - Attention: $QK^T$ 需跨卡通信，用 Ring-Attention 或 Zigzag 切分。
  - MLP: 沿 seq 切分，无需通信（逐 token 独立）。
- 代表: Megatron-LM SP、DeepSpeed Ulysses、Ring-Attention。
