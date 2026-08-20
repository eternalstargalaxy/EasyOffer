# AI Infra 专题面经

---

# 训练并行

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

---

# 推理优化

> 覆盖 KV Cache、PagedAttention、FlashAttention、Continuous Batching、投机解码、量化等推理高频考点。

## 一、KV Cache

### Q1: KV Cache 的作用？显存占用怎么算？
- 自回归解码时，每步生成 1 token，需对**所有历史 token** 算 attention。若不缓存，每步重算 $K/V$，复杂度 $O(n^2)$。
- **KV Cache**: 把每步算出的 $K_t, V_t$ 存起来，新步只需算新 token 的 $q, k, v$，把 $k, v$ 追加到 cache，attention 对 cache 做 $O(n)$。
- **显存**: $2 \times n_{layer} \times n_{head} \times d_{head} \times seq\_len \times batch \times 2\text{bytes(fp16)}$。
  - 7B 模型 32 层 32 头 128 维，seq=2048，batch=1: $2 \times 32 \times 32 \times 128 \times 2048 \times 2 = 1$ GB。

### Q2: prefill 和 decode 阶段的区别？
- **Prefill**: 处理 prompt，一次算出所有 prompt token 的 K/V 填入 cache。计算密集型（GEMM），算力是瓶颈。
- **Decode**: 每步生成 1 token，把新 k/v 追加到 cache，对全部历史做 attention。访存密集型（batch=1 时 GEMM 退化为 GEMV），显存带宽是瓶颈。
- 优化: prefill 用 FlashAttention/Chunked Prefill，decode 用 Continuous Batching 提升吞吐。

## 二、PagedAttention

### Q3: PagedAttention 解决什么问题？
- 传统 KV Cache 预分配 **max_seq_len** 连续显存，但实际生成长度不定 → **内部碎片**严重（预分配 2048 但只生成 200，浪费 90%）。
- PagedAttention 借鉴 OS 虚拟内存：把 KV Cache 分成固定大小的 **block**（如 16 token/block），按需分配，逻辑连续物理分散。
- 显存利用率从 ~20% 提升到 ~95%，支持更大的 batch。

### Q4: PagedAttention 的 block table 是什么？
- 每个序列有一个 **block table**：逻辑 block id → 物理 block 地址的映射。
- Attention 计算时通过 block table 查找物理地址，跨 block 做 attention。
- 类似 OS 页表：逻辑页号 → 物理页帧。

## 三、FlashAttention

### Q5: FlashAttention v1 的核心思想？为什么快？
- 标准 Attention: $S = QK^T \in \mathbb{R}^{n \times n}$，需 $O(n^2)$ 显存，对长序列爆显存。
- FlashAttention: 用 **tiling** 把 Q/K/V 分块加载到 SRAM，在 SRAM 内算局部 attention，用 **online softmax** 增量更新，从不物化完整 $n \times n$ 矩阵。
- **快的原因**: 减少 HBM 读写次数（$O(n^2)$ → $O(n^2/M)$，$M$ = SRAM 大小），是 **IO-aware** 优化，不是减少 FLOPS。

### Q6: FlashAttention v2 相比 v1 的改进？
- **减少非 matmul FLOPS**: v1 中 rescale 操作占比高，v2 重新组织计算顺序，把 rescale 推到循环外。
- **更好的并行**: v1 沿 seq 维并行（沿 K/V 切），v2 沿 Q 维也切 → 更好的 GPU 利用率。
- **work partitioning**: v2 把反向传播的 dK/dV 计算分到不同 warp，减少同步。
- 速度: v2 比 v1 快 ~2x，比标准 attention 快 5-9x（长序列）。

### Q7: FlashAttention 的 online softmax 原理？
- 标准 softmax: $\text{softmax}(x_i) = e^{x_i - m} / \sum e^{x_j - m}$，需两遍扫描（一遍求 max $m$，一遍求和）。
- **Online softmax**: 流式处理每个 block，维护 running max $m$ 和 running sum $l$：
  - 新 block 算局部 $m_{new} = \max(m_{old}, m_{block})$；
  - 更新 $l_{new} = l_{old} \cdot e^{m_{old} - m_{new}} + l_{block} \cdot e^{m_{block} - m_{new}}$；
  - 已有输出 rescale: $O_{new} = O_{old} \cdot e^{m_{old} - m_{new}} / l_{old} + O_{block} / l_{new}$。
- 一遍扫描即可，无需物化完整 $n \times n$。

## 四、Continuous Batching

### Q8: static batching 的问题？continuous batching 如何解决？
- **Static batching**: 凑齐 batch 中所有序列生成完才接新请求 → 短序列等长序列，GPU 空闲。
- **Continuous batching** (in-flight batching): 每步检查 batch 中是否有序列生成完（遇到 EOS），立即移出并**填入新请求**，batch 大小动态变化。
- 吞吐提升 2-4x，延迟不增加。代表: vLLM、TGI、TensorRT-LLM。

### Q9: continuous batching 中如何处理不同长度？
- 每步 iteration，batch 中各序列长度不同 → 用 **unpadded/variable-length** attention（FlashAttention 支持变长）。
- 不做 padding，避免计算浪费。各序列在 Q 维拼接成一维，用 cu_seqlens 记录边界。

## 五、量化

### Q10: W8A16 vs W4A16 的区别？为什么推理只量化权重？
- **W8A16**: 权重 8bit，激活 16bit。推理时 $Y = X W$，$X$ 保持 fp16 精度，$W$ 从 fp16 量化到 int8，**反量化**后与 $X$ 相乘。
- **只量化权重**的原因: 激活分布动态变化（每步不同），量化难度大；权重训练后固定，可离线量化。
- **W4A16** (AWQ/GPTQ): 权重 4bit，显存再减一半。4bit 权重需 **dequantize kernel**（如 `dequantize_int4_to_fp16`）高效反量化。

### Q11: AWQ 和 GPTQ 的区别？
- **GPTQ**: 基于**二阶信息**（Hessian）逐列量化权重，用校准数据补偿量化误差。精度高但量化慢（需逐层校准）。
- **AWQ**: 观察**激活大的通道对应权重更重要**，对重要通道保持高精度（per-channel scaling），其余激进量化。量化快，精度接近 GPTQ。
- 实践: AWQ 速度快、易实现，GPTQ 精度略高、量化慢。

## 六、投机解码 (Speculative Decoding)

### Q12: 投机解码的原理？为什么能加速？
- **动机**: 大模型 decode 每步只出 1 token，算力利用率低（访存密集）。
- **方法**: 用一个小 **draft model** 一次猜 $k$ 个 token，大模型 **批量验证** 这 $k$ 个 token（一次前向），接受正确前缀，拒绝处重新采样。
- **加速**: 大模型一次前向验证 $k$ 个 token，若接受率 $\alpha$，等效每步出 $\alpha \cdot k$ 个 token，加速 $\approx 1 + \alpha \cdot k$。
- **无损**: 接受/拒绝规则保证输出分布与原大模型完全一致。

---

# 通信与显存

> 覆盖集合通信原语、显存计算、混合精度、激活重计算、MoE 调度等。

## 一、集合通信原语

### Q1: AllReduce / AllGather / ReduceScatter 的关系？
- **ReduceScatter**: 每卡拿到**分片**后的聚合结果（输入 $N$ 份完整数据，输出 $1/N$ 份聚合数据）。
- **AllGather**: 每卡拿到**完整**的拼接结果（输入 $1/N$ 份，输出 $N$ 份完整）。
- **AllReduce = ReduceScatter + AllGather**: 先分片聚合，再广播拼回。
- 通信量: AllReduce $\approx 2M$，ReduceScatter $\approx M$，AllGather $\approx M$。

### Q2: All-to-All 在 MoE 中的作用？
- MoE 中每个 token 被路由到不同 expert，expert 分布在不同 GPU → 需把 token 发到对应 expert 所在 GPU。
- **All-to-All**: 每卡把不同部分数据发到不同目标卡（$N \times N$ 通信模式），正好匹配 MoE 的 expert dispatch。
- 通信量: 每卡发送 $\frac{N-1}{N} \cdot \text{data}$，约为一份数据量。

### Q3: Broadcast 和 Scatter 的区别？
- **Broadcast**: 一卡的数据**完整**复制到所有卡（$1 \to N$，每卡收到完整副本）。
- **Scatter**: 一卡的数据**分片**发到所有卡（$1 \to N$，每卡收到 $1/N$）。
- 通信量: Broadcast $= M$，Scatter $= M/N$。

## 二、显存计算

### Q4: 训练 7B 模型需要多少显存？（混合精度 + Adam）
- 参数 $\Phi = 7 \times 10^9$。
- **fp16 参数**: $2\Phi = 14$ GB
- **fp16 梯度**: $2\Phi = 14$ GB
- **fp32 Adam 状态** (momentum + variance + master weight): $12\Phi = 84$ GB
- **激活**: 取决于 batch/seq，batch=1 seq=2048 约 4-8 GB
- **总计**: $\approx 112 + 8 = 120$ GB → 单卡 80G A100 放不下，需 ZeRO/FSDP。

### Q5: 推理 7B 模型需要多少显存？
- **fp16 参数**: $2\Phi = 14$ GB
- **KV Cache**: $2 \times n_{layer} \times n_{head} \times d_{head} \times seq \times batch \times 2$
  - 7B: 32 层 32 头 128 维，seq=2048 batch=1: $2 \times 32 \times 32 \times 128 \times 2048 \times 2 = 1$ GB
- **激活**: 约 1-2 GB
- **总计**: $\approx 16$ GB → 单卡 24G 4090 可跑。

### Q6: int4 量化后 7B 推理显存？
- **int4 参数**: $0.5\Phi = 3.5$ GB
- **KV Cache** (fp16): 仍 1 GB（量化权重不量化 KV Cache）
- **总计**: $\approx 5$ GB → 单卡 8G 4060 可跑。

## 三、混合精度

### Q7: bf16 和 fp16 的区别？为什么训练优先 bf16？
- **fp16**: 1 位符号 + 5 位指数 + 10 位尾数。范围 $[6 \times 10^{-8}, 65504]$，**溢出风险高**（梯度 >65504 就 inf）。
- **bf16**: 1 位符号 + 8 位指数 + 7 位尾数。范围与 fp32 相同 $[1 \times 10^{-38}, 3 \times 10^{38}]$，**不易溢出**，但精度低（7 位尾数）。
- **训练优先 bf16**: 梯度范围大，fp16 易溢出需 loss scaling，bf16 无需。
- **推理可用 fp16**: 激活范围可控，fp16 精度更高。

### Q8: 混合精度训练的 master weight 是什么？
- 前向/反向用 **fp16**（快、省显存），但参数更新用 **fp32 master weight**（精度高）。
- 流程: fp32 master → 转 fp16 → 前向 → 反向得 fp16 梯度 → fp32 梯度 → Adam 更新 fp32 master。
- 额外显存: fp32 master + fp32 momentum + fp32 variance = $12\Phi$。

## 四、激活重计算

### Q9: 激活重计算 (Gradient Checkpointing) 的原理？以时间换多少显存？
- **问题**: 反向传播需要前向激活，$L$ 层激活全存 → 显存 $O(L)$。
- **重计算**: 只存 $\sqrt{L}$ 个检查点，反向时从最近检查点**重新前向**算出所需激活 → 显存 $O(\sqrt{L})$。
- **代价**: 前向算两遍（一遍正常，一遍反向时重算），时间增加约 33%。
- **选择性重计算**: 只重计算激活大的层（如 Attention），不重计算激活小的层（如 LayerNorm），时间增加更少。

### Q10: full vs selective activation checkpointing？
- **Full**: 所有层都重计算，显存 $O(\sqrt{L})$，时间 $\approx 2x$。
- **Selective**: 只重计算 attention（激活 $O(n^2)$），不重计算 MLP/LayerNorm（激活 $O(n)$），显存节省 80%，时间增加 <10%。
- 实践: Megatron-LM 默认 selective，几乎不增加时间。

## 五、MoE 调度

### Q11: MoE 的 load balancing 问题？
- MoE 中 token 被路由到 top-k expert，若某 expert 被选太多 → 该 expert 成为瓶颈，其他 expert 空闲。
- **辅助损失**: $L_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$，$f_i$ = token 到 expert $i$ 的比例，$P_i$ = router 给 expert $i$ 的平均概率。鼓励均匀分配。
- **capacity factor**: 每个 expert 设容量上限 $C = \text{cap\_factor} \times \text{avg\_tokens}$，超限 token 被丢弃或传给次优 expert。

### Q12: MoE 的 expert parallelism 和 TP 如何组合？
- **Expert Parallelism (EP)**: 不同 expert 放不同 GPU，token 通过 All-to-All 发到对应 expert 所在 GPU。
- **EP + TP**: 每个 expert 内部再做 TP（大 expert 跨多卡切分），适合 expert 较大的场景（如 DeepSeek-V2 的 shared expert）。
- 通信: EP 用 All-to-All（跨节点），TP 用 AllReduce（节点内），先 TP 后 EP。

---

# 系统设计

> 大模型 Infra 岗位常见系统设计题

## Q1: 设计一个支持 10 万 QPS 的 LLM 推理服务

### 需求分析
- 模型：70B 参数，fp16
- 每秒 10 万请求到达
- 平均 prompt 长度 512 token，平均生成长度 256 token
- 延迟 SLO：TTFT < 200ms, TPOT < 50ms

### 关键设计
1. **显存估算**
   - 70B fp16 权重: 140 GB
   - KV Cache (batch=256, seq=4096): ~40 GB
   - 总数约 180 GB -> 单卡 80G 放不下
   - 需要 TP=4 (4xA100, 280G 显存池)

2. **节点规格**
   - 每台 8xA100，单机可放 2 个 TP=4 实例
   - 10 万 QPS，假设 50% prefill/50% decode
   - Prefill latency ~100ms(batch=32) -> 320 QPS per instance
   - Decode latency ~10ms(batch=256) -> 25.6K QPS per instance
   - 需要 ~40 台 8xA100

3. **分离部署 (Disaggregated)**
   - Prefill node x 20: 计算密集，大 batch
   - Decode node x 20: 访存密集，大 batch + ContBatching
   - KV Cache 通过 RDMA 从 prefill -> decode

4. **优化手段**
   - Continuous Batching + PagedAttention (vLLM)
   - Prefix Caching (相同 system prompt)
   - Quantization: AWQ int4 -> 显存减半，可增 batch
   - Speculative Decoding: 2x decode 加速

### 面试追问
- 为什么 prefill 和 decode 要分离？
- PagedAttention 的 block size 怎么选？
- RDMA 传 KV Cache 的延迟和带宽瓶颈在哪？

## Q2: 设计 MoE 模型的训练系统

### 需求
- 1T 参数 MoE 模型，32 experts，top-2
- 1024 GPU 训练，跨节点

### 关键设计
1. **并行策略**
   - DP=64 (数据并行)
   - TP=4 (张量并行，节点内 NVLink)
   - EP=8 (Expert 并行，跨节点 IB)
   - 每卡: 1T/32 experts = 31B 激活参数

2. **通信模式**
   - Expert dispatch: All-to-All，每 token 发送到对应 expert GPU
   - Expert combine: All-to-All 反向，聚合 expert 输出
   - 通信量: 2 * B * S * D * 2 bytes per A2A

3. **负载均衡**
   - Auxiliary Loss: sum_i f_i * P_i
   - Capacity Factor: 1.25 (允许 25% 超限)
   - Drop tokens: 超限 token 丢弃，算在 loss 中

4. **专家选择**
   - 两级路由: group routing (8 groups) -> expert routing (4 experts/group)
   - 减少 top-k 计算量: 32 -> 8 groups -> 8 experts -> top-2

### 面试追问
- 为什么 MoE 用 A2A 而不是 AllReduce？
- 容量因子怎么调？太小丢 token 太多，太大专家利用率低

## Q3: 设计长上下文 (128K) 训练方案

### 需求
- 128K 上下文训练 Llama-70B
- 8x A100-80G 单节点

### 关键设计
1. **显存挑战**
   - 标准 attention 显存: O(N^2*d)
   - 128K 时单个 attention: 128K^2 * 128 * 2 * 80 layers ~ 3.3 TB

2. **必选优化**
   - FlashAttention: O(N) 显存，必须用
   - Ring Attention: 跨卡切 seq 维，通信替代显存
   - Activation Checkpointing: 重计算 attention 激活
   - 序列并行: Megatron SP，沿 seq 切分

3. **配置方案**
   - TP=8 (全节点)，FlashAttn + 激活重计算
   - micro_batch=1, seq=128K per batch
   - 预期显存: 70B*2(weight) + 128K*8192*80*2(activation with FA) ~160GB
   - 刚好 8xA100-80G = 640GB，可行

### 面试追问
- RingAttention 和 FlashAttention 如何一起用？
- 激活重计算选 full 还是 selective？

---

## 参考答案要点

### Q1 追问参考答案
- **为什么 prefill 和 decode 分离？** prefill 计算密集（大矩阵乘），decode 访存密集（小 batch 逐 token）。混合 batch 导致 GPU 利用率低，分离后各自优化。
- **PagedAttention block size**：太小→映射表大、overhead 高；太大→碎片多。通常 16-32 tokens。
- **RDMA KV Cache 瓶颈**：带宽（IB 400GB/s vs NVLink 600GB/s），延迟（~5μs/跨节点）。需 KV 压缩或量化。

### Q2 追问参考答案
- **MoE 用 A2A 而非 AllReduce**：每 token 去不同 expert，数据是 point-to-point 而非 all-to-all-reduce。A2A 天然匹配。
- **容量因子调参**：1.0-1.5 常用。监控 drop rate，>5% 时增大；expert 利用率 <70% 时减小。

### Q3 追问参考答案
- **RingAttention + FlashAttention**：RingAttention 跨卡切 seq 维，每卡用 FlashAttention 算本地 block，通信与计算 overlap。
- **激活重计算选择**：selective（只重算 attention）够用且快；full 在显存极度紧张时用。

