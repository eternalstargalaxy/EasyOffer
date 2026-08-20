# AI Infra 面经：通信与显存

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
