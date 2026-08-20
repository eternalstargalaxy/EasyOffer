# AIInfra 系统设计面经

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