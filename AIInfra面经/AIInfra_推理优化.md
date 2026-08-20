# AI Infra 面经：推理优化

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
