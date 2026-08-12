# Easy_AIInfra — AI Infra 训练/推理优化手撕合集

本目录面向 **AI Infra / 大模型系统工程** 岗位面试，收录训练、推理、部署方向高频"手撕"题。
每个 `.py` 文件顶部 docstring 给出题目描述、背景、要求与考察点，**需自行补全实现**。

## 目录划分

### 一、训练优化（01–08）
- 01_gradient_accumulation.py        梯度累积
- 02_mixed_precision_amp.py          混合精度 AMP + loss scaling
- 03_ddp_ring_allreduce.py           DDP 梯度同步 + Ring-AllReduce
- 04_zero_optimizer_sharding.py      ZeRO-1/2/3 状态/梯度/参数分片
- 05_tensor_parallelism.py           张量并行（列并行/行并行 + 通信）
- 06_pipeline_parallelism_1f1b.py    流水并行 1F1B 气泡调度
- 07_activation_checkpointing.py     激活重计算（以空间换显存）
- 08_fsdp_shard.py                   FSDP 参数分片 all-gather/reduce-scatter

### 二、推理优化（09–17）
- 09_kv_cache.py                     KV Cache 增量推理
- 10_paged_attention.py              PagedAttention 块状 KV 管理
- 11_flash_attention.py              FlashAttention tiling + online softmax
- 12_continuous_batching.py          Continuous Batching 动态批调度
- 13_speculative_decoding.py         投机采样 draft + verify
- 14_quantization_w8a16.py           W8A16 线性层量化
- 15_quantization_awq_gptq.py        AWQ / GPTQ 量化
- 16_prefix_caching.py               Prefix / RadixAttention 前缀复用
- 17_chunked_prefill.py              Chunked Prefill 与 decode 混排

### 三、进阶部署（18–20）
- 18_moe_all2all_dispatch.py         MoE all-to-all dispatch/combine
- 19_lora_multi_adapter_serve.py     多 LoRA adapter 推理调度
- 20_distserve_prefill_decode.py     Prefill/Decode 分离部署（Disaggregated Serving）

## 使用建议
- 先读 docstring，限时 30–45 分钟手写实现，再对照工业实现（Megatron / vLLM / DeepSpeed）复盘。
- 训练题可在单机多进程下用 `torch.distributed` 跑通；推理题优先保证逻辑正确，再考虑 kernel 化。