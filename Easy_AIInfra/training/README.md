# Training — 训练优化手撕

| 文件 | 题目 |
|------|------|
| `gradient_accumulation.py` | 梯度累积 |
| `mixed_precision_amp.py` | 混合精度 AMP + loss scaling |
| `ddp_ring_allreduce.py` | DDP + Ring-AllReduce |
| `zero_optimizer_sharding.py` | ZeRO-1/2/3 状态分片 |
| `tensor_parallelism.py` | 张量并行 (Column/Row Parallel) |
| `pipeline_parallelism_1f1b.py` | 1F1B 流水调度 |
| `activation_checkpointing.py` | 激活重计算 |
| `fsdp_shard.py` | FSDP 参数分片 |
| `megatron_sp.py` | Megatron 序列并行 (SP+TP) |
| `fsdp_hybrid.py` | FSDP Hybrid Shard + CPU Offload |
| `zero_plusplus.py` | ZeRO++ hpZ + qgZ 通信优化 |
| `megatron_interleaved.py` | Interleaved 1F1B 调度 |
| `fused_kernels.py` | Fused Adam + Fused LayerNorm |
| `expert_parallel.py` | Expert Parallel + All-to-All Dispatch + 负载均衡 |

每个文件含【题目】【背景】【输入/输出】【考察点】【测试验证】。
