# Inference — 推理优化手撕

| 文件 | 题目 |
|------|------|
| `kv_cache.py` | KV Cache 原理与显存计算 |
| `paged_attention.py` | PagedAttention (block table, 虚拟内存) |
| `flash_attention.py` | FlashAttention (tiling, online softmax) |
| `continuous_batching.py` | Continuous Batching 调度 |
| `speculative_decoding.py` | 投机解码 (draft+verify) |
| `eagle.py` | Eagle 特征级投机解码 |
| `medusa.py` | Medusa 多头并行解码 |
| `self_speculative.py` | 自投机解码 (early exit) |
| `vllm_scheduler.py` | vLLM prefill/decode 分离调度 |
| `ring_attention.py` | RingAttention 分布式长序列 |
| `cuda_graph.py` | CUDA Graph 捕获与重放，省 kernel launch |
| `triton_flash_attention.py` | Triton FlashAttention 分块实现 |

每个文件含【题目】【背景】【输入/输出】【考察点】【测试验证】。
