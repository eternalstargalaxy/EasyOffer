# Easy_Attention — 注意力与模型组件手撕

| 文件 | 内容 |
|------|------|
| `ActivationFunction.ipynb` | ReLU/Sigmoid/Tanh/SiLU/GELU，导数与对比 |
| `SelfAttention.ipynb` | 基础自注意力：scaled dot-product、多头、因果 mask、为什么除 √d_k |
| `MHA.ipynb` | 工程版 MHA：LLaMA 风格 + KV Cache（prefill/decode 验证） |
| `GQA_MLA.ipynb` | GQA 共享 KV 头；MLA 低秩压缩 + latent cache |
| `RoPE.ipynb` | 旋转位置编码、rotate_half、相对位置性质验证 |
| `RMSNorm.ipynb` | RMSNorm vs LayerNorm、rsqrt 实现 |
| `MLP.ipynb` | LLaMA SwiGLU MLP、2/3 缩放 |
| `FFN.ipynb` | PreNorm + GLU 家族 |
| `LoRA.ipynb` | LoRA 低秩适配、B=0 初始化、merge |
| `FlashAttentionKernel.ipynb` | IO-aware tiling + online softmax，FlashAttention 核心算法 |
| `SparseAttention.ipynb` | Sliding Window + Global Token 稀疏注意力 |
| `test.py` | 关键性质 smoke test |

每个 ipynb 末尾含测试验证 cell。`SelfAttention` 为基础教学版，`MHA` 为工程实现版（含 KV Cache），`FlashAttentionKernel` 为 IO 优化版。
