# Easy_Attention — 注意力与模型组件手撕

| 文件 | 内容 |
|------|------|
| `激活函数.ipynb` | ReLU/Sigmoid/Tanh/SiLU/GELU，导数与对比 |
| `自注意力机制.ipynb` | Scaled dot-product、多头、因果 mask、为什么除 √d_k |
| `MHA.ipynb` | LLaMA MHA + KV Cache（prefill/decode 验证） |
| `GQA&MLA.ipynb` | GQA 共享 KV 头；MLA 低秩压缩 + latent cache |
| `ROPE.ipynb` | 旋转位置编码、rotate_half、相对位置性质验证 |
| `RMSNorm.ipynb` | RMSNorm vs LayerNorm、rsqrt 实现 |
| `MLP.ipynb` | LLaMA SwiGLU MLP、2/3 缩放 |
| `FFN.ipynb` | PreNorm + GLU 家族 |
| `Lora.ipynb` | LoRA 低秩适配、B=0 初始化、merge |
| `test.py` | 关键性质 smoke test |

已修复原版 bug：`ROPE.ipynb`/`Lora.ipynb` 缺 import、`GQA&MLA.ipynb` MLA rope 函数作用域错误。