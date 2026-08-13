# Easy_Generator — 解码采样方法手撕合集

自回归生成时如何从 logits 选下一个 token。各方法对比：

| 方法 | 公式/操作 | 确定性 | 多样性 | 典型用途 |
|------|-----------|--------|--------|----------|
| 贪心搜索 | argmax | ✅ | ❌ | 评测/确定性输出 |
| 温度采样 | softmax(logits/T) | ❌ | T 大则多样 | 调节随机性 |
| Top-K | 保留最大 K 个再采样 | ❌ | 中 | 通用 |
| Top-P | 保留累积概率≥p 的最小集 | ❌ | 自适应 | 开放域生成 |

## 文件
- `贪心搜索.ipynb`：每步 argmax
- `TopK.ipynb`：Top-K 采样
- `TopP.ipynb`：Top-P / nucleus 采样
- `Temperature(温度采样)/温度采样.ipynb`：温度采样

## 组合顺序（工程默认）
`logits → /temperature → TopK → TopP → multinomial`
