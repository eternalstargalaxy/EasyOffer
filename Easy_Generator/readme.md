# Easy_Generator — 解码采样方法手撕合集

自回归生成时如何从 logits 选下一个 token。各方法对比：

| 方法 | 文件 | 确定性 | 多样性 | 典型用途 |
|------|------|--------|--------|----------|
| 贪心搜索 | `GreedyDecoding.ipynb` | ✅ | ❌ | 评测/确定性输出 |
| 温度采样 | `TemperatureSampling.ipynb` | ❌ | T 大则多样 | 调节随机性 |
| Top-K | `TopK.ipynb` | ❌ | 中 | 通用 |
| Top-P | `TopP.ipynb` | ❌ | 自适应 | 开放域生成 |

每个文件末尾含测试验证 cell。

## 组合顺序（工程默认）
`logits → /temperature → TopK → TopP → multinomial`
