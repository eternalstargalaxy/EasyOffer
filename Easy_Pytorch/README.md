# Easy_Pytorch — PyTorch 基础手撕

| 文件 | 内容 |
|------|------|
| `Softmax.ipynb` | 朴素/safe/online softmax，数值溢出与导数 |
| `CrossEntropy.ipynb` | log-sum-exp 稳定实现，梯度 $\hat p-y$ |
| `KLDivergence.ipynb` | KL 定义、非对称性、与 `F.kl_div` 约定、JS 散度 |
| `Backpropagation.ipynb` | 计算图、链式法则、两层 MLP 反向（详细解析+骨架） |
| `SGD.ipynb` | mini-batch SGD、动量、与 `torch.optim.SGD` 对拍（详细解析+骨架） |

每个文件含：原理推导 → 带注释实现 → 与 PyTorch 内置对拍验证 → 测试验证 cell → 易错点小结。
