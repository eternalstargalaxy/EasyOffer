# Easy_Pytorch — PyTorch 基础手撕

| 文件 | 内容 |
|------|------|
| `softmax.ipynb` | 朴素/safe/online softmax，数值溢出与导数 |
| `交叉熵损失函数.ipynb` | log-sum-exp 稳定实现，梯度 $\\hat p-y$ |
| `KL散度.ipynb` | KL 定义、非对称性、与 `F.kl_div` 约定、JS 散度 |
| `手撕反向传播.ipynb` | 计算图、链式法则、两层 MLP 反向（详细解析+骨架） |
| `随机梯度下降.ipynb` | mini-batch SGD、动量、与 `torch.optim.SGD` 对拍（详细解析+骨架） |

每个文件含：原理推导 → 带注释实现 → 与 PyTorch 内置对拍验证 → 易错点小结。