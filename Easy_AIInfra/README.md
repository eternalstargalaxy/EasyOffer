# Easy_AIInfra — AI Infra 训练/推理优化手撕合集（45 题）

面向 **AI Infra / 大模型系统工程** 岗位面试，按主题分为 6 个子目录。
每个 `.py` 文件顶部含【题目】【背景】【输入/输出】【考察点】，末尾含【测试验证】入口。

## 目录结构

```
Easy_AIInfra/
├── training/       # 训练优化（13 题）
├── inference/      # 推理优化（10 题）
├── quantization/   # 量化方法（4 题）
├── sparsity/       # 稀疏化方法（2 题）
├── ssm_attention/  # SSM + 线性注意力（5 题）
├── advanced/       # 进阶优化（11 题）
└── README.md
```

## 使用建议
- 先读 docstring，限时 30-45 分钟手写实现
- 训练题可在单机多进程下用 `torch.distributed` 跑通
- 推理题优先保证逻辑正确，再考虑 kernel 化
- 对照工业实现（Megatron / vLLM / DeepSpeed / Mamba）复盘