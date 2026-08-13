# EasyOffer

## 📝 项目介绍

**EasyOffer** 是一个面向大模型初学者与秋招/实习准备者的开源项目，收录主流大语言模型（LLM）面试中高频"手撕代码"实现与面经记录，帮助深入理解 LLM 底层原理。源于作者个人暑期实习与秋招备战笔记，持续完善中。

> ⚠️ **声明**：本项目部分代码注释参考自 DeepSeek 与 GPT，仅用于个人学习与秋招复习。

---

## 📂 目录结构

```
EasyOffer/
├── Easy_Pytorch/          # PyTorch 基础手撕（softmax / 交叉熵 / KL / 反向传播 / SGD）
├── Easy_Attention/        # 注意力与模型组件（MHA/GQA/MLA/RoPE/RMSNorm/MLP/LoRA/激活函数...）
├── Easy_Generator/        # 解码采样（贪心 / TopK / TopP / 温度）
├── Easy_AIInfra/          # AI Infra 训练/推理优化手撕（DDP/ZeRO/TP/PP/FSDP/KVCache/FlashAttn...）
├── Easy_deepseek/         # DeepSeek 模型完整实现与解析（MoE/MLA/MTP...）
├── Easy_RL/               # 强化学习对齐（DPO / PPO / GRPO）
├── LLM大厂面经合集/        # 各厂 LLM 岗面经（字节/阿里/腾讯/百度/快手/小米/旷视/蚂蚁/讯飞）
├── 大厂常见思维题/         # 思维题（小红书/字节）
├── make your hands dirty.ipynb
└── README.md
```

## ✨ 各模块说明

### 📌 Easy_Pytorch — 基础手撕
softmax（朴素/safe/online）、交叉熵（log-sum-exp 稳定实现）、KL 散度、反向传播、SGD。

### 📌 Easy_Attention — 注意力与组件
激活函数、自注意力、MHA(含 KV Cache)、GQA & MLA、RoPE、RMSNorm、MLP(SwiGLU)、FFN(PreNorm+GLU)、LoRA。

### 📌 Easy_Generator — 解码采样
贪心搜索、Top-K、Top-P(nucleus)、温度采样，含组合顺序说明。

### 📌 Easy_AIInfra — 训练/推理优化（20 题）
训练：梯度累积 / AMP / DDP+RingAllReduce / ZeRO / 张量并行 / 1F1B 流水并行 / 激活重计算 / FSDP
推理：KV Cache / PagedAttention / FlashAttention / Continuous Batching / 投机采样 / W8A16 / AWQ·GPTQ / Prefix 缓存 / Chunked Prefill
进阶：MoE all-to-all / 多 LoRA 调度 / DistServe

### 📌 Easy_deepseek — DeepSeek 实现
模型核心结构完整实现 + DeepSeekMoE / MLA / MTP 等关键模块详细解析。

### 📌 Easy_RL — 对齐算法
DPO（直接偏好优化）、PPO（RLHF 经典）、GRPO（组相对策略优化），均含原理概述。

### 📌 面经与思维题
`LLM大厂面经合集/` 按公司分目录；`大厂常见思维题/` 收录思维题。

---

## 🚧 未来计划
- LLaMA / Qwen 等更多模型完整手写
- 更多 Infra 优化（CUDA kernel、序列并行、Expert Parallel）
- 持续补充面经与手撕题

## 🤝 如何贡献
- 提 Issue 反馈问题或建议
- 提 PR 贡献代码或改进文档
- 分享项目让更多人受益

## 📜 开源许可
[MIT License](LICENSE)

---

<div align="center">
  <h3>⭐ 如果本项目对你有帮助，欢迎 Star 支持！⭐</h3>
</div>
