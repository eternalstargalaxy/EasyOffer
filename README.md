# EasyOffer

## 📝 项目介绍

**EasyOffer** 是一个面向大模型初学者与秋招/实习准备者的开源项目，收录主流大语言模型（LLM）面试中高频"手撕代码"实现与面经记录，帮助深入理解 LLM 底层原理。源于作者个人暑期实习与秋招备战笔记，持续完善中。

> ⚠️ **声明**：本项目部分代码注释参考自 DeepSeek 与 GPT，仅用于个人学习与秋招复习。

---

## 📂 目录结构

```
EasyOffer/
├── Easy_Pytorch/          # PyTorch 基础手撕（Softmax / CrossEntropy / KLDivergence / Backpropagation / SGD）
├── Easy_Attention/        # 注意力与模型组件（MHA/GQA_MLA/RoPE/RMSNorm/MLP/FFN/LoRA/SelfAttention/ActivationFunction）
├── Easy_Generator/        # 解码采样（GreedyDecoding / TopK / TopP / TemperatureSampling）
├── Easy_AIInfra/          # AI Infra 训练/推理优化（45题，分6个子目录：training/inference/quant/sparsity/ssm/advanced）
├── Easy_deepseek/         # DeepSeek 模型完整实现与解析（MoE/MLA/MTP...）
├── Easy_RL/               # 强化学习对齐（REINFORCE / PPO / GRPO / GSPO / DPO / SimPO）
├── AIInfra面经/            # AIInfra 面试面经（字节/阿里/腾讯/百度/快手/小米/旷视/蚂蚁/讯飞 + AIInfra专题）
├── 大厂常见思维题/         # 思维题（小红书/字节 + AIInfra专题）
├── hands_dirty.ipynb      # 综合手撕题合集
└── README.md
```

## ✨ 各模块说明

### 📌 Easy_Pytorch — 基础手撕
Softmax（朴素/safe/online）、交叉熵（log-sum-exp 稳定实现）、KL 散度、反向传播、SGD。每个文件末尾含测试验证 cell。

### 📌 Easy_Attention — 注意力与组件
激活函数、自注意力（基础版）、MHA(含 KV Cache，工程版)、GQA & MLA、RoPE、RMSNorm、MLP(SwiGLU)、FFN(PreNorm+GLU)、LoRA。每个文件末尾含测试验证 cell。

### 📌 Easy_Generator — 解码采样
贪心搜索、Top-K、Top-P(nucleus)、温度采样，含组合顺序说明。每个文件末尾含测试验证 cell。

### 📌 Easy_AIInfra — 训练/推理优化（45 题）
分 6 个子目录：training(13题)/inference(10题)/quantization(4题)/sparsity(2题)/ssm_attention(5题)/advanced(11题)
覆盖 DDP/ZeRO/TP/PP/FSDP/Hybrid/SP/FusedKernels，KVCache/PagedAttn/FlashAttn/vLLM/投机/Mamba/线性Attention/RingAttn/量化/稀疏/MTP/MCTS 等

### 📌 Easy_deepseek — DeepSeek 实现
模型核心结构完整实现 + DeepSeekMoE / MLA / MTP 等关键模块详细解析。

### 📌 Easy_RL — 对齐算法
REINFORCE（基础策略梯度）、PPO（RLHF 经典）、GRPO（组相对，省 critic）、GSPO（序列级重要性，Qwen3）、DPO（绕过 RM）、SimPO（去 reference），均含题目背景 + 原理 + 考察点 + 测试验证。

### 📌 面经与思维题
`AIInfra面经/` 每公司一个文件，含 AI Infra 专题面经；`大厂常见思维题/` 收录思维题，含 AI Infra 方向计算题。

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
