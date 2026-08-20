# EasyOffer

## 📝 项目介绍

**EasyOffer** 是一个面向大模型初学者与秋招/实习准备者的开源项目，收录主流大语言模型（LLM）面试中高频"手撕代码"实现与面经记录，帮助深入理解 LLM 底层原理。源于作者个人暑期实习与秋招备战笔记，持续完善中。

> ⚠️ **声明**：本项目部分代码注释参考自 DeepSeek 与 GPT，仅用于个人学习与秋招复习。

---

## 📂 目录结构

```
EasyOffer/
├── Easy_Pytorch/          # PyTorch 基础手撕（10题：Softmax/CE/KL/Backprop/SGD/AdamW/LayerNorm/Dropout/LabelSmoothing/LRScheduler）
├── Easy_Attention/        # 注意力与组件（14题：MHA/GQA_MLA/RoPE/RMSNorm/MLP/FFN/LoRA/SelfAttn/Activation/FlashAttn/SparseAttn/ALiBi/AttnSink/QKNorm）
├── Easy_Generator/        # 解码采样（8题：Greedy/TopK/TopP/Temperature/BeamSearch/ContrastiveSearch/RepetitionPenalty/LogitsProcessor）
├── Easy_Tokenizer/        # 分词算法（4题：BPE/WordPiece/Unigram/SentencePiece）
├── Easy_AIInfra/          # AI Infra 训练/推理优化（48题，分6个子目录：training/inference/quant/sparsity/ssm/advanced）
├── Easy_deepseek/         # DeepSeek 模型完整实现与解析（MoE/MLA/MTP...）
├── Easy_RL/               # 强化学习对齐（12题：REINFORCE/A2C/PPO/GRPO/GSPO/DPO/SimPO/KTO/ORPO/RewardModel/RLHF_Pipeline/IterativeDPO）
├── AIInfra面经/            # AIInfra 面试面经（字节/阿里/腾讯/百度/快手/小米/旷视/蚂蚁/讯飞/美团/京东 + AIInfra专题）
├── 大厂常见思维题/         # 思维题（小红书/字节 + AIInfra专题）
├── hands_dirty.ipynb      # 综合手撕题合集
└── README.md
```

## ✨ 各模块说明

### 📌 Easy_Pytorch — 基础手撕（10 题）
Softmax（朴素/safe/online）、交叉熵（log-sum-exp 稳定实现）、KL 散度、反向传播、SGD、Adam/AdamW、LayerNorm、Dropout、LabelSmoothing、WarmupCosineScheduler。每个文件末尾含测试验证 cell。

### 📌 Easy_Attention — 注意力与组件（14 题）
激活函数、自注意力（基础版）、MHA(含 KV Cache，工程版)、GQA & MLA、RoPE、RMSNorm、MLP(SwiGLU)、FFN(PreNorm+GLU)、LoRA、FlashAttention(IO-aware tiling)、SparseAttention(滑动窗口)、ALiBi(线性偏置)、AttentionSink(StreamingLLM)、QKNorm。每个文件末尾含测试验证 cell。

### 📌 Easy_Generator — 解码采样（8 题）
贪心搜索、Top-K、Top-P(nucleus)、温度采样、束搜索(BeamSearch)、对比搜索(ContrastiveSearch)、重复惩罚(RepetitionPenalty)、LogitsProcessor 框架，含组合顺序说明。每个文件末尾含测试验证 cell。

### 📌 Easy_Tokenizer — 分词算法（4 题）
BPE(GPT-2)、WordPiece(BERT)、Unigram(T5)、SentencePiece(LLaMA)，含 Viterbi 分词与 ▁ 空格标记。每个文件末尾含测试验证 cell。

### 📌 Easy_AIInfra — 训练/推理优化（48 题）
分 6 个子目录：training(14题)/inference(12题)/quantization(4题)/sparsity(2题)/ssm_attention(5题)/advanced(11题)
覆盖 DDP/ZeRO/TP/PP/FSDP/Hybrid/SP/FusedKernels，KVCache/PagedAttn/FlashAttn/vLLM/投机/Mamba/线性Attention/RingAttn/量化/稀疏/MTP/MCTS 等

### 📌 Easy_deepseek — DeepSeek 实现
模型核心结构完整实现 + DeepSeekMoE / MLA / MTP 等关键模块详细解析。

### 📌 Easy_RL — 对齐算法（12 题）
REINFORCE（基础策略梯度）、A2C（Actor-Critic）、PPO（RLHF 经典）、GRPO（组相对，省 critic）、GSPO（序列级重要性，Qwen3）、DPO（绕过 RM）、SimPO（去 reference）、KTO（前景理论，无需成对）、ORPO（无需 reference）、RewardModel(Bradley-Terry)、RLHF_Pipeline(SFT→RM→PPO)、IterativeDPO(多轮自改进)，均含题目背景 + 原理 + 考察点 + 测试验证。

### 📌 面经与思维题
`AIInfra面经/` 每公司一个文件（字节/阿里/腾讯/百度/快手/小米/旷视/蚂蚁/讯飞/美团/京东），含 AI Infra 专题面经与系统设计题；`大厂常见思维题/` 收录思维题，含 AI Infra 方向计算题。

---

## 🚧 未来计划
- LLaMA / Qwen 等更多模型完整手写
- 更多 Infra 优化（CUDA kernel、序列并行）
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
