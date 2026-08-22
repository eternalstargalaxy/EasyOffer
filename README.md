# EasyOffer

## 📝 项目介绍

**EasyOffer** 是一个面向大模型初学者与秋招/实习准备者的开源项目，收录主流大语言模型（LLM）面试中高频"手撕代码"实现与面经记录，帮助深入理解 LLM 底层原理。源于作者个人暑期实习与秋招备战笔记，持续完善中。

> ⚠️ **声明**：本项目部分代码注释参考自 DeepSeek 与 GPT，仅用于个人学习与秋招复习。

---

## 📂 目录结构

```
EasyOffer/
├── Easy_Pytorch/          # PyTorch 基础手撕（4 .py + 4 .ipynb：生产级损失函数/归一化/优化器/基础算法 + Backprop/Dropout/LabelSmoothing/WarmupScheduler）
├── Easy_Attention/        # 注意力与组件（1 .py + 10 .ipynb + test.py：core_attention.py(SDPA/MHA/RoPE/SwiGLU) + 专题 ipynb）
├── Easy_Generator/        # 解码采样（8 .ipynb：Greedy/TopK/TopP/Temperature/BeamSearch/ContrastiveSearch/RepetitionPenalty/LogitsProcessor）
├── Easy_Tokenizer/        # 分词算法（4 .ipynb：BPE/WordPiece/Unigram/SentencePiece）
├── Easy_AIInfra/          # AI Infra 训练/推理优化（48 .py，含 type hints + return hints，分6个子目录）
│   ├── training/          # 14题：DDP/ZeRO/TP/PP/FSDP/Hybrid/SP/FusedKernels/AMP/梯度累积/ExpertParallel/激活重计算
│   ├── inference/         # 12题：KV Cache/PagedAttn/FlashAttn/vLLM/投机/Medusa/Eagle/自投机/RingAttn/TritonFlash/CudaGraph
│   ├── quantization/      # 4题：AWQ/GPTQ/SmoothQuant/W8A16
│   ├── sparsity/          # 2题：2:4结构化稀疏/Wanda+SparseGPT
│   ├── ssm_attention/     # 5题：Mamba/Mamba2/LinearAttention/GLA/HybridSSM
│   └── advanced/          # 11题：ChunkedPrefill/预设分离/DistServe/动态Rollout/KV压缩/LoRA多适配器/MoE调度/负载均衡/MTP/前缀缓存/TestTimeCompute
├── Easy_deepseek/         # DeepSeek 模型完整实现（12 .py：MLA/MoE/MTP/FP8量化/分布式 等）
├── Easy_RL/               # 强化学习对齐（12 .ipynb：REINFORCE/A2C/PPO/GRPO/GSPO/DPO/SimPO/KTO/ORPO/RewardModel/RLHF_Pipeline/IterativeDPO）
├── AIInfra面经/            # AIInfra 面试面经（字节/阿里/腾讯/百度/快手/小米/旷视/蚂蚁/讯飞/美团/京东 + AIInfra专题）
├── 大厂常见思维题/         # 思维题（小红书/字节 + AIInfra专题）
└── README.md
```

## ✨ 各模块说明

### 📌 Easy_Pytorch — 基础手撕（4 .py + 4 .ipynb）
- **core_losses.py**: 生产级损失函数（Softmax/CE+LabelSmoothing/KLDiv/InfoNCE）
- **core_normalizations.py**: 生产级归一化（LayerNorm/RMSNorm/BatchNorm，Llama风格）
- **core_optimizers.py**: 生产级优化器（SGD/Adam/AdamW，与torch.optim对齐）
- **core_basics.py**: 基础算法（K-means/数值梯度检查）
- 4 ipynb: Backpropagation(两层MLP手动反向传播+autograd对拍) / Dropout / LabelSmoothing / WarmupCosineScheduler

### 📌 Easy_Attention — 注意力与组件（1 .py + 10 .ipynb + test.py）
- **core_attention.py**: 生产级注意力核心（SDPA/MHA/绝对PE/RoPE/SwiGLU，type hints+测试全覆盖）
- **test.py**: 18 项测试覆盖（含 core_attention.py 全部模块的 forward/gradient/数值性质验证）
- 10 ipynb: GQA_MLA / RMSNorm / MLP(SwiGLU) / FFN / LoRA / ALiBi / AttentionSink / QKNorm / ActivationFunction / FlashAttentionKernel(Tiling+Online Softmax) / SparseAttention(SlidingWindow+Global)

### 📌 Easy_Generator — 解码采样（8 题）
贪心搜索、Top-K、Top-P(nucleus)、温度采样、束搜索(BeamSearch)、对比搜索(ContrastiveSearch)、重复惩罚(RepetitionPenalty)、LogitsProcessor 框架，含组合顺序说明。每个文件末尾含测试验证 cell。

### 📌 Easy_Tokenizer — 分词算法（4 题）
BPE(GPT-2)、WordPiece(BERT)、Unigram(T5)、SentencePiece(LLaMA)，含 Viterbi 分词与 ▁ 空格标记。每个文件末尾含测试验证 cell。

### 📌 Easy_AIInfra — 训练/推理优化（48 .py，含完整 type hints + return hints + assert 测试）
分 6 个子目录：training(14)/inference(12)/quantization(4)/sparsity(2)/ssm_attention(5)/advanced(11)
覆盖 DDP/ZeRO/TP/PP/FSDP/Hybrid/SP/FusedKernels，KVCache/PagedAttn/FlashAttn/vLLM/投机/Mamba/线性Attention/RingAttn/量化/稀疏/MTP/MCTS 等，所有函数均有类型标注与测试 assert。

### 📌 Easy_deepseek — DeepSeek 实现（12 .py）
模型核心结构完整实现 + DeepSeekMoE / MLA / MTP / FP8量化 / 分布式 等关键模块详细解析。最近优化：补充 return type hints、删除重复 kernel.py、修复 dtype 参数类型标注。

### 📌 Easy_RL — 对齐算法（12 题）
REINFORCE（基础策略梯度）、A2C（Actor-Critic）、PPO（RLHF 经典）、GRPO（组相对，省 critic）、GSPO（序列级重要性，Qwen3）、DPO（绕过 RM）、SimPO（去 reference）、KTO（前景理论，无需成对）、ORPO（无需 reference）、RewardModel(Bradley-Terry)、RLHF_Pipeline(SFT→RM→PPO)、IterativeDPO(多轮自改进)，均含题目背景 + 原理 + 考察点 + 测试验证。

### 📌 面经与思维题
`AIInfra面经/` 每公司一个文件（字节/阿里/腾讯/百度/快手/小米/旷视/蚂蚁/讯飞/美团/京东），含 AI Infra 专题面经与系统设计题；`大厂常见思维题/` 收录思维题，含 AI Infra 方向计算题。

---

## 📊 代码质量

| 指标 | 状态 |
|------|------|
| `.py` 文件背景/考察点/测试 | **75/75** (100%) |
| `.ipynb` 文件背景/考察点/测试 | **38/38** (100%) |
| `.ipynb` 函数 type hints | **144/144** (100%) |
| `.py` 函数 type hints + return hints | **~200+** (Easy_AIInfra 46 文件全覆盖) |
| 语法错误 | **0** |

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
