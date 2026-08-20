# Easy_Tokenizer — 分词算法手撕合集

> 覆盖主流分词算法，每个文件头部含**题目背景 + 原理 + 考察点**，末尾含**测试验证 cell**。

## 文件索引

| 文件 | 算法 | 核心思想 | 典型应用 |
|------|------|----------|----------|
| `BPE.ipynb` | BPE | 贪心合并最高频 byte pair | GPT-2/GPT-3 |
| `WordPiece.ipynb` | WordPiece | ## 前缀 + 贪心最长匹配 | BERT |
| `Unigram.ipynb` | Unigram | 概率模型 + Viterbi 分词 | T5/ALBERT |
| `SentencePiece.ipynb` | SentencePiece | 统一框架，▁ 标记空格 | LLaMA/Qwen |

## 算法对比

| 维度 | BPE | WordPiece | Unigram | SentencePiece |
|------|-----|-----------|---------|---------------|
| 训练方向 | 合并（自底向上） | 合并（自底向上） | 删除（自顶向下） | 框架（可选） |
| 编码方式 | 贪心应用合并 | 贪心最长匹配 | Viterbi DP | 取决于子模型 |
| 空格处理 | </w> 后缀 | ## 前缀 | ▁ 前缀 | ▁ 前缀 |
| 可逆性 | 需后处理 | 需后处理 | 需后处理 | 原生可逆 |

每个 ipynb 末尾含测试验证 cell。