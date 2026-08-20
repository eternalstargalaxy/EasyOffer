"""
【题目】Medusa：多头并行解码加速

【背景】
Medusa(2024)在 LLM 最后一层后挂多个(通常3-5个)独立分类头(medusa heads)，
每个 head 预测未来第 k 步 token。LLM 一次前向算出 hidden state，
所有 medusa head 并行预测多步未来 token，大模型批量验证和接受。
树形注意力(tree attention)让候选 token 形成候选树，一次验证。
不需要单独 draft model，每个 medusa head 是一个线性层(hidden->vocab)。

【输入/输出】
- 输入：hidden_state [B,D], medusa_heads, lm_head, max_new_tokens
- 输出：生成的 token 序列

【考察点】
- 多头并行 vs 单 draft model 的 trade-off
- 树形注意力 mask 构造（torch.triu）
- 提示：torch.multinomial 批量采样
"""
import torch
import torch.nn as nn


class MedusaHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MedusaModel(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, num_heads: int = 4):
        super().__init__()
        self.heads = nn.ModuleList([MedusaHead(hidden_dim, vocab_size) for _ in range(num_heads)])

    def forward(self, h: torch.Tensor):
        raise NotImplementedError


def medusa_decode(llm, medusa: MedusaModel, lm_head, h: torch.Tensor,
                  max_new: int = 10, temperature: float = 1.0):
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    D, V, n_heads = 128, 32000, 4
    try:
        m = MedusaModel(D, V, n_heads)
        h = torch.randn(2, D)
        candidates = m(h)
        assert len(candidates) == n_heads
        print('✅' + " Medusa 测试通过")
    except NotImplementedError:
        print('ℹ' + " 待实现")
