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
import torch.nn.functional as F


class MedusaHead(nn.Module):
    """单个 medusa head：hidden -> vocab logits。"""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


class MedusaModel(nn.Module):
    """多个 medusa head 并行预测多步未来 token。"""

    def __init__(self, hidden_dim: int, vocab_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            MedusaHead(hidden_dim, vocab_size) for _ in range(num_heads)
        ])

    def forward(self, h: torch.Tensor):
        """h: [B, D] -> List[Tensor[B, V]]，每个 head 的 logits。"""
        return [head(h) for head in self.heads]

    def predict_tokens(self, h: torch.Tensor, temperature: float = 1.0):
        """返回每 head 的 top-1 token 和概率。"""
        tokens = []
        probs_list = []
        for head in self.heads:
            logits = head(h) / temperature
            probs = F.softmax(logits, dim=-1)
            tok = torch.argmax(logits, dim=-1)
            tokens.append(tok)
            probs_list.append(probs)
        return tokens, probs_list


def build_tree_mask(num_heads: int, top_k: int = 1) -> torch.Tensor:
    """
    构建树形注意力 mask。
    候选树：root + num_heads 个分支，每分支 top_k 个 token。
    总节点数 = 1 + num_heads * top_k。
    """
    total = 1 + num_heads * top_k
    mask = torch.zeros(total, total, dtype=torch.bool)
    for i in range(total):
        for j in range(i + 1):
            mask[i, j] = True
    return mask


def medusa_verify(target_logits: torch.Tensor, candidate_tokens: list,
                  candidate_probs: list) -> int:
    """
    验证 medusa 候选 token。
    target_logits: [num_heads, V] 大模型对每个候选位置的 logits。
    返回接受的头数。
    """
    accepted = 0
    for i, (cand, cprob) in enumerate(zip(candidate_tokens, candidate_probs)):
        target_prob = F.softmax(target_logits[i], dim=-1)
        ratio = target_prob[cand] / (cprob[cand] + 1e-12)
        r = torch.rand(1).item()
        if r < min(1.0, ratio.item()):
            accepted += 1
        else:
            break
    return accepted


def medusa_decode(llm: nn.Module, medusa: MedusaModel, lm_head: nn.Module, h: torch.Tensor,
                  max_new: int = 10, temperature: float = 1.0):
    """完整 Medusa 解码循环。"""
    tokens = []
    cur_h = h
    while len(tokens) < max_new:
        base_logits = lm_head(cur_h) / temperature
        base_token = torch.argmax(base_logits, dim=-1)
        tokens.append(base_token.item())

        cand_tokens, cand_probs = medusa.predict_tokens(cur_h, temperature)
        with torch.no_grad():
            target_logits = torch.stack([
                lm_head(cur_h) for _ in range(medusa.num_heads)
            ])
        accepted = medusa_verify(target_logits, cand_tokens, cand_probs)
        for i in range(accepted):
            if len(tokens) < max_new:
                tokens.append(cand_tokens[i].item())
        if len(tokens) < max_new:
            with torch.no_grad():
                cur_h = llm(torch.tensor([[tokens[-1]]]))
    return tokens[:max_new]


# ===== 测试验证 =====
if __name__ == '__main__':
    torch.manual_seed(42)
    D, V, n_heads = 128, 1000, 4

    m = MedusaModel(D, V, n_heads)
    h = torch.randn(2, D)
    logits_list = m(h)
    assert len(logits_list) == n_heads
    assert all(l.shape == (2, V) for l in logits_list)
    print(f"✅ MedusaModel: {h.shape} -> {n_heads} x {logits_list[0].shape}")

    tokens, probs = m.predict_tokens(h)
    assert len(tokens) == n_heads
    assert all(t.shape == (2,) for t in tokens)
    print(f"✅ predict_tokens: {n_heads} 个候选 token")

    mask = build_tree_mask(n_heads, top_k=1)
    total = 1 + n_heads
    assert mask.shape == (total, total)
    assert mask[0, 0] == True
    assert mask[1, 0] == True
    print(f"✅ build_tree_mask: {mask.shape} 树形 mask")

    target_logits = torch.randn(n_heads, V)
    cand_tokens = [torch.tensor([t]) for t in range(n_heads)]
    cand_probs = [F.softmax(torch.randn(1, V), dim=-1).squeeze(0) for _ in range(n_heads)]
    accepted = medusa_verify(target_logits, cand_tokens, cand_probs)
    assert 0 <= accepted <= n_heads
    print(f"✅ medusa_verify: 接受 {accepted}/{n_heads}")
    print("✅ 全部测试通过")
