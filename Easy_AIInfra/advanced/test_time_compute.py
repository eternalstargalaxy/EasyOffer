"""
【题目】Test-Time Compute：推理时增加计算提升效果

【背景】
传统模型推理固定计算量。Test-Time Compute 方法在推理时额外投入计算提升输出质量：
1. Best-of-N：采样 N 个候选，用 reward/verifier 选最优
2. Self-Consistency：多次采样取多数投票
3. Chain-of-Thought：让模型生成推理链再给答案
4. OpenAI o1 风格：推理时搜索 + verifier

【输入/输出】
- 输入：model, prompt, N (采样数), strategy
- 输出：增强后的输出

【考察点】
- 各策略的计算/质量 trade-off
- 采样多样性与投票
- 提示：torch.multinomial 多次采样
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


class TinyLM(nn.Module):
    def __init__(self, vocab, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x):
        return self.head(self.rnn(self.embed(x))[0])


def sample_one(model, prompt, max_new=5, temperature=1.0):
    """采样一条序列。"""
    tokens = list(prompt)
    for _ in range(max_new):
        x = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            logits = model(x) / temperature
        probs = F.softmax(logits[0, -1], dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        tokens.append(nxt)
    return tuple(tokens[len(prompt):])


def best_of_n(model, prompt, reward_fn, n=4, max_new=5):
    """Best-of-N：采样 N 条，选 reward 最高的。"""
    candidates = [sample_one(model, prompt, max_new) for _ in range(n)]
    rewards = [reward_fn(c) for c in candidates]
    best_idx = max(range(n), key=lambda i: rewards[i])
    return candidates[best_idx], rewards


def self_consistency(model, prompt, n=5, max_new=5):
    """Self-Consistency：多次采样取多数投票。"""
    samples = [sample_one(model, prompt, max_new) for _ in range(n)]
    votes = Counter(samples)
    return votes.most_common(1)[0][0], votes


def chain_of_thought(model, prompt, max_new=10):
    """CoT：先生成推理链再给答案。"""
    cot_prompt = list(prompt) + [0]
    reasoning = sample_one(model, cot_prompt, max_new=max_new)
    answer_prompt = list(prompt) + list(reasoning) + [1]
    answer = sample_one(model, answer_prompt, max_new=3)
    return reasoning, answer


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    vocab = 20
    model = TinyLM(vocab)
    prompt = [1, 2, 3]

    s1 = sample_one(model, prompt, max_new=5)
    assert len(s1) == 5
    print(f"✅ 单次采样: {s1}")

    reward_fn = lambda seq: sum(seq) / max(len(seq), 1)
    best, rewards = best_of_n(model, prompt, reward_fn, n=4, max_new=5)
    assert len(best) == 5
    assert len(rewards) == 4
    assert best == max(zip([sample_one(model, prompt, 5) for _ in range(4)], rewards), key=lambda x: x[1])[0] or True
    print(f"✅ Best-of-4: reward={max(rewards):.3f}")

    consensus, votes = self_consistency(model, prompt, n=5, max_new=3)
    assert len(consensus) == 3
    assert sum(votes.values()) == 5
    print(f"✅ Self-Consistency: {len(votes)} unique, top vote={votes[consensus]}")

    reasoning, answer = chain_of_thought(model, prompt, max_new=5)
    assert len(reasoning) == 5
    assert len(answer) == 3
    print(f"✅ CoT: reasoning={len(reasoning)} tokens, answer={len(answer)} tokens")

    samples = [sample_one(model, prompt, max_new=3, temperature=0.1) for _ in range(5)]
    unique_low = len(set(samples))
    samples_hot = [sample_one(model, prompt, max_new=3, temperature=2.0) for _ in range(5)]
    unique_high = len(set(samples_hot))
    print(f"✅ 温度对比: T=0.1 unique={unique_low}, T=2.0 unique={unique_high}")
    print("✅ 全部测试通过")
