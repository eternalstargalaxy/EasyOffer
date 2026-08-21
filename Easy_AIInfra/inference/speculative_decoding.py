"""
【题目】投机采样（Speculative Decoding：draft + verify）

【背景】
用小 draft 模型一次猜 K 个 token，再用大 target 模型对 [prefix, candidates] 一次并行前向
取每步概率 p_t，与 draft 概率 p_d 做接受/拒绝：
- 对每个候选位，r ~ U(0,1)，若 r < min(1, p_t(x)/p_d(x)) 接受该 token，继续下一位；
- 否则在该位用归一化的 max(0, p_t - p_d) 重采样一个 token 并停止。
最终输出分布严格等于纯 target 采样。被接受前缀的 target KV 可直接复用，draft KV 丢弃。

【输入/输出】
- 输入：draft_model, target_model, prefix tokens, K
- 输出：最终序列（分布等价于 target 自回归采样）

【考察点】
- 接受/拒绝规则与重采样分布的正确性（证明等价 target）
- target 并行验证的形状（一次前向算 K+1 个位置）与 KV 复用
- K 的选择、draft 与 target 的词表对齐
- 提示：torch.gather 收集接受部分；小模型先跑，大模型验证
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLM(nn.Module):
    """简易自回归语言模型：embedding -> GRU -> linear -> logits"""
    def __init__(self, vocab_size: int, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, tokens: list) -> torch.Tensor:
        h = self.embed(tokens)
        out, _ = self.rnn(h)
        return self.lm_head(out)


def draft(model_d: nn.Module, prefix: list, K: int) -> tuple:
    """小模型自回归生成 K 个候选 token，记录每步 draft 概率 p_d。"""
    tokens = list(prefix)
    candidates = []
    draft_probs = []
    for _ in range(K):
        x = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            logits = model_d(x)
        probs = F.softmax(logits[0, -1], dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        candidates.append(nxt)
        draft_probs.append(probs)
        tokens.append(nxt)
    return candidates, torch.stack(draft_probs)


def verify(model_t: nn.Module, prefix: list, candidates: list, draft_probs: torch.Tensor) -> tuple:
    """target 并行验证 + 逐位接受/拒绝。返回 accepted_tokens, num_accepted。"""
    full = prefix + candidates
    x = torch.tensor([full], dtype=torch.long)
    with torch.no_grad():
        logits = model_t(x)
    target_probs = F.softmax(logits[0], dim=-1)

    accepted = []
    num_accepted = 0
    for i, cand in enumerate(candidates):
        pt = target_probs[len(prefix) - 1 + i]
        pd = draft_probs[i]
        ratio = pt[cand].item() / max(pd[cand].item(), 1e-12)
        r = torch.rand(1).item()
        if r < min(1.0, ratio):
            accepted.append(cand)
            num_accepted += 1
        else:
            residual = torch.clamp(pt - pd, min=0)
            residual = residual / residual.sum()
            resample = torch.multinomial(residual, 1).item()
            accepted.append(resample)
            break
    return accepted, num_accepted


def speculative_step(model_d: nn.Module, model_t: nn.Module, prefix: list, K: int) -> tuple:
    """draft -> verify -> 拼接结果。"""
    candidates, draft_probs = draft(model_d, prefix, K)
    accepted, num_accepted = verify(model_t, prefix, candidates, draft_probs)
    return prefix + accepted, num_accepted


def speculative_generate(model_d: nn.Module, model_t: nn.Module, prefix: list, K: int, max_new: int) -> list:
    """循环投机采样直到生成 max_new 个 token。"""
    tokens = list(prefix)
    target_len = len(prefix) + max_new
    while len(tokens) < target_len:
        tokens, num_accepted = speculative_step(model_d, model_t, tokens, K)
    return tokens[:target_len]


def naive_generate(model: nn.Module, prefix: list, max_new: int) -> list:
    """纯 target 自回归采样（用于对比验证）。"""
    tokens = list(prefix)
    for _ in range(max_new):
        x = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        probs = F.softmax(logits[0, -1], dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        tokens.append(nxt)
    return tokens


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    vocab = 50
    draft_model = TinyLM(vocab, hidden=32).eval()
    target_model = TinyLM(vocab, hidden=64).eval()

    prefix = [1, 2, 3]
    K = 4

    candidates, draft_probs = draft(draft_model, prefix, K)
    assert len(candidates) == K, f"候选数应为 {K}"
    assert draft_probs.shape == (K, vocab), f"draft_probs 形状错误: {draft_probs.shape}"
    for i in range(K):
        assert abs(draft_probs[i].sum().item() - 1.0) < 1e-5, "draft 概率未归一化"
    print(f"✅ draft 生成 {K} 个候选: {candidates}")

    accepted, num_accepted = verify(target_model, prefix, candidates, draft_probs)
    assert num_accepted <= K, "接受数不应超过 K"
    assert len(accepted) >= 1, "至少应有一个输出（拒绝时重采样）"
    print(f"✅ verify 接受 {num_accepted} 个，输出: {accepted}")

    result = speculative_generate(draft_model, target_model, prefix, K, max_new=10)
    assert len(result) == len(prefix) + 10, f"最终长度错误: {len(result)}"
    print(f"✅ 投机采样生成 {len(result) - len(prefix)} 个新 token: {result}")

    naive_result = naive_generate(target_model, prefix, 10)
    assert len(naive_result) == len(prefix) + 10
    print(f"✅ 纯 target 采样: {naive_result}")
    print("✅ 全部测试通过")
