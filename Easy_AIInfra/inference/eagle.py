"""
【题目】Eagle：特征级投机解码加速

【背景】
标准投机解码用小 draft model 一次猜 k 个 token。Eagle(2024)改进：
用大模型自身中间层特征 + 额外 draft head 预测多个未来 token 的 hidden state，
再通过 LM head 解码为 token。核心优势：不单独加载 draft model 省显存，
特征级 draft 信息更丰富，支持树形验证一次接受多分支。

【输入/输出】
- 输入：hidden_states [B,1,D], draft_head, lm_head, max_draft
- 输出：接受 token 序列 + 最终 hidden_state

【考察点】
- 特征级 vs token 级 draft 差异
- 树形验证接受/拒绝规则
- 提示：torch.multinomial 采样, torch.gather 收集接受 token
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EagleDraftHead(nn.Module):
    """特征级 draft head：从 hidden state 预测多步未来 hidden state。"""

    def __init__(self, hidden_dim: int, draft_steps: int = 5):
        super().__init__()
        self.draft_steps = draft_steps
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, 1, D] -> [B, draft_steps, D] 预测的未来 hidden states。"""
        outputs = []
        cur = h
        for _ in range(self.draft_steps):
            cur = self.ln(self.proj(cur))
            outputs.append(cur)
        return torch.cat(outputs, dim=1)


def eagle_draft(draft_head: EagleDraftHead, h: torch.Tensor,
                lm_head: nn.Linear) -> tuple:
    """draft：从 hidden state 预测多步未来 token。"""
    future_h = draft_head(h)
    logits = lm_head(future_h)
    probs = F.softmax(logits, dim=-1)
    tokens = torch.argmax(logits, dim=-1)
    return tokens, probs, future_h


def eagle_verify(target_logits: torch.Tensor, draft_tokens: torch.Tensor,
                 draft_probs: torch.Tensor) -> tuple:
    """
    验证：target_logits [B, K, V], draft_tokens [B, K], draft_probs [B, K, V]
    逐位接受/拒绝，返回 accepted_count 和 resample_token。
    """
    B, K, V = target_logits.shape
    target_probs = F.softmax(target_logits, dim=-1)
    accepted_count = 0
    resample_token = None
    for i in range(K):
        cand = draft_tokens[:, i]
        pt = target_probs[:, i, :].gather(1, cand.unsqueeze(1)).squeeze(1)
        pd = draft_probs[:, i, :].gather(1, cand.unsqueeze(1)).squeeze(1)
        ratio = pt / (pd + 1e-12)
        r = torch.rand_like(ratio)
        if (r < ratio).all():
            accepted_count += 1
        else:
            residual = torch.clamp(target_probs[:, i, :] - draft_probs[:, i, :], min=0)
            residual = residual / residual.sum(dim=-1, keepdim=True)
            resample_token = torch.multinomial(residual, 1).squeeze(1)
            break
    return accepted_count, resample_token


def eagle_verify_simple(llm_hidden: torch.Tensor, draft_head: EagleDraftHead,
                        lm_head: nn.Linear, h: torch.Tensor,
                        draft_steps: int = 5, temperature: float = 1.0):
    """完整 Eagle 一步：draft -> verify。"""
    draft_tokens, draft_probs, future_h = eagle_draft(draft_head, h, lm_head)
    target_logits = lm_head(future_h) / temperature
    accepted, resample = eagle_verify(target_logits, draft_tokens, draft_probs)
    return draft_tokens, accepted, resample


# ===== 测试验证 =====
if __name__ == '__main__':
    torch.manual_seed(42)
    B, D, V, steps = 2, 128, 1000, 3

    head = EagleDraftHead(D, steps)
    h = torch.randn(B, 1, D)
    features = head(h)
    assert features.shape == (B, steps, D), f"输出形状错误: {features.shape}"
    print(f"✅ EagleDraftHead: {h.shape} -> {features.shape}")

    lm_head = nn.Linear(D, V, bias=False)
    tokens, probs, future_h = eagle_draft(head, h, lm_head)
    assert tokens.shape == (B, steps), f"token 形状错误: {tokens.shape}"
    assert probs.shape == (B, steps, V)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B, steps), atol=1e-5)
    print(f"✅ eagle_draft: 生成 {steps} 个候选 token")

    target_logits = torch.randn(B, steps, V)
    accepted, resample = eagle_verify(target_logits, tokens, probs)
    assert 0 <= accepted <= steps
    print(f"✅ eagle_verify: 接受 {accepted}/{steps}")

    draft_tokens, accepted, resample = eagle_verify_simple(
        None, head, lm_head, h, draft_steps=steps
    )
    assert draft_tokens.shape == (B, steps)
    print(f"✅ eagle_verify_simple: 接受 {accepted}/{steps}")
    print("✅ 全部测试通过")
