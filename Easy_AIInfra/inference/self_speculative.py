"""
【题目】自投机解码 (Self-Speculative Decoding)

【背景】
不依赖外部 draft model，用大模型自身跳过部分层快速生成 draft token，
再由完整大模型验证。核心思想：大模型去掉最后几层->轻量 draft，
完整大模型->验证器。两个模型共享前面所有层，只跑一次前向。
实现：前 K 层输出处 early exit，简单投影头预测 draft token，
同时完整前向验证。acceptance rate 比独立 draft model 高。

【输入/输出】
- 输入：LLM, early_exit_layer_idx, draft_head, draft_steps
- 输出：接受的 token 序列

【考察点】
- early exit 层选择：太浅精度低，太深加速少
- 共享前向 vs 独立 draft 的 trade-off
- 提示：torch.no_grad 减少显存开销
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class SelfSpeculativeLLM(nn.Module):
    """大模型 + early exit draft head。"""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, vocab_size: int, early_exit_layer: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.draft_head = nn.Linear(d_model, vocab_size, bias=False)
        self.early_exit = early_exit_layer

    def forward(self, x: torch.Tensor, use_early_exit: bool = False):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if use_early_exit and i == self.early_exit - 1:
                draft_logits = self.draft_head(h)
        full_logits = self.lm_head(h)
        if use_early_exit:
            return full_logits, draft_logits
        return full_logits


def self_speculative_step(llm_layers: list, draft_head: nn.Module, lm_head: nn.Module,
                          h: torch.Tensor, early_exit: int,
                          draft_steps: int = 3):
    """
    1. 前 early_exit 层 -> draft_head 预测 draft_steps 个 token
    2. 剩余层 + lm_head 完整前向验证
    3. 逐位接受/拒绝
    """
    for i in range(early_exit):
        h = llm_layers[i](h)
    draft_logits = draft_head(h)
    draft_probs = F.softmax(draft_logits, dim=-1)
    draft_tokens = torch.argmax(draft_logits, dim=-1)

    for i in range(early_exit, len(llm_layers)):
        h = llm_layers[i](h)
    target_logits = lm_head(h)
    target_probs = F.softmax(target_logits, dim=-1)

    accepted = 0
    for i in range(min(draft_steps, draft_tokens.shape[1])):
        cand = draft_tokens[:, i]
        pt = target_probs[:, i, :].gather(1, cand.unsqueeze(1)).squeeze(1)
        pd = draft_probs[:, i, :].gather(1, cand.unsqueeze(1)).squeeze(1)
        ratio = pt / (pd + 1e-12)
        r = torch.rand_like(ratio)
        if (r < ratio).all():
            accepted += 1
        else:
            break
    return draft_tokens, accepted


# ===== 测试验证 =====
if __name__ == '__main__':
    torch.manual_seed(42)
    d_model, n_heads, n_layers, vocab = 64, 4, 6, 100
    early_exit = 3

    model = SelfSpeculativeLLM(d_model, n_heads, n_layers, vocab, early_exit)
    x = torch.randn(2, 5, d_model)

    full_logits = model(x, use_early_exit=False)
    assert full_logits.shape == (2, 5, vocab)
    print(f"✅ 完整前向: {full_logits.shape}")

    full_logits, draft_logits = model(x, use_early_exit=True)
    assert full_logits.shape == (2, 5, vocab)
    assert draft_logits.shape == (2, 5, vocab)
    print(f"✅ Early exit: draft {draft_logits.shape} + full {full_logits.shape}")

    draft_tokens, accepted = self_speculative_step(
        model.layers, model.draft_head, model.lm_head, x, early_exit, draft_steps=3
    )
    assert draft_tokens.shape == (2, 5)
    assert 0 <= accepted <= 3
    print(f"✅ self_speculative_step: 接受 {accepted}/3")

    model2 = SelfSpeculativeLLM(d_model, n_heads, n_layers, vocab, early_exit=1)
    d_tok, acc = self_speculative_step(
        model2.layers, model2.draft_head, model2.lm_head, x, 1, draft_steps=3
    )
    assert d_tok.shape == (2, 5)
    print(f"✅ early_exit=1 (最浅): 接受 {acc}/3")

    model3 = SelfSpeculativeLLM(d_model, n_heads, n_layers, vocab, early_exit=5)
    d_tok3, acc3 = self_speculative_step(
        model3.layers, model3.draft_head, model3.lm_head, x, 5, draft_steps=3
    )
    assert d_tok3.shape == (2, 5)
    print(f"✅ early_exit=5 (较深): 接受 {acc3}/3")
    print("✅ 全部测试通过")
