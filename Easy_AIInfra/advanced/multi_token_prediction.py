"""
【题目】Multi-Token Prediction (MTP)

【背景】
传统自回归一次只预测一个 token，训练效率低。MTP 同时预测未来 k 个 token：
1. 主 head 预测 t+1
2. 额外 MTP head 预测 t+2, t+3, ..., t+k
3. 训练时所有 head 同时有监督信号，推理时可做投机验证
DeepSeek-V3 用 MTP 加速训练 + 推理。

【输入/输出】
- 输入：hidden_states [B, S, D], mtp_heads, k
- 输出：k 个未来 token 的 logits

【考察点】
- 多 head 共享 backbone 的效率
- 训练 loss = sum(CE(head_i, target_{t+i}))
- 推理时投机验证
- 提示：nn.ModuleList 多 head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    """单个 MTP head：预测第 k 步未来 token。"""

    def __init__(self, dim: int, vocab: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.ln(self.proj(h)))


class MTPModel(nn.Module):
    """Backbone + k 个 MTP head。"""

    def __init__(self, dim: int, vocab: int, num_mtp: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.main_head = nn.Linear(dim, vocab, bias=False)
        self.mtp_heads = nn.ModuleList([
            MTPHead(dim, vocab) for _ in range(num_mtp)
        ])
        self.num_mtp = num_mtp

    def forward(self, tokens: list) -> tuple:
        h = self.embed(tokens)
        h, _ = self.rnn(h)
        main_logits = self.main_head(h)
        mtp_logits = [head(h) for head in self.mtp_heads]
        return main_logits, mtp_logits

    def loss(self, tokens: list, targets: torch.Tensor) -> torch.Tensor:
        """训练 loss：主 head + MTP heads。"""
        main_logits, mtp_logits = self.forward(tokens)
        loss = F.cross_entropy(main_logits[:, :-1].reshape(-1, main_logits.shape[-1]),
                               targets[:, 1:].reshape(-1))
        for i, mtp_l in enumerate(mtp_logits):
            if i + 2 < targets.shape[1]:
                loss += F.cross_entropy(
                    mtp_l[:, :-(i+2)].reshape(-1, mtp_l.shape[-1]),
                    targets[:, i+2:].reshape(-1)
                )
        return loss


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    vocab, dim = 50, 32
    model = MTPModel(dim, vocab, num_mtp=3)
    tokens = torch.randint(0, vocab, (2, 10))

    main_l, mtp_l = model(tokens)
    assert main_l.shape == (2, 10, vocab)
    assert len(mtp_l) == 3
    assert all(l.shape == (2, 10, vocab) for l in mtp_l)
    print(f"✅ MTP forward: main {main_l.shape} + {len(mtp_l)} MTP heads")

    loss = model.loss(tokens, tokens)
    assert loss.item() > 0
    print(f"✅ MTP loss: {loss.item():.4f}")

    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    assert grad_count > 0
    print(f"✅ 反向传播: {grad_count} params 有梯度")

    for k in [1, 2, 5]:
        m_k = MTPModel(dim, vocab, num_mtp=k)
        _, mtp_k = m_k(tokens)
        assert len(mtp_k) == k
        print(f"  num_mtp={k}: {len(mtp_k)} heads")
    print("✅ 不同 MTP 数量正确")
    print("✅ 全部测试通过")
