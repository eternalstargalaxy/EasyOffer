"""
【题目】损失函数合集：Softmax / CrossEntropy / KLDivergence / InfoNCE

【背景】
大模型训练中损失函数是核心组件。本文件实现四种关键损失函数的 PyTorch 版本，
贴近真实训练框架（如 HuggingFace Transformers）的实现风格。
- Softmax：数值稳定的 log-softmax，避免溢出
- CrossEntropy：支持 label smoothing，与 F.cross_entropy 对齐
- KLDivergence：用于 DPO/KTO 等对齐算法
- InfoNCE：对比学习损失，用于 SimCSE/Embedding 训练

【考察点】
- log-sum-exp 数值稳定技巧
- label smoothing 的实现（改 Softmax 目标分布）
- InfoNCE 的正负样本构造
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数值稳定 softmax：减最大值防溢出。"""
    return logits.softmax(dim=dim)


def stable_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数值稳定 log-softmax：直接用 log-sum-exp，避免 log(0)。"""
    return logits.log_softmax(dim=dim)


def cross_entropy_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    支持 label smoothing 的交叉熵。
    当 label_smoothing=0 时等价于 F.cross_entropy。
    label_smoothing=ε 时，目标分布从 one-hot 变为 (1-ε)*one-hot + ε/V。
    """
    if label_smoothing == 0:
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)

    num_classes = logits.size(-1)
    log_probs = stable_log_softmax(logits, dim=-1)

    mask = targets != ignore_index
    nll_loss = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll_loss = nll_loss * mask

    smooth_loss = -log_probs.mean(dim=-1)
    smooth_loss = smooth_loss * mask

    n = mask.sum()
    loss = (1 - label_smoothing) * nll_loss + label_smoothing * smooth_loss
    return loss.sum() / n


def kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL(p || q) = sum(p * log(p/q))，输入为 logits。
    用于 DPO: KL(policy || reference)。
    """
    p_log_probs = stable_log_softmax(p_logits, dim=-1)
    q_log_probs = stable_log_softmax(q_logits, dim=-1)
    p_probs = p_log_probs.exp()
    kl = (p_probs * (p_log_probs - q_log_probs)).sum(dim=-1)
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    return kl


def info_nce_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE 对比学习损失（SimCSE 风格）。
    embeddings: [B, D]，labels: [B]，同 label 为正样本。
    """
    device = embeddings.device
    batch_size = embeddings.size(0)

    features = F.normalize(embeddings, dim=-1)
    sim = features @ features.T / temperature

    mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask.fill_diagonal_(False)

    pos_mask = mask.float()
    neg_mask = (~mask).float()
    neg_mask.fill_diagonal_(False)

    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_logits = logits.exp()

    pos_sum = (exp_logits * pos_mask).sum(dim=1)
    neg_sum = (exp_logits * neg_mask).sum(dim=1)

    loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8) + 1e-8)
    return loss.mean()


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)

    logits = torch.randn(4, 10)
    targets = torch.tensor([2, 0, 4, 7])

    p = stable_softmax(logits)
    assert p.shape == (4, 10)
    assert torch.allclose(p.sum(dim=-1), torch.ones(4), atol=1e-6)
    print("✅ stable_softmax: 归一化正确")

    loss = cross_entropy_with_label_smoothing(logits, targets, label_smoothing=0.0)
    ref = F.cross_entropy(logits, targets)
    assert torch.allclose(loss, ref, atol=1e-6), f"{loss} vs {ref}"
    print(f"✅ CE (no smoothing): 与 F.cross_entropy 一致 ({loss.item():.4f})")

    loss_smooth = cross_entropy_with_label_smoothing(logits, targets, label_smoothing=0.1)
    assert loss_smooth.item() > 0
    print(f"✅ CE (smoothing=0.1): {loss_smooth.item():.4f}")

    p_logits = torch.randn(4, 10)
    q_logits = torch.randn(4, 10)
    kl = kl_divergence(p_logits, p_logits)
    assert kl.item() < 1e-5, "KL(p||p) 应为 0"
    kl2 = kl_divergence(p_logits, q_logits)
    assert kl2.item() >= -1e-5, "KL 应非负"
    print(f"✅ KL: KL(p||p)={kl.item():.6f}, KL(p||q)={kl2.item():.4f}")

    emb = torch.randn(8, 32)
    lbl = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    cls_loss = info_nce_loss(emb, lbl, temperature=0.1)
    assert cls_loss.item() > 0
    print(f"✅ InfoNCE: {cls_loss.item():.4f}")
    print("✅ 全部测试通过")
