"""
【题目】Multi-Token Prediction (MTP)：多头预测未来 token

【背景】
标准自回归一次只预测 1 token，MTP(DeepSeek-V3/Meta 2024)一次预测 K 个。
DeepSeek MTP：在最后一层后挂 K 个独立 head，每个 head i 预测第 t+i token。
head i 用 causal mask 看 0..t+i-1 位置，把 i-1 的 logit 作为额外输入。
训练损失 = sum_i CE(loss_i)，推理时 MTP 头可直接用做投机解码 draft。
Meta MTP：共享 head 参数，用 position offset 控制预测目标。
优势：训练更高效(data efficiency)，推理可作为免费投机解码。

【输入/输出】
- 输入：hidden [B,L,D], MTP heads, future steps K
- 输出：K 个未来 token 的 logits

【考察点】
- MTP 的 causal mask 设计(每步递增)
- 训练时多个 CE loss 联合优化
- 推理时 MTP 作为投机解码加速
- 提示：nn.ModuleList 存多个 head，torch.triu 构造阶梯 mask
"""
import torch; import torch.nn as nn; import torch.nn.functional as F


class MultiTokenHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, num_tokens: int = 2):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, vocab_size) for _ in range(num_tokens)])

    def forward(self, h: torch.Tensor) -> list:
        raise NotImplementedError


def mtp_loss(logits_list, labels, shift: int = 1):
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    D, V, K = 64, 100, 3
    try:
        m = MultiTokenHead(D, V, K)
        h = torch.randn(2, 10, D)
        logs = m(h)
        assert len(logs) == K
        print('✅' + " MTP 测试通过")
    except NotImplementedError:
        print('ℹ' + " 待实现")
