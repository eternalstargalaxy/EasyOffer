"""
【题目】Dynamic Rollout：动态 rollout 长度

【背景】
RL 训练中 rollout 长度固定会导致短回复任务浪费算力、长回复任务截断。
Dynamic rollout 根据模型当前生成状态动态决定 rollout 长度：
- 设 min/max rollout 长度边界
- 检测 EOS 或 stop tokens 提前终止
- 根据历史平均长度自适应调整后续 rollout 预算

【输入/输出】
- 输入：model, prompt, min_len, max_len, stop_tokens
- 输出：动态长度的 rollout 序列

【考察点】
- 提前终止条件与 KV cache 清理
- 自适应长度调整策略
- 提示：while 循环 + 条件判断
"""
import torch
import torch.nn as nn


class DynamicRollout:
    def __init__(self, min_len: int = 4, max_len: int = 128, stop_tokens: torch.Tensor = None):
        self.min_len = min_len
        self.max_len = max_len
        self.stop_tokens = stop_tokens or []
        self.history = []

    def rollout(self, model: nn.Module, prompt: torch.Tensor) -> tuple:
        """动态 rollout：根据 EOS/stop 提前终止。"""
        tokens = list(prompt)
        while len(tokens) - len(prompt) < self.max_len:
            x = torch.tensor([tokens], dtype=torch.long)
            with torch.no_grad():
                logits = model(x)
            nxt = torch.argmax(logits[0, -1]).item()
            tokens.append(nxt)
            if (len(tokens) - len(prompt) >= self.min_len and
                    nxt in self.stop_tokens):
                break
        gen_len = len(tokens) - len(prompt)
        self.history.append(gen_len)
        return tokens, gen_len

    def adaptive_max_len(self, percentile: float = 0.9) -> int:
        """根据历史长度自适应调整 max_len。"""
        if not self.history:
            return self.max_len
        sorted_h = sorted(self.history)
        idx = int(len(sorted_h) * percentile)
        return max(self.min_len, min(self.max_len, int(sorted_h[idx] * 1.5)))


class TinyLM(nn.Module):
    def __init__(self, vocab: int, hidden: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.rnn(self.embed(x))[0])


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    vocab = 20
    model = TinyLM(vocab)
    dr = DynamicRollout(min_len=2, max_len=10, stop_tokens=[0])
    tokens, length = dr.rollout(model, [1, 2, 3])
    assert 2 <= length <= 10
    print(f"✅ Rollout: 生成长度 {length} (在 [2, 10] 内)")

    dr2 = DynamicRollout(min_len=5, max_len=10, stop_tokens=[0])
    tokens2, length2 = dr2.rollout(model, [1, 2])
    assert length2 >= 5
    print(f"✅ min_len 约束: 生成长度 {length2} >= 5")

    for _ in range(10):
        dr.rollout(model, [1, 2, 3])
    adaptive = dr.adaptive_max_len()
    assert adaptive >= dr.min_len
    print(f"✅ 自适应 max_len: {adaptive} (历史 {len(dr.history)} 次)")

    dr3 = DynamicRollout(min_len=1, max_len=3, stop_tokens=[])
    _, l3 = dr3.rollout(model, [1])
    assert l3 == 3, f"无 stop token 应到 max_len, 实际 {l3}"
    print("✅ 无 stop token: 到 max_len 终止")
    print("✅ 全部测试通过")
