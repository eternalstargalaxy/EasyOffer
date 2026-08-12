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
"""
import torch


def draft(model_d, prefix: list, K: int):
    """
    小模型自回归生成 K 个候选 token，同时记录每步 draft 概率 p_d（用于 verify）。
    返回 candidates: List[int], draft_probs: Tensor[K, vocab]
    """
    raise NotImplementedError


def verify(model_t, prefix: list, candidates: list, draft_probs: torch.Tensor):
    """
    1. target 对 [prefix, candidates] 一次前向，取每步概率 p_t: Tensor[K, vocab]
    2. 逐位接受/拒绝：
       r < min(1, p_t[i, cand_i]/p_d[i, cand_i]) -> 接受，继续
       否则用 norm(max(0, p_t[i] - p_d[i])) 重采样 resample_token，停止
    返回 accepted_tokens(含可能的 resample_token), num_accepted
    """
    raise NotImplementedError


def speculative_step(model_d, model_t, prefix: list, K: int):
    """draft -> verify -> 拼接结果 -> 复用 target KV，循环直到达到目标长度。"""
    raise NotImplementedError


def equivalence_check(model_t):
    """采样大量样本，验证投机采样输出分布与纯 target 采样一致（KL≈0）。"""
    raise NotImplementedError
