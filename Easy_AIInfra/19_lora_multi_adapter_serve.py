"""
【题目】多 LoRA adapter 推理调度

【背景】
服务多个用户各自挂不同 LoRA 时，若为每个 adapter 物化一份完整权重 (W + ΔW) 则显存爆炸。
利用 ΔW = B·A 的低秩结构（rank r 远小于 d），推理时共享 base 权重 W，只多存小秩矩阵 A/B，
按请求路由对应 adapter：base 部分统一算 x @ W，LoRA 部分按 adapter 分组算 (x·A)·B 再加回。
batch 内可混用不同 adapter，用 index map 把 token 路由到其 adapter 的 A/B。
高并发下"统一 base + 分组 LoRA"比合并权重更优：base GEMM 大算力可批量化，LoRA 部分小且可分组并行。

【输入/输出】
- 输入：x: Tensor[total_tokens, in_dim], adapter_ids: Tensor[total_tokens], base_W, adapters={id: (A,B)}
- 输出：y = x @ base_W + (x @ A_i) @ B_i，每 token 用各自 adapter

【考察点】
- 共享 base + 分组 LoRA 的计算组织
- batch 内多 adapter 的 token 路由与分组 GEMM
- rank、adapter 数对显存/算力的影响、热加载
"""
import torch
import torch.nn as nn


class MultiLoraLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, base_W: torch.Tensor):
        # TODO: base 权重 + adapters 字典 {id: (A [r, in], B [out, r])}
        raise NotImplementedError

    def load_adapter(self, adapter_id: int, A: torch.Tensor, B: torch.Tensor):
        raise NotImplementedError

    def unload_adapter(self, adapter_id: int):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor) -> torch.Tensor:
        """
        1. base_out = x @ base_W^T            # 一次大 GEMM，所有 token 共享
        2. 按 adapter_ids 把 token 分组
        3. 对每组 g: lora_out = (x_g @ A_g^T) @ B_g^T
        4. 把 lora_out 按 index 加回 base_out 对应行
        """
        raise NotImplementedError


def grouped_lora_gemm(x: torch.Tensor, groups: dict, adapters: dict):
    """
    groups: {adapter_id: token_indices}
    对每组算 (x[indices] @ A) @ B，拼回原位置
    （可用 torch.index_add_ 或 bmm 把组堆成 [G, r, ...] 批量算）
    """
    raise NotImplementedError


def mem_throughput_compare(num_adapters: int, rank: int, d: int):
    """返回"合并全权重" vs "共享 base + 分组 LoRA" 的显存与吞吐"""
    raise NotImplementedError
