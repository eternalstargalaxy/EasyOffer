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


def self_speculative_step(llm_layers, draft_head, lm_head,
                          h: torch.Tensor, early_exit: int,
                          draft_steps: int = 3):
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    print('ℹ' + " 自投机解码实现框架")
    print("定义 self_speculative_step 实现 early exit + draft token 生成")
    print("再由 llm_layers[early_exit:] 完整前向验证")
