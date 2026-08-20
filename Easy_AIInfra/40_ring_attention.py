"""
【题目】RingAttention：长序列分布式注意力

【背景】
序列长度超过单卡显存时，RingAttention 沿 seq 维切分 Q/K/V 到多卡，
各卡轮流发送 K 块给下一卡，收到后算 attention + 更新 softmax。
通信：环形的 P2P send/recv，每卡需 P-1 轮传输，但每轮只传 1/P 数据。
与 FlashAttention 互补：FA 优化单卡显存，Ring 扩展到多卡。
应用：训练 128K+ 长上下文模型(如 Llama 3 long context)。

【输入/输出】
- 输入：Q,K,V 各卡持 1/PP 的 seq chunk, rank, pp_size
- 输出：各卡持自己那部分 O [B, L/PP, D]

【考察点】
- ring 通信 vs all-to-all 通信 trade-off
- softmax rescale 在多轮传输间维护数值正确性
- 提示：注意梯度也需要 ring 反向传递
"""
import torch; import torch.nn.functional as F


def ring_attention_block(Q, K, V, rank: int, world_size: int):
    raise NotImplementedError


# ===== 测试验证 =====
if __name__ == '__main__':
    print('ℹ' + " RingAttention 需多卡分布式环境")
    print("验证：单机多卡用 NCCL send/recv 环形传输 K 块")
