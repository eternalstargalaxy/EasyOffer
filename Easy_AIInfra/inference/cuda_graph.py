"""
【题目】手撕 CUDA Graph 捕获与重放

【背景】
CUDA Graph 将一系列 GPU kernel 录制为图，之后直接重放，
省去 kernel launch overhead。对 decode 阶段（小 batch、大量小 kernel）效果显著。

【输入/输出】
输入: 模型 + 固定 shape 的输入
输出: CUDA Graph 包装后的快速推理器

【考察点】
- graph capture / replay 的流程
- 静态输入要求（需 fixed shape + static buffer）
- warmup 的必要性（cuDNN 等会做 autotuning）
"""

import torch
import torch.nn as nn


class CUDAGraphRunner:
    """
    用 CUDA Graph 加速固定 shape 的推理
    适用于 decode 阶段（输入 shape 固定且小）
    """

    def __init__(self, model: nn.Module, example_input: torch.Tensor, warmup_steps: int = 3):
        self.model = model
        self.device = example_input.device

        # 静态输入/输出 buffer
        self.static_input = example_input.clone()
        self.static_output = None

        # Warmup（让 cuDNN/cuBLAS 完成 autotuning）
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = model(self.static_input)
            torch.cuda.synchronize()

        # 捕获 graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = model(self.static_input)

    def run(self, input_tensor: torch.Tensor):
        """重放 graph，只需拷贝输入到 static buffer"""
        self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output.clone()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ 无 CUDA，跳过 CUDA Graph 验证")
        # CPU 模拟验证逻辑
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        x = torch.randn(1, 10, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape
        print("✅ CPU fallback 验证通过")
    else:
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)).cuda()
        x = torch.randn(1, 10, 64).cuda()

        # 普通推理
        with torch.no_grad():
            out_normal = model(x)

        # CUDA Graph 推理
        runner = CUDAGraphRunner(model, x)
        out_graph = runner.run(x)

        assert torch.allclose(out_normal, out_graph, atol=1e-5), "CUDA Graph 结果应一致"
        print("✅ CUDA Graph 与普通推理结果一致")