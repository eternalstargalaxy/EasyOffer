"""
【题目】LoRA 多适配器推理

【背景】
同时服务多个 LoRA 适配器时，为每个适配器单独加载 base model 浪费显存。
Multi-LoRA serving 共享 base model，只加载不同 LoRA adapter（A/B 矩阵），
推理时动态切换或 batch 混合不同 adapter。优势：显存 O(base + n*adapter) vs O(n*base)。

【输入/输出】
- 输入：base_model, adapters={name: (A, B)}, 请求指定 adapter
- 输出：用对应 adapter 推理的结果

【考察点】
- adapter 切换开销与 batch 混合
- LoRA 前向：W*x + B*A*x
- 提示：nn.ModuleDict 管理多 adapter
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8):
        super().__init__()
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor):
        return self.B(self.A(x))


class MultiLoRALinear(nn.Module):
    """共享 base weight + 多 LoRA adapter。"""

    def __init__(self, in_dim: int, out_dim: int, adapter_names: torch.Tensor, rank: int = 8):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim)
        self.adapters = nn.ModuleDict({
            name: LoRAAdapter(in_dim, out_dim, rank) for name in adapter_names
        })

    def forward(self, x: torch.Tensor, adapter_name: str = None):
        out = self.base(x)
        if adapter_name and adapter_name in self.adapters:
            out = out + self.adapters[adapter_name](x)
        return out

    def forward_batch(self, x_batch: torch.Tensor, adapter_names: torch.Tensor):
        """batch 混合：不同样本用不同 adapter。"""
        outputs = []
        for x, name in zip(x_batch, adapter_names):
            out = self.base(x.unsqueeze(0))
            if name in self.adapters:
                out = out + self.adapters[name](x.unsqueeze(0))
            outputs.append(out.squeeze(0))
        return torch.stack(outputs)


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    in_d, out_d = 64, 32
    layer = MultiLoRALinear(in_d, out_d, ["task_a", "task_b"], rank=4)

    x = torch.randn(4, in_d)
    y_base = layer(x)
    assert y_base.shape == (4, out_d)
    print("✅ Base forward (无 adapter)")

    y_a = layer(x, adapter_name="task_a")
    assert y_a.shape == (4, out_d)
    print("✅ LoRA forward (task_a)")

    y_b = layer(x, adapter_name="task_b")
    assert y_b.shape == (4, out_d)
    print("✅ LoRA forward (task_b)")

    x_batch = [torch.randn(in_d) for _ in range(3)]
    names = ["task_a", "task_b", "task_a"]
    y_mix = layer.forward_batch(x_batch, names)
    assert y_mix.shape == (3, out_d)
    print("✅ Batch 混合多 adapter")

    layer.adapters["task_a"].B.weight.data = torch.randn_like(layer.adapters["task_a"].B.weight)
    y_a2 = layer(x, adapter_name="task_a")
    assert not torch.allclose(y_a, y_a2), "更新 B 后输出应变化"
    print("✅ Adapter 参数更新生效")
    print("✅ 全部测试通过")
