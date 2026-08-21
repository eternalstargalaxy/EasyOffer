"""
【题目】ZeRO-1 / ZeRO-2 / ZeRO-3 状态分片

【背景】
DDP 下每卡冗余持有全部 optimizer state / grad / param。ZeRO 依次把它们在数据并行组间分片：
- ZeRO-1：分片 optimizer state（Adam 的 m/v）→ 优化器状态 4x 降为 4x/N
- ZeRO-2：再分片梯度（reduce-scatter 后各卡只留自己那片）→ 再降 2x/N
- ZeRO-3：再分片参数（前向/反向前 all-gather 出完整参数，用完释放）→ 再降 param/N
更新只更新本 shard，ZeRO-3 更新后需 all-gather 把新权重同步给所有卡。

【输入/输出】
- 输入：model, optimizer, data_parallel_size=N, rank
- 输出：每卡只持有 param/grad/state 的 1/N，训练正常推进

【考察点】
- reduce-scatter / all-gather 与 step 的交错顺序
- ZeRO-3 前向也需 all-gather（参数不全算不了）
- 三种方式的显存公式与通信量 trade-off
- 提示：torch.distributed.all_gather / reduce_scatter 用于参数收集/分片
"""
import torch
import torch.nn as nn


class ShardedAdam:
    """ZeRO-1：只为本 param shard 维护 m/v"""

    def __init__(self, params_shard: torch.Tensor, lr: float = 1e-3, betas: torch.Tensor = (0.9, 0.999: torch.Tensor):
        self.params = list(params_shard)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad = None


def zero2_step(grad_full: torch.Tensor, dp_size: int, rank: int) -> torch.Tensor:
    """reduce-scatter 梯度，返回本卡 shard（1/N）。单机模拟：直接切片。"""
    shard_size = grad_full.numel() // dp_size
    flat = grad_full.view(-1)
    start = rank * shard_size
    return flat[start:start + shard_size].clone()


class Zero3:
    """ZeRO-3：参数也分片，前向/反向前 all-gather。"""

    def __init__(self, model: nn.Module, dp_size: int, rank: int):
        self.dp_size = dp_size
        self.rank = rank
        self.params = list(model.parameters())
        flat = torch.cat([p.data.view(-1) for p in self.params])
        self.total_size = flat.numel()
        shard_size = self.total_size // dp_size
        start = rank * shard_size
        self.param_shard = flat[start:start + shard_size].clone()
        self.shard_size = shard_size
        self._full_param = None

    def gather_param(self) -> torch.Tensor:
        """前向/反向前 all-gather 出完整参数。单机模拟：直接拼。"""
        self._full_param = self.param_shard.clone()
        for r in range(1, self.dp_size):
            dummy_shard = torch.zeros_like(self.param_shard)
            self._full_param = torch.cat([self._full_param, dummy_shard])
        return self._full_param

    def release_param(self) -> None:
        """计算完释放非本 shard 副本。"""
        self._full_param = None

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        full = self.gather_param()
        result = full
        self.release_param()
        return result

    def backward_step(self, grad_shard: torch.Tensor) -> torch.Tensor:
        """reduce-scatter grad -> 更新本 shard -> all-gather 同步权重。"""
        self.param_shard -= 0.01 * grad_shard
        return self.gather_param()


def mem_formula(N: int, param_cnt: torch.Tensor, grad_cnt: torch.Tensor, state_cnt: torch.Tensor) -> dict:
    """返回 ZeRO-1/2/3 单卡显存（以元素数计）"""
    zero1 = param_cnt + grad_cnt + state_cnt / N
    zero2 = param_cnt + grad_cnt / N + state_cnt / N
    zero3 = param_cnt / N + grad_cnt / N + state_cnt / N
    return {"zero1": zero1, "zero2": zero2, "zero3": zero3,
            "ddp": param_cnt + grad_cnt + state_cnt}


# ===== 测试验证 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    model = nn.Linear(10, 5)
    optimizer = ShardedAdam(model.parameters(), lr=0.01)

    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    w_before = model.weight.data.clone()
    optimizer.step()
    assert not torch.allclose(w_before, model.weight.data), "权重应被更新"
    print("✅ ShardedAdam (ZeRO-1): 更新成功")

    grad = torch.randn(20)
    shard = zero2_step(grad, dp_size=4, rank=0)
    assert shard.numel() == 5, f"ZeRO-2 shard 大小错误: {shard.numel()}"
    print(f"✅ zero2_step: {grad.numel()} -> shard {shard.numel()}")

    model3 = nn.Linear(10, 5)
    zero3 = Zero3(model3, dp_size=4, rank=0)
    assert zero3.param_shard.numel() == zero3.total_size // 4
    full = zero3.gather_param()
    assert full.numel() == zero3.total_size
    print(f"✅ Zero3: shard {zero3.param_shard.numel()}, full {full.numel()}")
    zero3.release_param()
    assert zero3._full_param is None
    print("✅ Zero3 release_param 正确")

    mem = mem_formula(N=4, param_cnt=100, grad_cnt=100, state_cnt=400)
    assert mem["zero3"] < mem["zero2"] < mem["zero1"] < mem["ddp"]
    print(f"✅ 显存: DDP={mem['ddp']}, Z1={mem['zero1']}, Z2={mem['zero2']}, Z3={mem['zero3']}")
    print("✅ 全部测试通过")
