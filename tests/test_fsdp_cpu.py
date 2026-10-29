'''
CPU check of the FSDP2 path in scripts/train_dvs.py: 2 gloo processes vs. an unsharded
single-process reference on a toy model shaped like the DiT (a .blocks list of activation-
checkpointed blocks, plus a root parameter that never receives a gradient).

    torchrun --nproc_per_node=2 tests/test_fsdp_cpu.py
'''
import copy
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
import train_dvs  # noqa: E402


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(8, 32)
        self.blocks = nn.ModuleList(
            [checkpoint_wrapper(nn.Sequential(nn.Linear(32, 32), nn.GELU())) for _ in range(3)])
        self.unused = nn.Linear(32, 32)  # like the view-0 timestep embedder
        self.out = nn.Linear(32, 4)

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = x + b(x)
        return self.out(x)


def step(model, x, y):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1,
                            fused=True)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    grads = {n: p.grad.full_tensor() if hasattr(p.grad, 'full_tensor') else p.grad
             for (n, p) in model.named_parameters() if p.grad is not None}
    opt.step()
    return grads


def main():
    dist.init_process_group('gloo')
    (rank, world) = (dist.get_rank(), dist.get_world_size())
    torch.manual_seed(0)
    ref = Toy()
    model = copy.deepcopy(ref)
    x = torch.randn(world, 5, 8)
    y = torch.randn(world, 5, 4)

    # gloo implements FSDP2's ReduceOp.AVG as a sum (NCCL averages), so on gloo the reference
    # uses the summed loss; with NCCL it is the mean, which is what DDP averages to.
    scale = 1.0 if dist.get_backend() == 'gloo' else 1.0 / world
    loss = sum(((ref(x[r]) - y[r]) ** 2).mean() for r in range(world)) * scale
    loss.backward()
    ref_grads = {n: p.grad.clone() for (n, p) in ref.named_parameters() if p.grad is not None}
    torch.optim.AdamW(ref.parameters(), lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1,
                      fused=True).step()

    train_dvs.apply_fsdp(model, 'fp32')
    grads = step(model, x[rank], y[rank])
    worst_g = max(((grads[n] - ref_grads[n]).abs().max() / ref_grads[n].abs().max()).item()
                  for n in ref_grads)
    assert set(grads) == set(ref_grads), (sorted(grads), sorted(ref_grads))

    state = train_dvs.full_state_dict(model)
    if rank == 0:
        ref_state = ref.state_dict()
        assert list(state) == list(ref_state), 'checkpoint keys differ'
        diffs = {k: (state[k] - ref_state[k]).abs().max().item() for k in ref_state}
        worst_w = max(diffs.values())
        assert worst_w < 1e-5, f'weights after one AdamW step differ by {worst_w:.1e}'
        print(f'grad rel err {worst_g:.1e}; weights after AdamW step max diff {worst_w:.1e}; keys match ({len(state)}); unused param got no grad: '
              f'{"unused.weight" not in grads}')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
