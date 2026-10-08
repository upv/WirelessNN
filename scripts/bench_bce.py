"""Benchmark complex_bce_batched: mixed-modulation batch, forward + backward on the GPU."""
import sys, time, torch
from nr_ul_sim.losses import MODULATIONS, complex_bce_batched, constellation

dev = "cuda"
B, N = int(sys.argv[1]) if len(sys.argv) > 1 else 32, 14 * 816      # records x symbols per record (one 68-PRB slot)
g = torch.Generator().manual_seed(0)
mods = [MODULATIONS[i % 3] for i in range(B)]
x = torch.stack([constellation(m)[0][torch.randint(len(constellation(m)[0]), (N,), generator=g)] for m in mods]).to(dev)
x_hat = (x + 0.2 * torch.randn(x.shape, dtype=x.dtype, generator=g).to(dev)).requires_grad_()
nv = torch.full((B,), 0.1, device=dev)

def step():
    loss = complex_bce_batched(x_hat, x, mods, noise_var=nv)
    loss.backward()
    return loss

for _ in range(3): step(); x_hat.grad = None
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter(); iters = 10
for _ in range(iters):
    loss = step(); x_hat.grad = None
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / iters
print(f"B={B} N={N} symbols={B*N:,}  loss={loss.item():.5f}  {dt*1e3:.1f} ms/iter (fwd+bwd)  "
      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB  {B*N/dt/1e6:.1f} Msym/s")
