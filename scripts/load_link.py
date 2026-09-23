# load_link.py
import torch, time
dev = 'cuda:0'
x = torch.randn(100_000_000, pin_memory=True)
xg = torch.randn(100_000_000, device=dev)
t_end = time.time() + 30
while time.time() < t_end:
    _ = x.to(dev, non_blocking=True)
    _ = xg.to('cpu', non_blocking=True)
torch.cuda.synchronize()
print("done")