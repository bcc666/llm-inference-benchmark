import time, torch

dev = torch.device('cuda:0')     # 进程只见卡1，所以 cuda:0 就是物理卡1
x = torch.randn(100_000_000)     # 400MB，别加大——训练还占着卡0

# ① 冷启动：第一笔三层账（context + lazy kernel loading + 算子）
t = time.time()
_ = torch.randn(10, device=dev)
torch.cuda.synchronize()
print("首次 CUDA 算子总耗时:", time.time() - t)   # 预测一下量级

# warm-up 后再计时
_ = x[:1000].to(dev)
torch.cuda.synchronize()

def bench(fn):
    torch.cuda.synchronize(); t = time.time()
    fn(); torch.cuda.synchronize(); return time.time() - t

# ② pageable H2D
t_page = min(bench(lambda: x.to(dev)) for _ in range(5))
# ③ pinned H2D（异步）
xp = x.pin_memory()
t_pin = min(bench(lambda: xp.to(dev, non_blocking=True)) for _ in range(5))
# ④ D2H（同步）
xg = x.to(dev)
t_d2h = min(bench(lambda: xg.to('cpu')) for _ in range(5))
# ⑤ D2D 真拷贝（注意：.to 同设备返回自己，要用 clone）
t_d2d = min(bench(lambda: xg.clone()) for _ in range(5))

print(f"pageable H2D: {t_page*1e3:.1f} ms  ({0.4/t_page:.1f} GB/s)")
print(f"pinned   H2D: {t_pin*1e3:.1f} ms  ({0.4/t_pin:.1f} GB/s)")
print(f"D2H         : {t_d2h*1e3:.1f} ms  ({0.4/t_d2h:.1f} GB/s)")
print(f"D2D (clone) : {t_d2d*1e3:.1f} ms  ({0.8/t_d2d:.1f} GB/s)")

# 接在原脚本后面，xg 已在 GPU 上
dest = torch.empty(100_000_000)                    # 预分配 pageable 目标
t_pre  = min(bench(lambda: dest.copy_(xg)) for _ in range(5))
destp = torch.empty(100_000_000, pin_memory=True)  # pinned 目标
t_prep = min(bench(lambda: destp.copy_(xg, non_blocking=True)) for _ in range(5))
print(f"D2H pageable 预分配: {t_pre*1e3:.1f} ms ({0.4/t_pre:.1f} GB/s)")
print(f"D2H pinned 目标    : {t_prep*1e3:.1f} ms ({0.4/t_prep:.1f} GB/s)")