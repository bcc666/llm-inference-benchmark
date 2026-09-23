"""
stream_overlap_benchmark.py
===========================
Week 2 Day 4 (2026-09-22) 全部实验归档：CUDA Stream 与拷贝/计算重叠
硬件：RTX 2080 Ti（实验卡 = 物理卡 1，CUDA_VISIBLE_DEVICES=1）
运行：CUDA_VISIBLE_DEVICES=1 python stream_overlap_benchmark.py

每个 section 的预期值来自 2026-09-22 实测，偏差 >20% 请按四步法排查
（量级对齐 -> 分层枚举 -> 假设配证伪实验 -> 同时性确认）。
"""

import torch
import time
import subprocess

assert torch.cuda.is_available()
DEV = 'cuda'
print(f"torch {torch.__version__} | device: {torch.cuda.get_device_name(0)}")
print(f"uuid: {torch.cuda.get_device_properties(0).uuid}")
print(f"env: CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}")


def clocks(tag):
    """采样两卡的 SM/显存时钟与电源状态。nvidia-smi 报告所有物理卡，一行一张。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,pstate",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"[{tag}] {out}")


def timed(fn):
    """event 计时（GPU 时间轴）。要求计时对象在默认流上。"""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


# ----------------------------------------------------------------------
x1 = torch.randn(100_000_000, device=DEV)   # ~400MB
x2 = torch.empty_like(x1)
x3 = torch.empty_like(x1)
a = torch.randn(4096, 4096, device=DEV)
b = torch.randn(4096, 4096, device=DEV)
c = torch.empty(4096, 4096, device=DEV)     # 预分配输出（section 5 用）

# warmup：每个 op 先跑 10 遍，避免冷时钟/首次开销污染测量
for _ in range(10):
    x2.copy_(x1)
    torch.mm(a, b, out=c)

# ======================================================================
# S1. 锚点：copy ≈ 1.5ms（800MB 流量 / 524 GB/s ≈ 1.5，注意读写乘 2）
#           matmul ≈ 11~13ms（2*4096^3 FLOP / 13.4 TFLOP/s ≈ 10.3 理论下界）
# ======================================================================
print("\n=== S1 anchors ===")
print(f"copy:   {timed(lambda: x2.copy_(x1)):.3f} ms  (expect ~1.5)")
print(f"matmul: {timed(lambda: torch.mm(a, b, out=c)):.3f} ms  (expect 11~13)")

# ======================================================================
# S2. 单队列串行：5 op 交替排默认流，queued total ≈ sum(items)
# 结论：同一条流上 copy(copy engine) 与 matmul(SM) 虽用不同硬件，
#       但队列语义强制按提交顺序执行，无重叠。
# ======================================================================
print("\n=== S2 single-queue serial ===")
ops = [("copy1",   lambda: x2.copy_(x1)),
       ("matmul1", lambda: torch.mm(a, b, out=c)),
       ("copy2",   lambda: x3.copy_(x2)),
       ("matmul2", lambda: torch.mm(b, a, out=c)),
       ("copy3",   lambda: x1.copy_(x3))]
for _ in range(5):
    for _, f in ops:
        f()
items = [timed(f) for _, f in ops]
for (n, _), t in zip(ops, items):
    print(f"{n}: {t:.3f} ms")
ts = torch.cuda.Event(enable_timing=True)
te = torch.cuda.Event(enable_timing=True)
ts.record()
for _, f in ops:
    f()
te.record()
torch.cuda.synchronize()
print(f"sum(items) = {sum(items):.3f} | queued = {ts.elapsed_time(te):.3f}  (expect 差 <2%)")

# ======================================================================
# S3. 双显式流重叠：s1 跑 9 copy，s2 跑 1 matmul（时长故意调到同量级）
# 结论：
#  (a) 两条流真重叠（时间窗有交集，offset ≈ 0）
#  (b) 但 copy 被拖慢 1.5 -> ~2.4ms/个：copy engine 与 SM 共享显存带宽，
#      带宽竞争下"搬密集"的 copy 受伤远重于"算密集"的 matmul
#  (c) 实测加速比仅 ~3%（22.9 -> 22.2），远低于理想的 41%（13.5/22.9）
# 陷阱备忘：total event 若 record 在默认流上，测的是 CPU 提交时间
#      （实测 0.339ms）——异步语义下"提交"≠"完成"，终点必须由各流的
#      end event 或 torch.cuda.synchronize() 收口。
# ======================================================================
print("\n=== S3 two-stream overlap ===")
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
s1_s = torch.cuda.Event(enable_timing=True); s1_e = torch.cuda.Event(enable_timing=True)
s2_s = torch.cuda.Event(enable_timing=True); s2_e = torch.cuda.Event(enable_timing=True)


def overlap_once():
    with torch.cuda.stream(s1):
        s1_s.record()
        for _ in range(9):
            x2.copy_(x1)
        s1_e.record()
    with torch.cuda.stream(s2):
        s2_s.record()
        torch.mm(a, b, out=c)
        s2_e.record()


# 串行对照（全在默认流）
for _ in range(5):
    for _ in range(9):
        x2.copy_(x1)
    torch.mm(a, b, out=c)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(9):
    x2.copy_(x1)
torch.mm(a, b, out=c)
torch.cuda.synchronize()
serial = (time.time() - t0) * 1000
print(f"serial total: {serial:.2f} ms  (expect ~23)")

for _ in range(5):
    overlap_once()
torch.cuda.synchronize()
t0 = time.time()
overlap_once()
torch.cuda.synchronize()   # 墙钟 + synchronize 天然覆盖两条流
overlap_total = (time.time() - t0) * 1000
print(f"overlap total: {overlap_total:.2f} ms  (理想下界 ~13.5)")

d_copy = s1_s.elapsed_time(s1_e)
d_mm = s2_s.elapsed_time(s2_e)
offset = s1_s.elapsed_time(s2_s)
print(f"copy   window: [0, {d_copy:.2f}]")
print(f"matmul window: [{offset:.2f}, {offset + d_mm:.2f}]")
print(f"overlap length: {min(d_copy - offset, d_mm):.2f} ms")
print(f"speedup: {(serial - overlap_total) / serial * 100:.1f}%  (expect ~3%)")

# ======================================================================
# S4. 时钟采样：空闲 P8（SM 300/显存 405 MHz）-> 负载 P2 升档
# 观察：显存时钟二元跳变（405 -> 6800），SM 时钟连续爬坡（1350->1890）
# ======================================================================
print("\n=== S4 clock ramping ===")
clocks("idle-before")
t0 = time.time()
for i in range(100):
    x2.copy_(x1)
    if i == 30:
        clocks("copy@30"); torch.cuda.synchronize()
        print(f"  per-copy: {(time.time() - t0) / (i + 1) * 1000:.2f} ms")
torch.cuda.synchronize()
clocks("copy@100-end")
for _ in range(10):
    torch.mm(a, b, out=c)
torch.cuda.synchronize()
clocks("after-matmul")

# ======================================================================
# S5. 隐藏同步点：输出张量分配在 GPU 繁忙时阻塞 CPU
# 现象（2026-09-22 实测）：9 copy 排队后调 a@b（新建 64MB 输出），
#   CPU 在调用内部被堵 ~29.6ms（cudaMalloc 隐式同步，等 copy 清空）
#   GPU matmul 窗口被拖成 29ms；预分配 out= 后 CPU 提交 0.17ms、GPU 12ms
# 结论：算子的隐藏成本不只在 GPU 侧——任何 CPU 等待 GPU 的同步点都要警惕
# ======================================================================
print("\n=== S5 allocation sync point ===")
torch.cuda.empty_cache()   # 强制下一次分配走真实 cudaMalloc 路径（保证复现）
torch.cuda.synchronize()
for _ in range(9):         # 制造"GPU 繁忙"：9 个 copy 排队执行
    x2.copy_(x1)
t0 = time.time()
d = a @ b                  # 新建输出 -> allocator -> cudaMalloc 隐式同步
cpu_block = (time.time() - t0) * 1000
torch.cuda.synchronize()
print(f"a@b with fresh alloc: CPU blocked {cpu_block:.2f} ms  (expect ~29)")

torch.cuda.synchronize()
t0 = time.time()
torch.mm(a, b, out=c)      # 预分配输出，allocator 不参与
cpu_dt = (time.time() - t0) * 1000
torch.cuda.synchronize()
print(f"mm(out=c):            CPU submit {cpu_dt:.2f} ms  (expect ~0.2)")

print("\nDone. 数据来源：keg211 2080 Ti, 2026-09-22, torch 2.5.1+cu124")