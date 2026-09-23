#!/usr/bin/env python
"""
dl_bench.py — DataLoader 对照实验 (Week 2 Day 5)

对照: pin_memory={False,True} x num_workers={0,2,4} 共 6 组
每组: warmup 1 epoch (不计时) + 计时 2 epoch, 取均值
并行: 后台 nvidia-smi 采样 GPU 利用率 (>=100ms 间隔)
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------- 全局配置 ----------------
N_SAMPLES = 10000
BATCH_SIZE = 64
TIMED_EPOCHS = 2
WARMUP_EPOCHS = 1
GPU_ID = 0                      # nvidia-smi 显示双卡全空闲, 用 0 号
SAMPLE_INTERVAL_MS = 100
RESULTS_DIR = Path("results")
UTIL_DIR = RESULTS_DIR / "gpu_util"

CONFIGS = [
    {"pin": False, "workers": 0},
    {"pin": False, "workers": 2},
    {"pin": False, "workers": 4},
    {"pin": True,  "workers": 0},
    {"pin": True,  "workers": 2},
    {"pin": True,  "workers": 4},
]


# ---------------- 数据: 一次性预生成, getitem 只剩索引 ----------------
class FakeImageDataset(Dataset):
    def __init__(self, n, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randn(n, 3, 224, 224, generator=g)
        self.labels = torch.randint(0, 10, (n,), generator=g)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ---------------- 小 CNN ----------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 224->112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 112->56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 56->28
            nn.AdaptiveAvgPool2d(1),                                      # 128x1x1
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------- GPU 利用率采样: 独立子进程 ----------------
class GpuUtilSampler:
    """后台运行 nvidia-smi -lms, 输出逐行解析为 util 序列"""

    def __init__(self, gpu_id, interval_ms=SAMPLE_INTERVAL_MS):
        self.cmd = [
            "nvidia-smi",
            f"--query-gpu=utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
            "-lms", str(interval_ms),
            "-i", str(gpu_id),
        ]
        self.proc = None
        self.samples = []  # (epoch_ms, gpu_util, mem_util)

    def start(self):
        self.proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            for line in self.proc.stdout:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) == 2 and parts[0].isdigit():
                    self.samples.append((int(parts[0]), int(parts[1])))
            self.proc = None
        return self.samples


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


# ---------------- 训练循环 ----------------
def run_one_config(model, dataset, device, cfg):
    """返回 (每 epoch wall time 列表, util 样本列表)"""
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=cfg["workers"],
        pin_memory=cfg["pin"],
        persistent_workers=cfg["workers"] > 0,   # 计时 epoch 之间不重建 worker
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()

    sampler = GpuUtilSampler(GPU_ID)
    wall_times = []
    sampler.start()
    try:
        for epoch in range(WARMUP_EPOCHS + TIMED_EPOCHS):
            model.train()
            t0 = time.perf_counter()
            for x, y in loader:
                x = x.to(device, non_blocking=cfg["pin"])   # pin 配合 non_blocking 才有意义
                y = y.to(device, non_blocking=cfg["pin"])
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
            torch.cuda.synchronize(device)
            dt = time.perf_counter() - t0
            if epoch >= WARMUP_EPOCHS:
                wall_times.append(dt)
                print(f"  [timed] epoch {epoch}: {dt:.3f}s "
                      f"({len(dataset)/dt:.0f} img/s)")
    finally:
        samples = sampler.stop()

    del loader
    torch.cuda.empty_cache()
    time.sleep(10)   # 组间冷却, 避免热状态串组
    return wall_times, samples


def main():
    global TIMED_EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_SAMPLES,
                        help="样本数, 快速自检用 --n 500")
    parser.add_argument("--epochs", type=int, default=TIMED_EPOCHS)
    args = parser.parse_args()

    TIMED_EPOCHS = args.epochs

    RESULTS_DIR.mkdir(exist_ok=True)
    UTIL_DIR.mkdir(exist_ok=True)

    device = torch.device(f"cuda:{GPU_ID}")
    torch.cuda.set_device(device)
    torch.manual_seed(42)

    print(f"[setup] 预生成 {args.n} 张假数据 ...")
    t0 = time.time()
    dataset = FakeImageDataset(args.n)
    print(f"[setup] 完成, 耗时 {time.time()-t0:.1f}s, "
          f"数据 {dataset.data.numel()*4/1e9:.2f} GB")

    rows = []
    for i, cfg in enumerate(CONFIGS):
        tag = f"pin{int(cfg['pin'])}_w{cfg['workers']}"
        print(f"\n[{i+1}/6] config={tag}")
        model = SmallCNN().to(device)          # 每组重建模型, 排除状态残留
        wall_times, samples = run_one_config(model, dataset, device, cfg)

        utils = sorted(u for _, u in samples)
        util_csv = UTIL_DIR / f"gpu_util_{tag}.csv"
        with open(util_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gpu_util", "mem_util"])
            w.writerows(samples)

        for ep, wt in enumerate(wall_times, start=1):
            rows.append({
                "config": tag,
                "epoch": ep,
                "wall_time_s": round(wt, 3),
                "imgs_per_s": round(args.n / wt, 1),
                "gpu_util_mean": round(sum(utils)/len(utils), 1) if utils else "nan",
                "gpu_util_p10": round(percentile(utils, 10), 1) if utils else "nan",
                "gpu_util_p90": round(percentile(utils, 90), 1) if utils else "nan",
                "n_samples": len(samples),
            })
        # 增量写结果, 中途崩了也保留已跑组
        with open(RESULTS_DIR / "w2d5_dataloader_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n===== 汇总 =====")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()