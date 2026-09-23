# DataLoader 六组对照 + PCIe 链路 settling 闭环（Week 2 Day 5）

- 日期：2026-09-23
- 环境：keg211，RTX 2080 Ti ×2（用卡 0），torch，125GB RAM
- 脚本：`scripts/dl_bench.py`；产物：`results/w2d5_dataloader_summary.csv`、`results/gpu_util/*.csv`、`logs/w2d5/pcie_gen_poll.csv`
- 状态：**两项均已结案**（DataLoader 六组结论 + 8 GB/s 天花板与 gen2 关系证伪）

---

## 实验一：DataLoader 六组对照

### 设计

人工 dataset（10000×3×224×224 一次性预生成 6.02 GB，seed 固定，`__getitem__` 只剩索引）+ 小 CNN（3 conv + GAP + fc，10 类）。
对照 pin_memory={False,True} × num_workers={0,2,4}，每组 warmup 1 epoch + 计时 2 epoch，batch=64。
并行采集：nvidia-smi 100ms 间隔采样 GPU 利用率（独立子进程，含 mean/p10/p90）。

### 实测（每 epoch wall time，2 epoch 均值）

| config | wall time | img/s | gpu_util mean/p10/p90 |
| --- | --- | --- | --- |
| pin0_w0 | 7.53 s | 1328 | 62.1 / 63 / 66 |
| pin0_w2 | 8.06 s | 1241 | 57.8 / 59 / 62 |
| pin0_w4 | 8.05 s | 1242 | 57.8 / 58 / 62 |
| **pin1_w0（最优）** | **7.05 s** | **1420** | 66.3 / 68 / 70 |
| pin1_w2 | 7.11 s | 1407 | 65.4 / 65 / 70 |
| pin1_w4 | 7.10 s | 1408 | 65.1 / 65 / 70 |

### 结论

1. **pin_memory=True 稳定收益 ~5%**（7.53→7.05 s）。机制：`pin=True` 并未消灭 host 内拷贝，是把驱动的同步 staging 换成 DataLoader pinned 池的异步预拷；它只优化了每 step 48ms 里 ~5ms 的那一段。
2. **num_workers 在本 workload 是负优化**（数据已在 RAM、getitem 免费，worker 只增加 fork/IPC 搬运）。workers 是为"取数贵"（读盘、解码、增强）场景准备的——有价值的 negative result。
3. **利用率全部稳定 62~66%，p10/p90 差 <7 点** → 固定环节瓶颈，非抖动喂不满。

### 每 step 48.3ms 的分解（利用率为锚倒推 + 带宽估算）

| 段 | 时长 | 依据 |
| --- | --- | --- |
| GPU 计算（SM 忙） | ~30 ms | 7.53s × 62.1% ÷ 156 step |
| H2D 拷贝（SM 闲） | ~5 ms | 38.5MB/batch ÷ ~8 GB/s |
| host 空隙（SM 闲） | ~13 ms | 余项（Python/collate/launch/同步） |

一致性检验：30/48.3 = 62.1%，与实测 gpu_util_mean 完全吻合——**utilization.gpu 数的是 SM 活跃占比，不是"GPU 有多努力"**。
分解中两段为推算，待微实验坐实（数据预放 GPU 只算 / 只拷贝不训练）。

### 工程记录（脚本踩坑三连）

1. `global TIMED_EPOCHS` 声明在使用之后 → SyntaxError；global 必须放在函数内所有使用之前
2. 打印累计 GB/s 时总字节写死（只算了第一轮迭代量），数值随 elapsed 衰减到 0.1——**"看着像硬件降速"的现象，第一反应查测量代码**
3. 6 组跑完后进程卡在退出阶段：DataLoader worker/pin_memory 线程 join 死等（已知怪癖）→ benchmark 脚本用 `sys.stdout.flush(); os._exit(0)` 跳过 teardown

## 实验二：PCIe 负载下 gen settling（2 分钟，闭环 Day 3 天花板）

### 设计与结果

pinned H2D 持续负载（256MB × 20 连续拷贝 60s）+ `nvidia-smi --query-gpu=pcie.link.gen.current` 200ms 轮询（lspci -vv 需 root，用 Day 4 的绕过方案）。

- 空闲 gen1 → 负载后 ~1s 内升 **gen3** → 全程 41s 稳定 gen3 x16 → 负载结束逐级降回 gen2 → gen1
- **负载下无降档**

### 结论

"~8 GB/s 天花板是因为链路只协商到 gen2"**证伪**（7.8 ≈ gen2 理论纯属数字巧合）。
天花板归因维持 Day 3 原判（copy engine / 平台 DMA 路径），**开放问题结案**。
副产品：settling 全过程（升档快、降档逐级）首次获得实测时间线。

## 项目整理（当日工程）

目录重构：`scripts/`（全部 py）、`logs/`（按实验分子目录）、`notes/`、`results/`。
.gitignore 修正：`logs/` 排除；`results/` 整体排除的写法会让 `!results/*.json` 失效，改为 `results/*` + `!results/*.json`。

## 关键认识

1. "速度"是分层的数：D2D ~524 GB/s / PCIe gen3 x16 ~15.8 链路 / pinned H2D ~7.8 应用层 / 训练循环 ~0.86 GB/s——四个数不矛盾，各描述不同环节
2. 训练慢 ≠ 拷贝慢：6GB/epoch 拷贝只需 ~0.75s，7.5s 的大头是计算 + host 空隙
3. workers/pin 的调优空间由 workload 决定：数据已在 RAM 时，换更大的模型比调 DataLoader 参数有效得多
4. 跨天知识必须落笔记：lspci 需 root 的解法 Day 4 破过案，Day 5 又踩——笔记不是文书，是防重复踩坑的保险