# GPU 拷贝带宽四向实测（Week 2 Day 3）

- 日期：2026-09-21
- 环境：keg211 共享机，RTX 2080 Ti（PCIe gen3 x16），单卡独占时段，torch 2.x
- 脚本：`scripts/exp_gpu_copy.py`
- 状态：**已结案**（2026-09-23 追加验证，见第 7 节）

---

## 1. 问题

数据在 host 内存 ↔ GPU 显存之间流动，各方向各有多快？
理论分层的天花板（PCIe 链路、显存带宽、缺页、pageable 中转）在实测中各占多少？

## 2. 方法

- 400 MB 张量（100M float32），warm-up 后 min-of-5 计时
- 六个测量：pageable H2D / pinned H2D / D2H 新分配 / D2H 预分配 / D2H pinned 目标 / D2D
- D2D 注意：`.to()` 同设备返回自己，必须用 `clone()` 产生真拷贝
- D2H 三组对照的设计意图：区分"传输的钱"和"分配（缺页）的钱"

## 3. 理论值

| 层 | 理论 | 备注 |
|---|---|---|
| PCIe gen1 x16 | ~4 GB/s | 空闲链路已确证停在此档 |
| PCIe gen2 x16 | ~8 GB/s | |
| PCIe gen3 x16 | ~15.8 GB/s | 2080 Ti 能力上限 |
| HBM2（D2D） | ~616 GB/s | 2080 Ti 显存带宽 |
| pageable H2D | 低于链路 | 驱动需先 CPU 中转拷到 staging buffer，再 DMA |

## 4. 实测结果

| 测量 | 数值 | 解读 |
|---|---|---|
| pageable H2D | 7.4 GB/s | |
| pinned H2D | 7.8 GB/s | ≈ pageable → **staging 中转假设被证伪**（开销仅 ~5%） |
| D2H（`xg.to('cpu')`，每次新分配） | 2.1 GB/s | 慢的主要是缺页中断，不是传输 |
| D2H（预分配 pageable 目标） | 8.3 GB/s | 去掉分配成本后，4 倍差 |
| D2H（pinned 目标） | 8.6 GB/s | |
| D2D（clone，读+写双向账） | 524 GB/s | 达 HBM 理论 85%，健康 |
| 首算子冷启动 | ~0.30 s | lazy init 三层账：context 创建 + cubin 模块上传（主体）+ 算子首开销 |

## 5. 推理链（四步法实录）

1. **量级对齐**：实测 ~7.4-8.6 GB/s，明显低于 gen3 理论 15.8 → 有异常
2. **分层枚举假设**：x8 宽度 / gen2 降档 / pageable staging / copy engine 上限
3. **证伪**：
   - pageable staging：pinned 7.8 ≈ pageable 7.4 → 排除为主因
   - 拟合陷阱：x8 和 gen2 都能拟合 7.8 这个数——**拟合好 ≠ 证明**，只有直接观测能裁决
   - 首次链路查询因"采样未与负载同时"无效（仪器问题，非数据）
4. **收敛**：判定链路 gen3 x16 健康；**~8 GB/s 有效天花板归因到 copy engine / 平台 DMA 路径**（开放问题：精确归因）

## 6. 关键概念（当时纠正过的）

- **DMA**：数据绕开 CPU，控制必经 CPU；pageable 拷贝 = CPU 中转 + DMA 接力
- **锁页保的是"内容所有权"**：swap 换走的是内容，物理地址号码从不变
- **缺页的钱可能比传输贵 4 倍**：计时必须声明是否包含分配成本
- **车道 × 车速 × 货**：PCIe 链路（车道）× copy engine（车速）× 缺页/中转（货）缺一不可

## 7. 结案更新（2026-09-23，Week 2 Day 5）

**实验**：H2D 持续负载（pinned，256 MB × 20 连续拷贝 60 s）+ `nvidia-smi --query-gpu=pcie.link.gen.current` 200 ms 轮询。

**结果**（`logs/w2d5/pcie_gen_poll.csv`）：

- 空闲：gen1 → 负载开始后 ~1 s 内升 gen3 → 全程 41 s 稳定 gen3 x16 → 负载结束后逐级降回 gen2 → gen1
- **负载下链路 settling 到 gen3，无降档**

**结论**："8 GB/s 是因为链路只协商到 gen2"假设**证伪**。
~8 GB/s 天花板与链路协商无关，维持原归因（copy engine / 平台 DMA 路径），**开放问题结案**。
副产品：settling 全过程（升档快、降档逐级）首次获得实测时间线。

**工具备注**：`lspci -vv` 需 root（`Capabilities: &lt;access denied&gt;`），链路状态用
`nvidia-smi --query-gpu=pcie.link.gen.*` 绕过。

## 8. 遗留教训（写给未来的自己）

- 采样必须与负载同时（第一次链路查询无效）
- 命令参数被误吞（grep `-g`、sed 行号当文件名）——**先读报错第一行**
- D2D 带宽账漏乘 2（读+写双向流量）
- 共享机环境漂移是常态：未复现的异常标记存疑，不强行解释