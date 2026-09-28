import csv
import matplotlib.pyplot as plt

# ---------- 读数据 ----------
rows = []
with open("results/w39/sweep_batch.csv") as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})

B        = [r["batch"] for r in rows]
latency  = [r["latency_ms"] for r in rows]
through  = [r["total_tokens_per_s"] for r in rows]
tpot     = [r["tpot_ms"] for r in rows]
mem      = [r["peak_mem_gb"] for r in rows]

CONFIG = "Qwen2.5-0.5B-Instruct, fp16, RTX 2080 Ti, max_new=64, min-of-5"

# ---------- 图 1: latency vs batch ----------
plt.figure(figsize=(6, 4))
plt.plot(B, latency, "o-")
plt.xlabel("Batch Size")
plt.ylabel("Latency (ms)")
plt.title(f"End-to-End Latency vs Batch Size\n{CONFIG}")
plt.grid(alpha=0.3)
plt.savefig("figures/latency_vs_batch.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图 2: throughput vs batch (主角图) ----------
plt.figure(figsize=(6, 4))
plt.plot(B, through, "s-", color="darkorange")
plt.plot(B, [through[0] * b for b in B], "--", alpha=0.5,   # 理想线性参考线
         label="Ideal linear scaling")
plt.xlabel("Batch Size")
plt.ylabel("Total Throughput (tokens/s)")
plt.title(f"Throughput vs Batch Size\n{CONFIG}")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/throughput_vs_batch.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图 3: 双指标小图 (TPOT + 显存) ----------
fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(B, tpot, "o-", color="tab:blue", label="TPOT (ms/step)")
ax1.set_xlabel("Batch Size")
ax1.set_ylabel("TPOT (ms/step)", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(B, mem, "s--", color="tab:red", label="Peak Memory (GB)")
ax2.set_ylabel("Peak Memory (GB)", color="tab:red")
plt.title(f"TPOT & Memory vs Batch Size\n{CONFIG}")
fig.savefig("figures/tpot_mem_vs_batch.png", dpi=150, bbox_inches="tight")
plt.close()

print("saved -> figures/latency_vs_batch.png, throughput_vs_batch.png, tpot_mem_vs_batch.png")