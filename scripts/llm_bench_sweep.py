import time, torch, csv
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "results" / "w39" / "sweep_batch.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 64          # 锁死生成长度
WARMUP_RUNS = 3
TIMED_RUNS = 5
BATCH_SIZES = [1, 2, 4, 8, 16]
PROMPT = "用一句话解释什么是 KV Cache。"

# ===== 预测(先填再跑) =====
# 1. 拐点前 TPOT ≈ ? ms 不动; 拐点大约在 B = ?
# 2. 拐点前总吞吐随 B 怎么涨? 拐点后呢?
# 3. 峰值显存随 B 怎么涨? (提示: 权重不变, 变的是什么?)

class TimeStamper:
    def __init__(self): self.ts = []
    def put(self, token_ids): self.ts.append(time.time())
    def end(self): pass

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda:0")
model.eval()

def make_batch(B):
    # tokenizer 接受字符串列表 -> batch 张量, attention_mask 自动生成
    return tok([PROMPT] * B, return_tensors="pt", padding=True).to("cuda:0")

rows = []
for B in BATCH_SIZES:
    print(f"\n===== B={B} =====")
    # 每个 B 单独 warmup: kernel 按形状惰性加载, 小 B 不预热会虚高
    for _ in range(WARMUP_RUNS):
        model.generate(**make_batch(B), max_new_tokens=MAX_NEW,
                       min_new_tokens=MAX_NEW, streamer=TimeStamper())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    rec = []
    for i in range(TIMED_RUNS):
        ts = TimeStamper()
        inputs = make_batch(B)
        torch.cuda.synchronize(); t = time.time()
        model.generate(**inputs, max_new_tokens=MAX_NEW,
                       min_new_tokens=MAX_NEW, streamer=ts)
        torch.cuda.synchronize()
        total = time.time() - t
        steps = len(ts.ts)                       # 预期 = 65 (64 强制 + 1 EOS)
        tpot = (ts.ts[-1] - ts.ts[0]) / (steps - 1)
        rec.append((total, tpot, steps))

    for i, r in enumerate(rec):
        print(f"  run{i}: latency={r[0]*1e3:7.1f}ms  tpot={r[1]*1e3:5.1f}ms  步数={r[2]}")

    best_lat  = min(r[0] for r in rec)
    best_tpot = min(r[1] for r in rec)
    total_tok_s = B * MAX_NEW / best_lat         # 吞吐 = 全部请求总 token / 时间
    mem = torch.cuda.max_memory_allocated() / 1e9
    rows.append((B, round(best_lat*1e3, 1), round(total_tok_s, 1),
                 round(best_tpot*1e3, 1), round(mem, 2)))
    print(f"--> B={B:2d}  latency={best_lat*1e3:7.1f}ms  "
          f"吞吐={total_tok_s:6.1f} tok/s  tpot={best_tpot*1e3:5.1f}ms  峰值显存={mem:.2f}GB")

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["batch", "latency_ms", "total_tokens_per_s", "tpot_ms", "peak_mem_gb"])
    w.writerows(rows)
print("\nsaved -> results/w39/sweep_batch.csv")