import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW = 64
WARMUP_RUNS = 3
TIMED_RUNS = 5

class TimeStamper:
    """回调对象: generate 每产出一个 token 就调一次 put(), 结束时调 end()"""
    def __init__(self):
        self.ts = []
    def put(self, token_ids):
        self.ts.append(time.time())
    def end(self):
        pass

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda:0")
model.eval()

def make_inputs(prompt="用一句话解释什么是 KV Cache。"):
    return tok(prompt, return_tensors="pt").to("cuda:0")

# ---------- 阶段 1: warmup ----------
print(f"[warmup] {WARMUP_RUNS} 次 ...")
for _ in range(WARMUP_RUNS):
    model.generate(**make_inputs(), max_new_tokens=MAX_NEW,
                   min_new_tokens=MAX_NEW,              # ← 新增: warmup 也走同一路径
                   streamer=TimeStamper())
torch.cuda.synchronize()

# ---------- 阶段 2: decode 计时, min-of-N ----------
torch.cuda.reset_peak_memory_stats()
records = []
for i in range(TIMED_RUNS):
    ts = TimeStamper()
    inputs = make_inputs()
    t_start = time.time()
    model.generate(**inputs, max_new_tokens=MAX_NEW,
                   min_new_tokens=MAX_NEW,              # ← 新增: 强制生成满 64, 不许提前停
                   streamer=ts)
    torch.cuda.synchronize()

    ttft_submit = ts.ts[0] - t_start
    tpot = (ts.ts[-1] - ts.ts[0]) / (len(ts.ts) - 1)
    total = time.time() - t_start
    records.append((ttft_submit, tpot, total, len(ts.ts)))   # ← 修改: 记录实际生成数

for i, r in enumerate(records):                                # ← 新增: 打印每次 run
    print(f"    run{i}: latency={r[2]*1e3:6.1f}ms  tpot={r[1]*1e3:5.1f}ms  实际生成={r[3]} token")

best = tuple(min(r[j] for r in records) for j in range(3))
print(f"\nbatch=1  min-of-{TIMED_RUNS}:")
print(f"  TTFT(提交时间, 假象) = {best[0]*1e3:7.1f} ms")
print(f"  TPOT                 = {best[1]*1e3:7.1f} ms/token")
print(f"  latency              = {best[2]*1e3:7.1f} ms  (全程 {MAX_NEW} token)")
print(f"  单请求速度           = {MAX_NEW/best[2]:7.1f} tokens/s")
print(f"  峰值显存             = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# ---------- 真实 prefill 测量 (同步前向) ----------
for _ in range(3):
    model(**make_inputs())
torch.cuda.synchronize()

pre = []
for _ in range(TIMED_RUNS):
    inputs = make_inputs()
    torch.cuda.synchronize(); t = time.time()
    model(**inputs)
    torch.cuda.synchronize()
    pre.append(time.time() - t)
print(f"\n真实 prefill (同步前向, min-of-{TIMED_RUNS}) = {min(pre)*1e3:.1f} ms")

# ---------- 线性性检验 ----------
print("\n线性性检验:")
for n in (64, 128, 256):
    ts = TimeStamper()
    torch.cuda.synchronize(); t = time.time()
    model.generate(**make_inputs(), max_new_tokens=n,
                   min_new_tokens=n,                      # ← 新增: 同样强制生成满
                   streamer=ts)
    torch.cuda.synchronize()
    dt = time.time() - t
    # ← 修改: 分母用实际生成数, 并打印出来
    print(f"  n={n:3d}  latency={dt*1e3:8.1f} ms   实际生成={len(ts.ts):3d} token   每token={dt/len(ts.ts)*1e3:5.1f} ms")