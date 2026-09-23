import argparse
import logging

parser = argparse.ArgumentParser(description="LLM Inference Benchmark")
parser.add_argument("--batch-size", type=int, default=1,
                    help="batch size for inference")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                    help="model name or path")
parser.add_argument("--output-dir", type=str, default="results",
                    help="directory to save results")

args = parser.parse_args()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                    # 输出到屏幕
        logging.FileHandler("benchmark.log"),       # 同时写入文件
    ]
)

logging.info("benchmark started")
logging.info(f"model={args.model}, batch_size={args.batch_size}")