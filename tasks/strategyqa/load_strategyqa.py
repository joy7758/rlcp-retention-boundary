from datasets import load_dataset
import json
from pathlib import Path

out_path = Path(__file__).resolve().parent / "strategyqa_subset.json"
dataset = load_dataset("tasksource/strategy-qa", split="train")
subset = dataset.shuffle(seed=42).select(range(200))

with out_path.open("w", encoding="utf-8") as f:
    json.dump(subset.to_list(), f, ensure_ascii=False)

print(f"saved {len(subset)} rows -> {out_path}")
