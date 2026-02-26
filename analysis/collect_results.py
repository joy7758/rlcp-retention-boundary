#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc


def _discover_metric_files(results_root):
    root = Path(results_root)
    return sorted(root.glob("**/metrics.json"))


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_rows(results_root):
    rows = []
    for metrics_path in _discover_metric_files(results_root):
        run_dir = metrics_path.parent
        attn_path = run_dir / "attention_stats.json"
        flops_path = run_dir / "flops.json"
        cfg_path = run_dir / "run_config.yaml"

        if not attn_path.exists() or not flops_path.exists() or not cfg_path.exists():
            continue

        metrics = _read_json(metrics_path)
        attn = _read_json(attn_path)
        flops = _read_json(flops_path)
        cfg = _read_yaml(cfg_path)
        runtime = (cfg or {}).get("runtime", {})

        rows.append(
            {
                "run_id": metrics.get("run_id"),
                "model": metrics.get("model"),
                "mode": metrics.get("mode"),
                "retention_rate": metrics.get("retention_rate"),
                "seed": metrics.get("seed"),
                "seed_index": metrics.get("seed_index"),
                "hallucination_rate": metrics.get("hallucination_rate"),
                "hit_rate": metrics.get("hit_rate", 1.0 - float(metrics.get("hallucination_rate", 0.0))),
                "reasoning_depth_proxy": metrics.get("reasoning_depth_proxy"),
                "mean_attention_entropy": attn.get("mean_entropy"),
                "mean_loss": metrics.get("mean_loss"),
                "final_loss": metrics.get("final_loss"),
                "num_steps": metrics.get("num_steps"),
                "total_flops": flops.get("total_flops"),
                "flops_records": flops.get("num_records"),
                "git_commit": runtime.get("git_commit"),
                "git_commit_available": runtime.get("git_commit_available"),
                "timestamp_utc": metrics.get("timestamp_utc"),
                "run_dir": str(run_dir),
            }
        )

    rows.sort(key=lambda x: (str(x.get("model")), int(x.get("seed_index") or 0), -float(x.get("retention_rate") or 0.0)))
    return rows


def write_csv(rows, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Collect RLCP experiment results into a CSV")
    parser.add_argument("--results-root", default="results", help="Root results directory")
    parser.add_argument("--out", default="results/results.csv", help="CSV output path")
    args = parser.parse_args()

    rows = collect_rows(args.results_root)
    write_csv(rows, args.out)
    print(f"Collected {len(rows)} runs into: {args.out}")


if __name__ == "__main__":
    main()
