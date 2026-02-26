#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def _load_rows_for_dir(results_dir):
    rows = []
    for r_dir in Path(results_dir).glob("r_*"):
        metrics_path = r_dir / "metrics.json"
        attn_path = r_dir / "attention_stats.json"
        if not metrics_path.exists() or not attn_path.exists():
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        with open(attn_path, "r", encoding="utf-8") as f:
            attn = json.load(f)

        rows.append(
            {
                "rate": float(metrics["retention_rate"]),
                "hr": float(metrics.get("hit_rate", 1.0 - float(metrics["hallucination_rate"]))),
                "rdp": float(metrics["reasoning_depth_proxy"]),
                "h_attn": float(attn["mean_entropy"]),
            }
        )

    rows.sort(key=lambda x: x["rate"], reverse=True)
    return rows


def _discover_seed_dirs(results_dir):
    root = Path(results_dir)
    seed_dirs = sorted([d for d in root.glob("seed_*") if d.is_dir()])
    if seed_dirs:
        return seed_dirs

    if any(d.is_dir() for d in root.glob("r_*")):
        return [root]

    return []


def _aggregate_by_rate(seed_rows):
    by_rate = {}
    for rows in seed_rows.values():
        for row in rows:
            rate = row["rate"]
            by_rate.setdefault(rate, {"hr": [], "rdp": [], "h_attn": []})
            by_rate[rate]["hr"].append(row["hr"])
            by_rate[rate]["rdp"].append(row["rdp"])
            by_rate[rate]["h_attn"].append(row["h_attn"])

    rates = sorted(by_rate.keys(), reverse=True)
    hr_mean = [float(np.mean(by_rate[r]["hr"])) for r in rates]
    rdp_mean = [float(np.mean(by_rate[r]["rdp"])) for r in rates]
    h_attn_mean = [float(np.mean(by_rate[r]["h_attn"])) for r in rates]
    return rates, hr_mean, rdp_mean, h_attn_mean


def plot_curves(results_dir, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc

    seed_dirs = _discover_seed_dirs(results_dir)
    if not seed_dirs:
        raise SystemExit(f"No retention results found in: {results_dir}")

    seed_rows = {}
    for idx, seed_dir in enumerate(seed_dirs, start=1):
        label = seed_dir.name if seed_dir != Path(results_dir) else f"seed_{idx:03d}"
        rows = _load_rows_for_dir(seed_dir)
        if rows:
            seed_rows[label] = rows

    if not seed_rows:
        raise SystemExit(f"No retention results found in: {results_dir}")

    rates, hr_mean, rdp_mean, h_attn_mean = _aggregate_by_rate(seed_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for label, rows in seed_rows.items():
        r = [x["rate"] for x in rows]
        axes[0].plot(r, [x["hr"] for x in rows], marker="o", alpha=0.35, linewidth=1, label=label)
        axes[1].plot(r, [x["rdp"] for x in rows], marker="o", alpha=0.35, linewidth=1, label=label)
        axes[2].plot(r, [x["h_attn"] for x in rows], marker="o", alpha=0.35, linewidth=1, label=label)

    axes[0].plot(rates, hr_mean, marker="o", color="black", linewidth=2.2, label="mean")
    axes[1].plot(rates, rdp_mean, marker="o", color="black", linewidth=2.2, label="mean")
    axes[2].plot(rates, h_attn_mean, marker="o", color="black", linewidth=2.2, label="mean")

    axes[0].set_title("Hit Rate (HR) vs Retention")
    axes[0].set_xlabel("Retention r")
    axes[0].set_ylabel("HR")

    axes[1].set_title("Reasoning Depth Proxy vs Retention")
    axes[1].set_xlabel("Retention r")
    axes[1].set_ylabel("RDP")

    axes[2].set_title("Attention Entropy vs Retention")
    axes[2].set_xlabel("Retention r")
    axes[2].set_ylabel("H_attn")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.invert_xaxis()

    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Saved plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot retention sweep curves")
    parser.add_argument("--results-dir", required=True, help="Path like results/0.5B")
    parser.add_argument("--out", default="analysis/retention_curves.png")
    args = parser.parse_args()

    plot_curves(results_dir=args.results_dir, out_path=args.out)


if __name__ == "__main__":
    main()
