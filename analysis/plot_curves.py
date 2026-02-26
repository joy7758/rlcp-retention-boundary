#!/usr/bin/env python3
import argparse
from pathlib import Path

from plot_utils import (
    aggregate_metric_by_retention,
    filter_rows,
    keep_latest_per_key,
    load_results_csv,
    pooled_regime_band_from_rows,
    set_ieee_style,
)


def plot_curves(input_csv, model, out_path, task="default", mode="rlcp", dpi=320):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc

    set_ieee_style()

    rows = load_results_csv(input_csv)
    rows = filter_rows(rows, model=model, task=task, mode=mode)
    rows = keep_latest_per_key(rows, key_fields=["seed_index", "retention_rate"])
    if not rows:
        raise SystemExit(f"No rows found for model={model}, task={task}, mode={mode} in {input_csv}")

    band = pooled_regime_band_from_rows(rows, bootstrap_iters=1000, seed=2026)

    metric_specs = [
        ("hallucination_rate", "Hallucination Rate", "#2f4f4f"),
        ("reasoning_depth_proxy", "Reasoning Depth Proxy", "#1f77b4"),
        ("mean_attention_entropy", "Attention Entropy", "#5c6f7b"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    for ax, (metric_key, ylabel, color) in zip(axes, metric_specs):
        rates, means, cis, counts = aggregate_metric_by_retention(rows, metric_key=metric_key)
        lower = [m - c for m, c in zip(means, cis)]
        upper = [m + c for m, c in zip(means, cis)]

        ax.plot(rates, means, color=color, marker="o", linewidth=2.0)
        ax.fill_between(rates, lower, upper, color=color, alpha=0.28, linewidth=0.8, edgecolor=color)

        if band is not None:
            ax.axvspan(band["low"], band["high"], color="#f0ad4e", alpha=0.18)

        ax.set_xlabel("Retention r")
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()

    if band is not None:
        fig.suptitle(f"{model} | task={task} | mode={mode}", y=1.03)
        axes[0].text(
            0.02,
            0.98,
            (
                f"r_c ≈ {band['center']:.3f} "
                f"(CI: [{band['low']:.3f}, {band['high']:.3f}])"
            ),
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
        )
    else:
        fig.suptitle(f"{model} | task={task} | mode={mode} | regime not detected", y=1.03)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    print(f"Saved plot to: {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot retention sweep curves with mean +/- CI")
    parser.add_argument("--input", required=True, help="CSV from collect_results.py")
    parser.add_argument("--model", required=True, help="Model label, e.g. 0.5B")
    parser.add_argument("--task", default="default", help="Task namespace")
    parser.add_argument("--mode", default="rlcp", help="Experiment mode")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=320)
    args = parser.parse_args()

    plot_curves(
        input_csv=args.input,
        model=args.model,
        out_path=args.out,
        task=args.task,
        mode=args.mode,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
