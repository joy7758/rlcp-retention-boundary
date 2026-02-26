#!/usr/bin/env python3
import argparse
from pathlib import Path

from plot_utils import (
    aggregate_metric_by_retention,
    filter_rows,
    keep_latest_per_key,
    load_results_csv,
    regime_band_from_rows,
    set_ieee_style,
)

MODE_ORDER = ["baseline", "random", "gr", "beta", "rlcp"]
MODE_STYLE = {
    "baseline": {"color": "#4d4d4d", "linestyle": "-"},
    "random": {"color": "#7f7f7f", "linestyle": "--"},
    "gr": {"color": "#a6a6a6", "linestyle": "-."},
    "beta": {"color": "#bdbdbd", "linestyle": ":"},
    "rlcp": {"color": "#1f4e79", "linestyle": "-"},
}


def plot_ablation(input_csv, model, out_path, task="default", metric="reasoning_depth_proxy", dpi=320):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc

    set_ieee_style()

    rows = load_results_csv(input_csv)
    rows = filter_rows(rows, model=model, task=task)
    rows = keep_latest_per_key(rows, key_fields=["mode", "seed_index", "retention_rate"])

    if not rows:
        raise SystemExit(f"No rows found for model={model}, task={task}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)

    rlcp_rows = [r for r in rows if r["mode"] == "rlcp"]
    rlcp_band = regime_band_from_rows(rlcp_rows)

    if rlcp_band is not None:
        ax.axvspan(rlcp_band["low"], rlcp_band["high"], color="#f0ad4e", alpha=0.18)

    plotted_modes = []
    for mode in MODE_ORDER:
        mode_rows = [r for r in rows if r["mode"] == mode]
        if not mode_rows:
            continue
        rates, means, cis, _ = aggregate_metric_by_retention(mode_rows, metric_key=metric)
        style = MODE_STYLE[mode]

        ax.plot(
            rates,
            means,
            label=mode,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2 if mode == "rlcp" else 1.6,
            marker="o",
            markersize=4,
        )
        lower = [m - c for m, c in zip(means, cis)]
        upper = [m + c for m, c in zip(means, cis)]
        ax.fill_between(rates, lower, upper, color=style["color"], alpha=0.10, linewidth=0)
        plotted_modes.append(mode)

    ylabel = {
        "reasoning_depth_proxy": "Reasoning Depth Proxy",
        "mean_attention_entropy": "Attention Entropy",
        "hallucination_rate": "Hallucination Rate",
    }.get(metric, metric)

    ax.set_xlabel("Retention r")
    ax.set_ylabel(ylabel)
    ax.invert_xaxis()

    title = f"Ablation at {model} | task={task}"
    if rlcp_band is not None:
        title += f" | RLCP band [{rlcp_band['low']:.3f}, {rlcp_band['high']:.3f}]"
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    print(f"Saved plot to: {out}")
    print(f"Plotted modes: {plotted_modes}")


def main():
    parser = argparse.ArgumentParser(description="Plot RLCP ablation comparison")
    parser.add_argument("--input", required=True, help="CSV from collect_results.py")
    parser.add_argument("--model", required=True, help="Model label, e.g. 0.5B")
    parser.add_argument("--task", default="default", help="Task namespace")
    parser.add_argument("--metric", default="reasoning_depth_proxy")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=320)
    args = parser.parse_args()

    plot_ablation(
        input_csv=args.input,
        model=args.model,
        out_path=args.out,
        task=args.task,
        metric=args.metric,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
