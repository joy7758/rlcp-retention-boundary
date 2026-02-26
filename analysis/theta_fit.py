#!/usr/bin/env python3
import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from plot_utils import (
    filter_rows,
    keep_latest_per_key,
    load_results_csv,
    mean_ci,
    model_to_billions,
    regime_band_from_rows,
    set_ieee_style,
)


def fit_theta(N_values, theta_values):
    logN = np.log(N_values)
    logTheta = np.log(theta_values)
    alpha, log_c = np.polyfit(logN, logTheta, 1)
    return -alpha, np.exp(log_c)


def fit_theta_with_stats(N_values, theta_values, z=1.96):
    N_values = np.asarray(N_values, dtype=float)
    theta_values = np.asarray(theta_values, dtype=float)

    if N_values.shape != theta_values.shape:
        raise ValueError("N_values and theta_values must have the same shape")
    if np.any(N_values <= 0) or np.any(theta_values <= 0):
        raise ValueError("N_values and theta_values must be strictly positive")

    logN = np.log(N_values)
    logTheta = np.log(theta_values)

    if N_values.size <= 2:
        slope, intercept = np.polyfit(logN, logTheta, 1)
        cov = None
    else:
        coeffs, cov = np.polyfit(logN, logTheta, 1, cov=True)
        slope, intercept = coeffs

    pred = slope * logN + intercept
    ss_res = np.sum((logTheta - pred) ** 2)
    ss_tot = np.sum((logTheta - np.mean(logTheta)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0))

    if cov is None:
        se_slope = math.nan
        se_intercept = math.nan
    else:
        se_slope = float(np.sqrt(max(cov[0, 0], 0.0)))
        se_intercept = float(np.sqrt(max(cov[1, 1], 0.0)))

    alpha = float(-slope)
    c = float(np.exp(intercept))

    if math.isnan(se_slope) or math.isnan(se_intercept):
        alpha_ci = (math.nan, math.nan)
        c_ci = (math.nan, math.nan)
    else:
        alpha_ci = (
            float(-(slope + z * se_slope)),
            float(-(slope - z * se_slope)),
        )
        c_ci = (
            float(np.exp(intercept - z * se_intercept)),
            float(np.exp(intercept + z * se_intercept)),
        )

    return {
        "alpha": alpha,
        "c": c,
        "r2": r2,
        "alpha_ci": alpha_ci,
        "c_ci": c_ci,
    }


def _detect_shift_rate(seed_rows):
    ordered = sorted(seed_rows, key=lambda x: x["retention_rate"], reverse=True)
    if not ordered:
        return None

    base = ordered[0]
    base_hr = float(base["hit_rate"])
    base_rdp = float(base["reasoning_depth_proxy"])
    base_h = float(base["mean_attention_entropy"])

    for row in ordered:
        h_drop = (base_h - float(row["mean_attention_entropy"])) / max(base_h, 1e-12)
        rdp_rise = (float(row["reasoning_depth_proxy"]) - base_rdp) / max(base_rdp, 1e-12)
        hr_drop = (base_hr - float(row["hit_rate"])) / max(base_hr, 1e-12)
        if h_drop > 0.20 and rdp_rise > 0.15 and hr_drop > 0.10:
            return float(row["retention_rate"])

    return None


def _theta_summary_from_rows(rows):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed_index"])].append(row)

    shifts = []
    for _, seed_rows in by_seed.items():
        shift = _detect_shift_rate(seed_rows)
        if shift is not None:
            shifts.append(float(shift))

    if not shifts:
        return None

    arr = np.asarray(shifts, dtype=float)
    mean, ci = mean_ci(arr)
    return {
        "theta_c": float(np.median(arr)),
        "theta_mean": float(mean),
        "theta_ci_low": float(np.quantile(arr, 0.025)),
        "theta_ci_high": float(np.quantile(arr, 0.975)),
        "theta_ci_halfwidth": float(ci),
        "agreement": float(len(arr) / max(len(by_seed), 1)),
        "n_detected": int(len(arr)),
        "n_seeds": int(len(by_seed)),
    }


def plot_theta_from_csv(input_csv, out_path, task="default", mode="rlcp", dpi=320):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc

    set_ieee_style()

    rows = load_results_csv(input_csv)
    rows = filter_rows(rows, task=task, mode=mode)
    rows = keep_latest_per_key(rows, key_fields=["model", "seed_index", "retention_rate"])

    models = sorted({row["model"] for row in rows}, key=model_to_billions)
    if not models:
        raise SystemExit(f"No rows found for task={task}, mode={mode}")

    points = []
    for model in models:
        mrows = [r for r in rows if r["model"] == model]
        summary = _theta_summary_from_rows(mrows)
        if summary is None:
            continue
        points.append((model, model_to_billions(model), summary))

    if not points:
        raise SystemExit("No detectable theta_c points were found for plotting")

    x = np.asarray([p[1] for p in points], dtype=float)
    y = np.asarray([p[2]["theta_c"] for p in points], dtype=float)
    yerr_low = np.asarray([p[2]["theta_c"] - p[2]["theta_ci_low"] for p in points], dtype=float)
    yerr_high = np.asarray([p[2]["theta_ci_high"] - p[2]["theta_c"] for p in points], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.errorbar(
        x,
        y,
        yerr=[yerr_low, yerr_high],
        fmt="o",
        color="#1f4e79",
        ecolor="#1f4e79",
        elinewidth=1.5,
        capsize=4,
        markersize=6,
    )
    ax.plot(x, y, color="#1f4e79", linewidth=1.4, alpha=0.8)

    for model, xb, summary in points:
        ax.annotate(
            f"{model}\n[{summary['theta_ci_low']:.3f}, {summary['theta_ci_high']:.3f}]",
            (xb, summary["theta_c"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Model Size N (billions, log scale)")
    ax.set_ylabel(r"$\theta_c$")
    ax.set_title("Retention boundary across tested scales")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    print(f"Saved plot to: {out}")

    if len(points) >= 2:
        stats = fit_theta_with_stats(x, y)
        print(f"alpha = {stats['alpha']:.6f}")
        print(f"R^2 = {stats['r2']:.6f}")
        print(f"95% CI(alpha) = [{stats['alpha_ci'][0]:.6f}, {stats['alpha_ci'][1]:.6f}]")

    for model, _, summary in points:
        print(
            f"theta_c({model}) = {summary['theta_c']:.3f}, "
            f"CI=[{summary['theta_ci_low']:.3f}, {summary['theta_ci_high']:.3f}], "
            f"agreement={summary['agreement']:.3f} ({summary['n_detected']}/{summary['n_seeds']})"
        )


def _parse_csv_floats(text):
    return np.asarray([float(x.strip()) for x in text.split(",") if x.strip()], dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Fit or plot theta boundary")
    parser.add_argument("--N", default=None, help="Comma-separated N values, e.g. 0.5,1.5,7")
    parser.add_argument("--theta", default=None, help="Comma-separated theta values")
    parser.add_argument("--input", default=None, help="CSV from collect_results.py")
    parser.add_argument("--out", default=None, help="Figure output path")
    parser.add_argument("--task", default="default")
    parser.add_argument("--mode", default="rlcp")
    parser.add_argument("--dpi", type=int, default=320)
    args = parser.parse_args()

    if args.input:
        if not args.out:
            raise SystemExit("--out is required when using --input")
        plot_theta_from_csv(
            input_csv=args.input,
            out_path=args.out,
            task=args.task,
            mode=args.mode,
            dpi=args.dpi,
        )
        return

    if not args.N or not args.theta:
        raise SystemExit("Either provide --input/--out, or provide both --N and --theta")

    N_values = _parse_csv_floats(args.N)
    theta_values = _parse_csv_floats(args.theta)
    stats = fit_theta_with_stats(N_values, theta_values)

    print(f"alpha = {stats['alpha']:.6f}")
    print(f"R^2 = {stats['r2']:.6f}")
    print(f"95% CI(alpha) = [{stats['alpha_ci'][0]:.6f}, {stats['alpha_ci'][1]:.6f}]")
    print(f"95% CI(c) = [{stats['c_ci'][0]:.6f}, {stats['c_ci'][1]:.6f}]")


if __name__ == "__main__":
    main()
