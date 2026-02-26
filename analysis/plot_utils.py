#!/usr/bin/env python3
import csv
import math
from collections import defaultdict

import numpy as np


def set_ieee_style():
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.facecolor": "white",
        }
    )


def _to_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value, default=None):
    if value is None:
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def load_results_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "run_id": row.get("run_id", ""),
                    "model": row.get("model", ""),
                    "task": row.get("task", "default") or "default",
                    "mode": row.get("mode", ""),
                    "retention_rate": _to_float(row.get("retention_rate")),
                    "seed_index": _to_int(row.get("seed_index")),
                    "hallucination_rate": _to_float(row.get("hallucination_rate")),
                    "hit_rate": _to_float(row.get("hit_rate")),
                    "reasoning_depth_proxy": _to_float(row.get("reasoning_depth_proxy")),
                    "mean_attention_entropy": _to_float(row.get("mean_attention_entropy")),
                    "timestamp_utc": row.get("timestamp_utc", ""),
                    "git_commit": row.get("git_commit", ""),
                    "run_dir": row.get("run_dir", ""),
                }
            )
    return rows


def filter_rows(rows, model=None, task=None, mode=None):
    out = []
    for row in rows:
        if model is not None and row["model"] != model:
            continue
        if task is not None and row["task"] != task:
            continue
        if mode is not None and row["mode"] != mode:
            continue
        if row["retention_rate"] is None or row["seed_index"] is None:
            continue
        out.append(row)
    return out


def keep_latest_per_key(rows, key_fields):
    latest = {}
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        prev = latest.get(key)
        if prev is None or row.get("timestamp_utc", "") > prev.get("timestamp_utc", ""):
            latest[key] = row
    return list(latest.values())


def mean_ci(values, z=1.96):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    stderr = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return mean, float(z * stderr)


def aggregate_metric_by_retention(rows, metric_key):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(metric_key)
        if value is None:
            continue
        grouped[float(row["retention_rate"])].append(float(value))

    rates = sorted(grouped.keys(), reverse=True)
    means, cis, counts = [], [], []
    for rate in rates:
        mean, ci = mean_ci(grouped[rate])
        means.append(mean)
        cis.append(ci)
        counts.append(len(grouped[rate]))
    return rates, means, cis, counts


def detect_shift_rate(rows):
    """Return first retention rate where SRM thresholds are all satisfied."""
    ordered = sorted(rows, key=lambda x: x["retention_rate"], reverse=True)
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


def regime_band_from_rows(rows):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed_index"])].append(row)

    shifts = []
    for _, seed_rows in by_seed.items():
        shift = detect_shift_rate(seed_rows)
        if shift is not None:
            shifts.append(float(shift))

    if not shifts:
        return None

    num_seeds = max(len(by_seed), 1)
    return {
        "low": float(min(shifts)),
        "high": float(max(shifts)),
        "center": float(np.median(np.asarray(shifts, dtype=float))),
        "agreement": float(len(shifts) / num_seeds),
        "n_detected": len(shifts),
        "n_seeds": len(by_seed),
    }


def _detect_shift_from_arrays(rates, hr, rdp, h_attn):
    if len(rates) == 0:
        return None
    base_hr = hr[0]
    base_rdp = rdp[0]
    base_h = h_attn[0]

    for i, rate in enumerate(rates):
        h_drop = (base_h - h_attn[i]) / max(base_h, 1e-12)
        rdp_rise = (rdp[i] - base_rdp) / max(base_rdp, 1e-12)
        hr_drop = (base_hr - hr[i]) / max(base_hr, 1e-12)
        if h_drop > 0.20 and rdp_rise > 0.15 and hr_drop > 0.10:
            return float(rate)
    return None


def _bootstrap_shift_rates(seed_rows, bootstrap_iters=1000, seed=2026):
    ordered = sorted(seed_rows, key=lambda x: x["retention_rate"], reverse=True)
    rates = np.asarray([float(x["retention_rate"]) for x in ordered], dtype=float)
    hr = np.asarray([float(x["hit_rate"]) for x in ordered], dtype=float)
    rdp = np.asarray([float(x["reasoning_depth_proxy"]) for x in ordered], dtype=float)
    h_attn = np.asarray([float(x["mean_attention_entropy"]) for x in ordered], dtype=float)
    n = len(rates)
    if n == 0:
        return np.asarray([], dtype=float), None

    point = _detect_shift_from_arrays(rates, hr, rdp, h_attn)
    if point is None:
        return np.asarray([], dtype=float), None

    rng = np.random.default_rng(seed)
    detected = []
    for _ in range(int(bootstrap_iters)):
        idx = rng.integers(0, n, size=n)
        sr = rates[idx]
        shr = hr[idx]
        srdp = rdp[idx]
        sh = h_attn[idx]
        order = np.argsort(-sr)
        sr = sr[order]
        shr = shr[order]
        srdp = srdp[order]
        sh = sh[order]
        shift = _detect_shift_from_arrays(sr, shr, srdp, sh)
        if shift is not None:
            detected.append(float(shift))
    return np.asarray(detected, dtype=float), float(point)


def pooled_regime_band_from_rows(rows, bootstrap_iters=1000, seed=2026):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed_index"])].append(row)

    if not by_seed:
        return None

    per_seed_bootstrap = []
    per_seed_points = []
    for seed_idx, seed_rows in by_seed.items():
        boot, point = _bootstrap_shift_rates(
            seed_rows,
            bootstrap_iters=bootstrap_iters,
            seed=seed + int(seed_idx),
        )
        per_seed_bootstrap.append(boot)
        if point is not None:
            per_seed_points.append(point)

    if not per_seed_points:
        return None

    pooled = np.concatenate([x for x in per_seed_bootstrap if x.size > 0], dtype=float)
    if pooled.size > 0:
        low = float(np.quantile(pooled, 0.025))
        high = float(np.quantile(pooled, 0.975))
        confidence = float(pooled.size / float(max(1, bootstrap_iters * len(by_seed))))
    else:
        low = float(min(per_seed_points))
        high = float(max(per_seed_points))
        confidence = 0.0

    return {
        "low": low,
        "high": high,
        "center": float(np.median(np.asarray(per_seed_points, dtype=float))),
        "agreement": float(len(per_seed_points) / float(max(1, len(by_seed)))),
        "confidence": confidence,
        "n_detected": len(per_seed_points),
        "n_seeds": len(by_seed),
        "method": "pooled_bootstrap",
    }


def model_to_billions(model_name):
    text = str(model_name).strip().upper().replace("B", "")
    return float(text)
