#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RegimeShiftResult:
    detected: bool
    rate: float | None
    ci_low: float | None
    ci_high: float | None
    confidence: float


def _find_shift(rates, hr, rdp, h_attn):
    base_hr = hr[0]
    base_rdp = rdp[0]
    base_h = h_attn[0]

    for i, rate in enumerate(rates):
        h_drop = (base_h - h_attn[i]) / max(base_h, 1e-12)
        rdp_rise = (rdp[i] - base_rdp) / max(base_rdp, 1e-12)
        hr_drop = (base_hr - hr[i]) / max(base_hr, 1e-12)

        if h_drop > 0.20 and rdp_rise > 0.15 and hr_drop > 0.10:
            return i, rate

    return None, None


def _bootstrap_shift_rates(rates, hr, rdp, h_attn, bootstrap_iters, seed):
    rng = np.random.default_rng(seed)
    n = len(rates)
    detected = []

    for _ in range(int(bootstrap_iters)):
        sampled = rng.integers(0, n, size=n)
        s_rates = rates[sampled]
        s_hr = hr[sampled]
        s_rdp = rdp[sampled]
        s_h = h_attn[sampled]

        order = np.argsort(-s_rates)
        s_rates = s_rates[order]
        s_hr = s_hr[order]
        s_rdp = s_rdp[order]
        s_h = s_h[order]

        _, s_rate = _find_shift(s_rates, s_hr, s_rdp, s_h)
        if s_rate is not None:
            detected.append(float(s_rate))

    return np.asarray(detected, dtype=float)


def detect_regime_shift(rates, hr, rdp, h_attn, bootstrap_iters=1000, seed=2026):
    rates = np.asarray(rates, dtype=float)
    hr = np.asarray(hr, dtype=float)
    rdp = np.asarray(rdp, dtype=float)
    h_attn = np.asarray(h_attn, dtype=float)

    if len(rates) == 0:
        return RegimeShiftResult(False, None, None, None, 0.0), np.asarray([], dtype=float)

    order = np.argsort(-rates)
    rates = rates[order]
    hr = hr[order]
    rdp = rdp[order]
    h_attn = h_attn[order]

    _, rate = _find_shift(rates, hr, rdp, h_attn)
    if rate is None:
        return RegimeShiftResult(False, None, None, None, 0.0), np.asarray([], dtype=float)

    bootstrap_rates = _bootstrap_shift_rates(rates, hr, rdp, h_attn, bootstrap_iters=bootstrap_iters, seed=seed)
    if bootstrap_rates.size == 0:
        return RegimeShiftResult(True, float(rate), None, None, 0.0), bootstrap_rates

    ci_low = float(np.quantile(bootstrap_rates, 0.025))
    ci_high = float(np.quantile(bootstrap_rates, 0.975))
    confidence = float(bootstrap_rates.size / float(bootstrap_iters))
    return RegimeShiftResult(True, float(rate), ci_low, ci_high, confidence), bootstrap_rates


def _load_series(results_dir):
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
    rates = [row["rate"] for row in rows]
    hr = [row["hr"] for row in rows]
    rdp = [row["rdp"] for row in rows]
    h_attn = [row["h_attn"] for row in rows]
    return rates, hr, rdp, h_attn


def _discover_seed_dirs(results_dir):
    root = Path(results_dir)
    seed_dirs = sorted([d for d in root.glob("seed_*") if d.is_dir()])
    if seed_dirs:
        return seed_dirs

    has_flat = any(d.is_dir() for d in root.glob("r_*"))
    if has_flat:
        return [root]

    return []


def _format_result(label, result):
    lines = [
        f"[{label}] Regime shift detected at r = {result.rate:.2f}"
        if result.detected
        else f"[{label}] Regime shift detected at r = none"
    ]
    if result.detected and result.ci_low is not None and result.ci_high is not None:
        lines.append(f"[{label}] Confidence interval = [{result.ci_low:.2f}, {result.ci_high:.2f}]")
    else:
        lines.append(f"[{label}] Confidence interval = unavailable")
    lines.append(f"[{label}] Bootstrap confidence = {result.confidence:.3f}")
    return lines


def _merge_pooled_bootstrap(per_seed_results, per_seed_bootstrap, bootstrap_iters):
    detected_rates = np.asarray([x.rate for x in per_seed_results if x.detected and x.rate is not None], dtype=float)
    if detected_rates.size == 0:
        return None

    pooled = np.concatenate([arr for arr in per_seed_bootstrap if arr.size > 0], dtype=float)
    if pooled.size == 0:
        ci_candidates = [
            (x.ci_low, x.ci_high)
            for x in per_seed_results
            if x.detected and x.ci_low is not None and x.ci_high is not None
        ]
        if ci_candidates:
            ci_low = float(min(x[0] for x in ci_candidates))
            ci_high = float(max(x[1] for x in ci_candidates))
        else:
            point = float(np.median(detected_rates))
            ci_low = point
            ci_high = point
        confidence = 0.0
    else:
        ci_low = float(np.quantile(pooled, 0.025))
        ci_high = float(np.quantile(pooled, 0.975))
        confidence = float(pooled.size / float(max(1, bootstrap_iters * len(per_seed_results))))

    return {
        "rate": float(np.median(detected_rates)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": confidence,
        "seed_agreement": float(detected_rates.size / float(max(1, len(per_seed_results)))),
        "method": "pooled_bootstrap",
    }


def _merge_union_band(per_seed_results):
    detected_rates = np.asarray([x.rate for x in per_seed_results if x.detected and x.rate is not None], dtype=float)
    if detected_rates.size == 0:
        return None

    ci_candidates = [
        (x.ci_low, x.ci_high)
        for x in per_seed_results
        if x.detected and x.ci_low is not None and x.ci_high is not None
    ]
    if ci_candidates:
        ci_low = float(min(x[0] for x in ci_candidates))
        ci_high = float(max(x[1] for x in ci_candidates))
    else:
        ci_low = float(np.min(detected_rates))
        ci_high = float(np.max(detected_rates))

    return {
        "rate": float(np.median(detected_rates)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": float(np.mean([x.confidence for x in per_seed_results if x.detected])),
        "seed_agreement": float(detected_rates.size / float(max(1, len(per_seed_results)))),
        "method": "union_band",
    }


def main():
    parser = argparse.ArgumentParser(description="SRM regime detector")
    parser.add_argument("--results-dir", required=True, help="Path like results/0.5B/rlcp")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--merge-method",
        choices=["pooled", "union"],
        default="pooled",
        help="Merged CI method across seeds",
    )
    args = parser.parse_args()

    seed_dirs = _discover_seed_dirs(args.results_dir)
    if not seed_dirs:
        raise SystemExit(f"No runs found under: {args.results_dir}")

    per_seed_results = []
    per_seed_bootstrap = []

    for idx, seed_dir in enumerate(seed_dirs, start=1):
        rates, hr, rdp, h_attn = _load_series(seed_dir)
        if not rates:
            continue

        result, bootstrap_rates = detect_regime_shift(
            rates=rates,
            hr=hr,
            rdp=rdp,
            h_attn=h_attn,
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed + idx,
        )

        label = seed_dir.name if seed_dir != Path(args.results_dir) else "seed_001"
        for line in _format_result(label, result):
            print(line)

        per_seed_results.append(result)
        per_seed_bootstrap.append(bootstrap_rates)

    if not per_seed_results:
        raise SystemExit(f"No valid seed runs found under: {args.results_dir}")

    if args.merge_method == "union":
        merged = _merge_union_band(per_seed_results)
    else:
        merged = _merge_pooled_bootstrap(
            per_seed_results=per_seed_results,
            per_seed_bootstrap=per_seed_bootstrap,
            bootstrap_iters=args.bootstrap_iters,
        )

    if merged is None:
        print("[merged] Regime shift detected at r = none")
        print("[merged] Confidence interval = unavailable")
        print("[merged] Seed agreement = 0.000")
        print(f"[merged] Merge method = {args.merge_method}")
        return

    print(f"[merged] Regime shift detected at r = {merged['rate']:.2f}")
    print(f"[merged] Confidence interval = [{merged['ci_low']:.2f}, {merged['ci_high']:.2f}]")
    print(f"[merged] Seed agreement = {merged['seed_agreement']:.3f}")
    print(f"[merged] Bootstrap confidence = {merged['confidence']:.3f}")
    print(f"[merged] Merge method = {merged['method']}")


if __name__ == "__main__":
    main()
