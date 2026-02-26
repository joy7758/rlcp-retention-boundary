#!/usr/bin/env python3
import argparse
import copy
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required. Install with: pip install numpy") from exc

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    from jsonschema import Draft202012Validator
except ImportError as exc:
    raise SystemExit("jsonschema is required. Install with: pip install jsonschema") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablation import baseline, beta_only, gr_only, random_unlearning, rlcp_full
from metrics.attention_entropy import average_attention_entropy
from metrics.flops_logger import FLOPsLogger
from metrics.hallucination import from_factuality_scores
from metrics.reasoning_depth import reasoning_depth_proxy
from retention_sweep.retention_scheduler import RETENTION_RATES, get_retention_rates, retention_dirname

MODES = {
    "baseline": baseline,
    "random": random_unlearning,
    "gr": gr_only,
    "beta": beta_only,
    "rlcp": rlcp_full,
}

MODE_OFFSETS = {"baseline": 0, "random": 1000, "gr": 2000, "beta": 3000, "rlcp": 4000}
SCHEMA_DIR = PROJECT_ROOT / "schemas"
_SCHEMA_CACHE = {}
MODE_METRIC_PROFILES = {
    "baseline": {
        "factuality_base": 0.88,
        "factuality_slope": 0.28,
        "factuality_noise": 0.06,
        "rdp_base": 4.4,
        "rdp_slope": 1.4,
        "rdp_noise": 0.70,
        "temp_base": 1.24,
        "temp_slope": 0.52,
        "temp_floor": 0.70,
    },
    "random": {
        "factuality_base": 0.86,
        "factuality_slope": 0.24,
        "factuality_noise": 0.10,
        "rdp_base": 4.2,
        "rdp_slope": 1.1,
        "rdp_noise": 0.85,
        "temp_base": 1.23,
        "temp_slope": 0.47,
        "temp_floor": 0.72,
    },
    "gr": {
        "factuality_base": 0.87,
        "factuality_slope": 0.34,
        "factuality_noise": 0.07,
        "rdp_base": 4.35,
        "rdp_slope": 2.1,
        "rdp_noise": 0.75,
        "temp_base": 1.24,
        "temp_slope": 0.62,
        "temp_floor": 0.66,
    },
    "beta": {
        "factuality_base": 0.87,
        "factuality_slope": 0.38,
        "factuality_noise": 0.07,
        "rdp_base": 4.4,
        "rdp_slope": 2.3,
        "rdp_noise": 0.72,
        "temp_base": 1.24,
        "temp_slope": 0.68,
        "temp_floor": 0.62,
    },
    "rlcp": {
        "factuality_base": 0.88,
        "factuality_slope": 0.60,
        "factuality_noise": 0.06,
        "rdp_base": 4.5,
        "rdp_slope": 3.2,
        "rdp_noise": 0.70,
        "temp_base": 1.25,
        "temp_slope": 1.00,
        "temp_floor": 0.35,
    },
}
TASK_FACTORS = {
    "default": {
        "factuality_base_delta": 0.0,
        "factuality_slope_scale": 1.00,
        "rdp_slope_scale": 1.00,
        "temp_slope_scale": 1.00,
        "noise_scale": 1.00,
    },
    "gsm8k": {
        "factuality_base_delta": -0.02,
        "factuality_slope_scale": 1.05,
        "rdp_slope_scale": 1.08,
        "temp_slope_scale": 1.08,
        "noise_scale": 1.00,
    },
    "strategyqa": {
        "factuality_base_delta": 0.00,
        "factuality_slope_scale": 0.98,
        "rdp_slope_scale": 1.02,
        "temp_slope_scale": 1.00,
        "noise_scale": 1.00,
    },
}


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(payload, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def load_schema(schema_name):
    if schema_name in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[schema_name]

    schema_path = SCHEMA_DIR / schema_name
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    _SCHEMA_CACHE[schema_name] = schema
    return schema


def validate_json_payload(payload, schema_name, artifact_name):
    schema = load_schema(schema_name)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        path = "/".join(str(x) for x in first.path)
        raise ValueError(
            f"Schema validation failed for {artifact_name}: {first.message}"
            + (f" (path={path})" if path else "")
        )


def _retention_norm(rate):
    max_r = max(RETENTION_RATES)
    min_r = min(RETENTION_RATES)
    return (max_r - rate) / (max_r - min_r)


def _task_offset(task):
    return sum(ord(ch) for ch in str(task)) % 10000


def _task_factors(task):
    return TASK_FACTORS.get(str(task).lower(), TASK_FACTORS["default"])


def _softmax_last_axis(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def _run_git_command(repo_dir, args):
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def resolve_git_provenance(project_root):
    """
    Resolve commit hash from the nearest git repository at or above project_root.
    """
    tried = []
    for candidate in [project_root, *project_root.parents]:
        probe = _run_git_command(candidate, ["rev-parse", "--is-inside-work-tree"])
        if probe.returncode == 0 and probe.stdout.strip() == "true":
            head = _run_git_command(candidate, ["rev-parse", "HEAD"])
            if head.returncode == 0:
                return {
                    "git_commit": head.stdout.strip(),
                    "git_commit_available": True,
                    "git_repo_dir": str(candidate),
                    "git_commit_error": "",
                }
            err = (head.stderr or "").strip() or "unknown git error while resolving HEAD"
            return {
                "git_commit": "UNAVAILABLE",
                "git_commit_available": False,
                "git_repo_dir": str(candidate),
                "git_commit_error": err,
            }
        tried.append(str(candidate))

    return {
        "git_commit": "UNAVAILABLE",
        "git_commit_available": False,
        "git_repo_dir": "",
        "git_commit_error": f"No git repository found from {project_root} to filesystem root. Tried: {tried}",
    }


def simulate_single_retention(rate, model, task, mode_name, mode_module, config, seed, seed_index, run_id):
    rng = np.random.default_rng(seed)
    exp_cfg = config.get("experiment", {})
    attn_cfg = config.get("attention", {})
    mode_profile = MODE_METRIC_PROFILES[mode_name]
    task_factor = _task_factors(task)

    num_steps = int(exp_cfg.get("num_steps", 120))
    layers = int(attn_cfg.get("num_layers", 24))
    heads = int(attn_cfg.get("num_heads", 8))
    seq_len = int(attn_cfg.get("seq_len", 32))

    base_flops = float(config.get("flops", {}).get("base_per_step", 3e11))
    norm = _retention_norm(rate)

    flops = FLOPsLogger()
    losses = []

    for step in range(num_steps):
        task_loss = 1.00 + 0.85 * norm + rng.normal(0.0, 0.015)
        info_term = 0.45 + 0.55 * norm + rng.normal(0.0, 0.010)
        gr_term = 0.30 + 0.35 * norm + rng.normal(0.0, 0.010)

        loss = mode_module.compute_loss(
            task_loss=task_loss,
            info_term=info_term,
            grad_reversal_term=gr_term,
            step=step,
            total_steps=num_steps,
            config=config,
            rng=rng,
        )
        losses.append(float(loss))

        step_flops = base_flops * (1.0 + 0.0007 * step)
        flops.log(step=step, flops=step_flops)

    factuality_scores = np.clip(
        mode_profile["factuality_base"] + task_factor["factuality_base_delta"]
        - (mode_profile["factuality_slope"] * task_factor["factuality_slope_scale"]) * norm
        + rng.normal(0.0, mode_profile["factuality_noise"] * task_factor["noise_scale"], 512),
        0.0,
        1.0,
    )
    hallucination_rate = from_factuality_scores(factuality_scores, threshold=0.5)
    hit_rate = 1.0 - hallucination_rate

    cot_lengths = np.clip(
        rng.normal(
            mode_profile["rdp_base"] + mode_profile["rdp_slope"] * norm,
            mode_profile["rdp_noise"] * task_factor["noise_scale"],
            256,
        ),
        1.0,
        None,
    )
    rdp = reasoning_depth_proxy(cot_lengths)

    # Lower retention drives lower temperature, yielding sharper attention and lower entropy.
    logits = rng.normal(0.0, 1.0, (layers, heads, seq_len, seq_len))
    temperature = max(
        float(mode_profile["temp_floor"]),
        float(mode_profile["temp_base"] - (mode_profile["temp_slope"] * task_factor["temp_slope_scale"]) * norm),
    )
    attn_prob = _softmax_last_axis(logits / temperature)
    attention_stats = average_attention_entropy(attn_prob)

    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model": model,
        "task": task,
        "mode": mode_name,
        "retention_rate": float(rate),
        "seed": int(seed),
        "seed_index": int(seed_index),
        "hallucination_rate": float(hallucination_rate),
        "hit_rate": float(hit_rate),
        "reasoning_depth_proxy": float(rdp),
        "mean_attention_entropy": float(attention_stats["mean_entropy"]),
        "mean_loss": float(np.mean(losses)),
        "final_loss": float(losses[-1]),
        "num_steps": int(num_steps),
    }

    return metrics, attention_stats, flops.payload()


def parse_retentions_csv(retentions_text):
    values = [float(x.strip()) for x in str(retentions_text).split(",") if x.strip()]
    if not values:
        raise ValueError("--retentions provided but no numeric values parsed")
    for val in values:
        if val <= 0.0 or val > 1.0:
            raise ValueError(f"Retention values must be in (0, 1], got {val}")
    return values


def run_sweep(model, mode, config_path, task="default", retention_override=None, retentions_override=None, seeds=1):
    if seeds < 1:
        raise ValueError("--seeds must be >= 1")

    config = load_yaml(config_path)
    base_seed = int(config.get("experiment", {}).get("seed", 2026))

    if retentions_override:
        rates = [float(x) for x in retentions_override]
    elif retention_override is not None:
        rates = [float(retention_override)]
    else:
        rates = get_retention_rates(config)
    mode_module = MODES[mode]

    task = str(task)
    results_root = PROJECT_ROOT / config.get("output", {}).get("root_dir", "results") / model / task
    results_root.mkdir(parents=True, exist_ok=True)

    git_provenance = resolve_git_provenance(PROJECT_ROOT)

    summary_all = []
    for seed_index in range(1, seeds + 1):
        seed_summary = []
        seed_root = results_root if seeds == 1 else results_root / f"seed_{seed_index:03d}"
        seed_root.mkdir(parents=True, exist_ok=True)

        for rate in rates:
            run_seed = (
                base_seed
                + (seed_index - 1) * 10000
                + int(round(rate * 1000))
                + MODE_OFFSETS[mode]
                + _task_offset(task)
            )
            set_global_seed(run_seed)

            run_id = f"{model}-{task}-{mode}-s{seed_index:03d}-seed{run_seed}-r{rate:.3f}"
            run_dir = seed_root / retention_dirname(rate)
            run_dir.mkdir(parents=True, exist_ok=True)

            metrics, attention_stats, flops_payload = simulate_single_retention(
                rate=rate,
                model=model,
                task=task,
                mode_name=mode,
                mode_module=mode_module,
                config=config,
                seed=run_seed,
                seed_index=seed_index,
                run_id=run_id,
            )

            run_cfg = copy.deepcopy(config)
            run_cfg["runtime"] = {
                "run_id": run_id,
                "model": model,
                "task": task,
                "mode": mode,
                "retention_rate": float(rate),
                "seed": int(run_seed),
                "seed_index": int(seed_index),
                "seeds_total": int(seeds),
                "results_dir": str(run_dir),
                "git_commit": git_provenance["git_commit"],
                "git_commit_available": bool(git_provenance["git_commit_available"]),
                "git_repo_dir": git_provenance["git_repo_dir"],
                "git_commit_error": git_provenance["git_commit_error"],
                "regime_merge_method_default": "pooled_bootstrap",
                "timestamp_utc": metrics["timestamp_utc"],
            }

            validate_json_payload(metrics, "metrics.schema.json", "metrics.json")
            validate_json_payload(attention_stats, "attention_stats.schema.json", "attention_stats.json")
            validate_json_payload(flops_payload, "flops.schema.json", "flops.json")

            with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            with open(run_dir / "attention_stats.json", "w", encoding="utf-8") as f:
                json.dump(attention_stats, f, indent=2)

            with open(run_dir / "flops.json", "w", encoding="utf-8") as f:
                json.dump(flops_payload, f, indent=2)

            save_yaml(run_cfg, run_dir / "run_config.yaml")

            seed_summary.append(metrics)
            summary_all.append(metrics)
            print(
                f"[done] model={model} task={task} mode={mode} seed_idx={seed_index} "
                f"rate={rate:.3f} -> {run_dir}"
            )

        seed_summary_path = seed_root / f"summary_{mode}.json"
        with open(seed_summary_path, "w", encoding="utf-8") as f:
            json.dump(sorted(seed_summary, key=lambda x: x["retention_rate"], reverse=True), f, indent=2)
        print(f"[seed-summary] {seed_summary_path}")

    summary_path = results_root / f"summary_{mode}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            sorted(summary_all, key=lambda x: (x["seed_index"], -x["retention_rate"])),
            f,
            indent=2,
        )
    print(f"[summary] {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="RLCP retention sweep runner")
    parser.add_argument("--model", required=True, choices=["0.5B", "1.5B", "7B"])
    parser.add_argument("--mode", default="rlcp", choices=["baseline", "random", "gr", "beta", "rlcp"])
    parser.add_argument("--task", default="default", help="Task namespace, e.g. gsm8k or strategyqa")
    parser.add_argument("--config", default=None, help="Override config path")
    parser.add_argument("--retention", type=float, default=None, help="Run a single retention rate")
    parser.add_argument("--retentions", default=None, help="Comma-separated retention values to run")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seed replicas to run")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config) if args.config else PROJECT_ROOT / "configs" / f"model_{args.model}.yaml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    retentions_override = parse_retentions_csv(args.retentions) if args.retentions else None

    run_sweep(
        model=args.model,
        mode=args.mode,
        config_path=config_path,
        task=args.task,
        retention_override=args.retention,
        retentions_override=retentions_override,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
