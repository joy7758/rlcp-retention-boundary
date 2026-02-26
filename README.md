# rlcp-retention-boundary

Reproducible retention-dependent structural transition under controlled unlearning (RLCP).

## Reproducibility policy

- Fixed random seed in every model YAML.
- Per-run `run_config.yaml` snapshot saved with outputs.
- Deterministic run seed derived from `(base_seed, mode, retention_rate)`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
bash run_experiments.sh rlcp --model 0.5B --retention 0.07 --seeds 1
```

## Modes

- `baseline`
- `random`
- `gr`
- `beta`
- `rlcp`

## Publication-grade checks

- Version-pinned dependencies in `requirements.txt`.
- JSON Schema validation for `metrics.json`, `attention_stats.json`, and `flops.json`.
- `run_config.yaml` includes `run_id` and `git_commit`.
- `analysis/collect_results.py` exports a flat `results.csv`.
