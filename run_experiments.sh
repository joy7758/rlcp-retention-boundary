#!/usr/bin/env bash
set -euo pipefail

MODE="rlcp"
if [[ $# -gt 0 && "$1" != --* ]]; then
  MODE="$1"
  shift
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

MODEL=""
RETENTION=""
RETENTIONS=""
SEEDS="1"
CONFIG_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --retention)
      RETENTION="$2"
      shift 2
      ;;
    --retentions)
      RETENTIONS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash run_experiments.sh [mode] [--model 0.5B|1.5B|7B] [--retention 0.07] [--retentions 0.08,0.07,0.06] [--seeds 3] [--config path]"
      exit 1
      ;;
  esac
done

case "$MODE" in
  baseline|random|gr|beta|rlcp) ;;
  *)
    echo "Unsupported mode: $MODE"
    echo "Use one of: baseline random gr beta rlcp"
    exit 1
    ;;
esac

if [[ -n "$RETENTION" && -n "$RETENTIONS" ]]; then
  echo "Use either --retention or --retentions, not both."
  exit 1
fi

if [[ -n "$MODEL" ]]; then
  MODELS=("$MODEL")
else
  MODELS=("0.5B" "1.5B")
fi

for m in "${MODELS[@]}"; do
  cmd=("$PYTHON_BIN" "retention_sweep/sweep_runner.py" "--model" "$m" "--mode" "$MODE" "--seeds" "$SEEDS")

  if [[ -n "$RETENTION" ]]; then
    cmd+=("--retention" "$RETENTION")
  fi

  if [[ -n "$RETENTIONS" ]]; then
    cmd+=("--retentions" "$RETENTIONS")
  fi

  if [[ -n "$CONFIG_PATH" ]]; then
    cmd+=("--config" "$CONFIG_PATH")
  fi

  echo "[run] ${cmd[*]}"
  "${cmd[@]}"
done

# Optional large run example:
# bash run_experiments.sh rlcp --model 7B --seeds 3
