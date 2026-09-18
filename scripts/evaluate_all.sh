#!/usr/bin/env bash
# Evaluate available epoch checkpoints for the six research algorithms.
# Usage: ./scripts/evaluate_all.sh [--dataset cifar10|celeba] [--epoch 100] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "ERROR: run ./scripts/setup.sh --yes first." >&2; exit 1; }

EPOCH=100
DATASET="cifar10"
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epoch)
      [[ $# -ge 2 ]] || { echo "ERROR: --epoch requires a value" >&2; exit 2; }
      EPOCH="$2"; shift 2 ;;
    --dataset)
      [[ $# -ge 2 ]] || { echo "ERROR: --dataset requires a value" >&2; exit 2; }
      DATASET="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$EPOCH" =~ ^[0-9]+$ ]] || { echo "ERROR: --epoch must be an integer" >&2; exit 2; }
[[ "$DATASET" == "cifar10" || "$DATASET" == "celeba" ]] || {
  echo "ERROR: --dataset must be cifar10 or celeba" >&2; exit 2;
}

AVAILABLE=0
evaluate_one() {
  local algorithm=$1
  local run_dir=$2
  local class_name=$3
  local config=$4
  local checkpoint
  checkpoint="$("$PYTHON" scripts/checkpoint_path.py --run-dir "$run_dir" \
    --class-name "$class_name" --epoch "$EPOCH" 2>/dev/null || true)"
  if [[ ! -f "$checkpoint" ]]; then
    echo "[SKIP] $algorithm epoch-$EPOCH checkpoint not found in: $run_dir/checkpoints"
    return
  fi
  AVAILABLE=$((AVAILABLE + 1))
  echo "[PLAN] $PYTHON evaluate.py --algorithm $algorithm --checkpoint $checkpoint --config $config --make-plots"
  if [[ "$DRY_RUN" == false ]]; then
    "$PYTHON" evaluate.py --algorithm "$algorithm" --checkpoint "$checkpoint" --config "$config" --make-plots
  fi
}

if [[ "$DATASET" == "celeba" ]]; then
  SUFFIX="celeba"
  CONFIG_SUFFIX="celeba64"
else
  SUFFIX="cifar10"
  CONFIG_SUFFIX="full"
fi

evaluate_one fm "results/fm_${SUFFIX}" FlowMatchingAlgorithm "config/fm_${CONFIG_SUFFIX}.json"
evaluate_one fm_lognorm "results/fm_lognorm_${SUFFIX}" FlowMatchingLognormAlgorithm "config/fm_lognorm_${CONFIG_SUFFIX}.json"
evaluate_one mf "results/mf_${SUFFIX}" MeanFlowAlgorithm "config/mf_${CONFIG_SUFFIX}.json"
evaluate_one mf_distill "results/mf_distill_${SUFFIX}" MeanFlowDistillAlgorithm "config/mf_distill_${CONFIG_SUFFIX}.json"
evaluate_one consistency "results/consistency_${SUFFIX}" ConsistencyAlgorithm "config/consistency_${CONFIG_SUFFIX}.json"
evaluate_one reflow "results/reflow_${SUFFIX}" ReflowAlgorithm "config/reflow_${CONFIG_SUFFIX}.json"

if [[ "$AVAILABLE" -eq 0 ]]; then
  echo "ERROR: no epoch-$EPOCH checkpoints were found." >&2
  exit 1
fi
echo "Evaluation workflow found $AVAILABLE checkpoint(s)."
