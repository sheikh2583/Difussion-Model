#!/usr/bin/env bash
# Evaluate available epoch checkpoints for the six research algorithms.
# Usage: ./scripts/evaluate_all.sh [--epoch 100] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "ERROR: run ./scripts/setup.sh --yes first." >&2; exit 1; }

EPOCH=100
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epoch)
      [[ $# -ge 2 ]] || { echo "ERROR: --epoch requires a value" >&2; exit 2; }
      EPOCH="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$EPOCH" =~ ^[0-9]+$ ]] || { echo "ERROR: --epoch must be an integer" >&2; exit 2; }

AVAILABLE=0
evaluate_one() {
  local algorithm=$1
  local checkpoint=$2
  local config=$3
  if [[ ! -f "$checkpoint" ]]; then
    echo "[SKIP] $algorithm checkpoint not found: $checkpoint"
    return
  fi
  AVAILABLE=$((AVAILABLE + 1))
  echo "[PLAN] $PYTHON evaluate.py --algorithm $algorithm --checkpoint $checkpoint --config $config --make-plots"
  if [[ "$DRY_RUN" == false ]]; then
    "$PYTHON" evaluate.py --algorithm "$algorithm" --checkpoint "$checkpoint" --config "$config" --make-plots
  fi
}

evaluate_one fm "results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch${EPOCH}.pt" "config/fm_full.json"
evaluate_one fm_lognorm "results/fm_lognorm_cifar10/checkpoints/FlowMatchingLognormAlgorithm_epoch${EPOCH}.pt" "config/fm_lognorm_full.json"
evaluate_one mf "results/mf_cifar10/checkpoints/MeanFlowAlgorithm_epoch${EPOCH}.pt" "config/mf_full.json"
evaluate_one mf_distill "results/mf_distill_cifar10/checkpoints/MeanFlowDistillAlgorithm_epoch${EPOCH}.pt" "config/mf_distill_full.json"
evaluate_one consistency "results/consistency_cifar10/checkpoints/ConsistencyAlgorithm_epoch${EPOCH}.pt" "config/consistency_full.json"
evaluate_one reflow "results/reflow_cifar10/checkpoints/ReflowAlgorithm_epoch${EPOCH}.pt" "config/reflow_full.json"

if [[ "$AVAILABLE" -eq 0 ]]; then
  echo "ERROR: no epoch-$EPOCH checkpoints were found." >&2
  exit 1
fi
echo "Evaluation workflow found $AVAILABLE checkpoint(s)."
