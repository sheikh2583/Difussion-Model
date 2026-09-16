#!/usr/bin/env bash
# Train the supported algorithm suite in dependency order.
# Usage:
#   ./scripts/train_all.sh [--dataset cifar10|celeba] [--only ALGORITHM]
#                          [--skip-ALGORITHM] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: project environment not found. Run ./scripts/setup.sh --yes first." >&2
  exit 1
fi

SKIP_FM=false
SKIP_FM_LOGNORM=false
SKIP_MF=false
SKIP_MF_DISTILL=false
SKIP_CONSISTENCY=false
SKIP_REFLOW=false
ONLY=""
DATASET="cifar10"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-fm)          SKIP_FM=true; shift ;;
    --skip-fm-lognorm)  SKIP_FM_LOGNORM=true; shift ;;
    --skip-mf)          SKIP_MF=true; shift ;;
    --skip-mf-distill)  SKIP_MF_DISTILL=true; shift ;;
    --skip-consistency) SKIP_CONSISTENCY=true; shift ;;
    --skip-reflow)      SKIP_REFLOW=true; shift ;;
    --only)
      [[ $# -ge 2 ]] || { echo "ERROR: --only requires a value" >&2; exit 2; }
      ONLY="$2"; shift 2 ;;
    --dataset)
      [[ $# -ge 2 ]] || { echo "ERROR: --dataset requires a value" >&2; exit 2; }
      DATASET="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help)
      sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'
      exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$DATASET" in
  cifar10|celeba) ;;
  *) echo "ERROR: --dataset must be cifar10 or celeba" >&2; exit 2 ;;
esac

case "$ONLY" in
  ""|fm|fm_lognorm|mf|mf_distill|consistency|reflow) ;;
  *) echo "ERROR: unknown algorithm for --only: $ONLY" >&2; exit 2 ;;
esac

if [[ "$DATASET" == "celeba" ]]; then
  FM_CONFIG="config/fm_celeba64.json"
  FM_LOGNORM_CONFIG="config/fm_lognorm_celeba64.json"
  MF_CONFIG="config/mf_celeba64.json"
  MF_DISTILL_CONFIG="config/mf_distill_celeba64.json"
  CONSISTENCY_CONFIG="config/consistency_celeba64.json"
  REFLOW_CONFIG="config/reflow_celeba64.json"
  FM_CKPT="results/fm_celeba/checkpoints/FlowMatchingAlgorithm_epoch100.pt"
  REFLOW_PAIRS="data/reflow_pairs_celeba.pt"
else
  FM_CONFIG="config/fm_full.json"
  FM_LOGNORM_CONFIG="config/fm_lognorm_full.json"
  MF_CONFIG="config/mf_full.json"
  MF_DISTILL_CONFIG="config/mf_distill_full.json"
  CONSISTENCY_CONFIG="config/consistency_full.json"
  REFLOW_CONFIG="config/reflow_full.json"
  FM_CKPT="results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt"
  REFLOW_PAIRS="data/reflow_pairs_cifar10.pt"
fi

FAILED=0

run_training() {
  local algorithm=$1
  local config=$2
  local skip=$3
  [[ "$skip" == false ]] || return 0
  [[ -z "$ONLY" || "$ONLY" == "$algorithm" ]] || return 0
  if [[ ! -f "$config" ]]; then
    echo "[MISSING] config: $config" >&2
    FAILED=1
    return 0
  fi
  echo "[PLAN] $PYTHON train.py --algorithm $algorithm --config $config"
  if [[ "$DRY_RUN" == false ]]; then
    "$PYTHON" train.py --algorithm "$algorithm" --config "$config"
  fi
}

require_file() {
  local path=$1
  local purpose=$2
  if [[ ! -f "$path" ]]; then
    echo "[BLOCKED] $purpose: $path" >&2
    FAILED=1
    return 1
  fi
  echo "[OK] $purpose: $path"
}

run_training fm "$FM_CONFIG" "$SKIP_FM"
run_training fm_lognorm "$FM_LOGNORM_CONFIG" "$SKIP_FM_LOGNORM"
run_training mf "$MF_CONFIG" "$SKIP_MF"

if [[ "$SKIP_MF_DISTILL" == false && ( -z "$ONLY" || "$ONLY" == "mf_distill" ) ]]; then
  READY=true
  require_file "$FM_CKPT" "FM teacher checkpoint" || READY=false
  if [[ "$READY" == true || "$DRY_RUN" == true ]]; then
    run_training mf_distill "$MF_DISTILL_CONFIG" false
  fi
fi
if [[ "$SKIP_CONSISTENCY" == false && ( -z "$ONLY" || "$ONLY" == "consistency" ) ]]; then
  READY=true
  require_file "$FM_CKPT" "FM teacher checkpoint" || READY=false
  if [[ "$READY" == true || "$DRY_RUN" == true ]]; then
    run_training consistency "$CONSISTENCY_CONFIG" false
  fi
fi
if [[ "$SKIP_REFLOW" == false && ( -z "$ONLY" || "$ONLY" == "reflow" ) ]]; then
  READY=true
  require_file "$REFLOW_PAIRS" "Reflow pairs" || READY=false
  if [[ "$READY" == true || "$DRY_RUN" == true ]]; then
    run_training reflow "$REFLOW_CONFIG" false
  fi
fi

if [[ "$FAILED" -ne 0 ]]; then
  echo "Workflow validation found missing prerequisites." >&2
  exit 1
fi
if [[ "$DRY_RUN" == true ]]; then
  echo "Dry-run validation complete."
else
  echo "Training workflow complete."
fi
