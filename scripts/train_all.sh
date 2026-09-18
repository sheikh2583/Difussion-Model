#!/usr/bin/env bash
# Train the supported algorithm suite in dependency order.
# Usage:
#   ./scripts/train_all.sh [--dataset cifar10|celeba] [--only ALGORITHM]
#                          [--skip-ALGORITHM] [--mode continue|fresh]
#                          [--batch-size N] [--checkpoint-every N]
#                          [--train-only] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/venv/bin/python"
elif [[ -f "$PROJECT_ROOT/venv/Scripts/python.exe" ]]; then
  # Git Bash on Windows; native Linux clones use venv/bin/python.
  PYTHON="$PROJECT_ROOT/venv/Scripts/python.exe"
else
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
MODE="continue"
BATCH_SIZE=""
CHECKPOINT_EVERY=10
TRAIN_ONLY=false

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
    --mode)
      [[ $# -ge 2 ]] || { echo "ERROR: --mode requires a value" >&2; exit 2; }
      MODE="$2"; shift 2 ;;
    --batch-size)
      [[ $# -ge 2 ]] || { echo "ERROR: --batch-size requires a value" >&2; exit 2; }
      BATCH_SIZE="$2"; shift 2 ;;
    --checkpoint-every)
      [[ $# -ge 2 ]] || { echo "ERROR: --checkpoint-every requires a value" >&2; exit 2; }
      CHECKPOINT_EVERY="$2"; shift 2 ;;
    --train-only) TRAIN_ONLY=true; shift ;;
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

case "$MODE" in
  continue|fresh) ;;
  *) echo "ERROR: --mode must be continue or fresh" >&2; exit 2 ;;
esac

if [[ "$DATASET" == "celeba" ]]; then
  FM_CONFIG="config/fm_celeba64.json"
  FM_LOGNORM_CONFIG="config/fm_lognorm_celeba64.json"
  MF_CONFIG="config/mf_celeba64.json"
  MF_DISTILL_CONFIG="config/mf_distill_celeba64.json"
  CONSISTENCY_CONFIG="config/consistency_celeba64.json"
  REFLOW_CONFIG="config/reflow_celeba64.json"
  FM_RUN_DIR="results/fm_celeba"
  REFLOW_PAIRS="data/reflow_pairs_celeba.pt"
else
  FM_CONFIG="config/fm_full.json"
  FM_LOGNORM_CONFIG="config/fm_lognorm_full.json"
  MF_CONFIG="config/mf_full.json"
  MF_DISTILL_CONFIG="config/mf_distill_full.json"
  CONSISTENCY_CONFIG="config/consistency_full.json"
  REFLOW_CONFIG="config/reflow_full.json"
  FM_RUN_DIR="results/fm_cifar10"
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
  local command=(
    "$PYTHON" train.py --algorithm "$algorithm" --config "$config"
    --mode "$MODE" --checkpoint-every "$CHECKPOINT_EVERY"
  )
  [[ -z "$BATCH_SIZE" ]] || command+=(--batch-size "$BATCH_SIZE")
  [[ "$TRAIN_ONLY" == false ]] || command+=(--train-only)
  echo "[PLAN] ${command[*]}"
  if [[ "$DRY_RUN" == false ]]; then
    "${command[@]}"
  fi
}

require_file() {
  local path=$1
  local purpose=$2
  if [[ ! -f "$path" ]]; then
    if [[ "$DRY_RUN" == true ]]; then
      echo "[PLANNED] $purpose: $path"
      return 0
    fi
    echo "[BLOCKED] $purpose: $path" >&2
    FAILED=1
    return 1
  fi
  echo "[OK] $purpose: $path"
}

run_training fm "$FM_CONFIG" "$SKIP_FM"
run_training fm_lognorm "$FM_LOGNORM_CONFIG" "$SKIP_FM_LOGNORM"
run_training mf "$MF_CONFIG" "$SKIP_MF"
if [[ "$DRY_RUN" == true ]]; then
  # Nothing has been created during a dry run, so resolve the path that the
  # selected lifecycle mode would create.
  FM_CKPT="$("$PYTHON" scripts/checkpoint_path.py --run-dir "$FM_RUN_DIR" \
    --class-name FlowMatchingAlgorithm --epoch 100 --planned-mode "$MODE")"
else
  # FM has already run by this point. Resolve the checkpoint that now exists;
  # using --planned-mode fresh here would incorrectly advance to run_(N+1).
  FM_CKPT="$("$PYTHON" scripts/checkpoint_path.py --run-dir "$FM_RUN_DIR" \
    --class-name FlowMatchingAlgorithm --epoch 100)"
fi

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
  if [[ "$MODE" == "fresh" && -f "$REFLOW_PAIRS" ]]; then
    HISTORY_DIR="data/history"
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    PAIR_BASENAME="${REFLOW_PAIRS##*/}"
    PAIR_STEM="${PAIR_BASENAME%.pt}"
    ARCHIVED_PAIRS="$HISTORY_DIR/${PAIR_STEM}_${TIMESTAMP}_pid$$.pt"
    echo "[PLAN] Preserve existing Reflow pairs: $REFLOW_PAIRS -> $ARCHIVED_PAIRS"
    if [[ "$DRY_RUN" == false ]]; then
      mkdir -p "$HISTORY_DIR"
      mv -- "$REFLOW_PAIRS" "$ARCHIVED_PAIRS"
    fi
  fi
  if [[ "$MODE" == "fresh" || ! -f "$REFLOW_PAIRS" ]]; then
    READY=true
    require_file "$FM_CKPT" "FM teacher checkpoint" || READY=false
    if [[ "$READY" == true || "$DRY_RUN" == true ]]; then
      echo "[PLAN] Generate Reflow pairs: $REFLOW_PAIRS"
      if [[ "$DRY_RUN" == false ]]; then
        "$PYTHON" scripts/generate_reflow_pairs.py \
          --checkpoint "$FM_CKPT" --config "$FM_CONFIG" \
          --n-pairs 50000 --nfe 50 --output "$REFLOW_PAIRS"
      fi
    fi
  fi
  READY=true
  require_file "$REFLOW_PAIRS" "Reflow pairs" || READY=false
  [[ "$READY" == false ]] || run_training reflow "$REFLOW_CONFIG" false
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
