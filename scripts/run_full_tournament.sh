#!/usr/bin/env bash
# Run the complete experiment tournament sequentially on one GPU.
#
# Usage:
#   ./scripts/run_full_tournament.sh --dry-run
#   ./scripts/run_full_tournament.sh
#   ./scripts/run_full_tournament.sh --dataset cifar10
#   ./scripts/run_full_tournament.sh --dataset celeba --batch-size 32
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/venv/bin/python"
elif [[ -f "$PROJECT_ROOT/venv/Scripts/python.exe" ]]; then
  # Also support Git Bash on Windows; normal Linux clones use venv/bin/python.
  PYTHON="$PROJECT_ROOT/venv/Scripts/python.exe"
else
  echo "ERROR: project environment not found. Run ./init_all.sh first." >&2
  exit 1
fi

DATASET="all"
EPOCH=100
BATCH_SIZE=0
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      [[ $# -ge 2 ]] || { echo "ERROR: --dataset requires a value" >&2; exit 2; }
      DATASET="$2"; shift 2 ;;
    --epoch)
      [[ $# -ge 2 ]] || { echo "ERROR: --epoch requires a value" >&2; exit 2; }
      EPOCH="$2"; shift 2 ;;
    --batch-size)
      [[ $# -ge 2 ]] || { echo "ERROR: --batch-size requires a value" >&2; exit 2; }
      BATCH_SIZE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help)
      sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'
      exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$DATASET" in all|cifar10|celeba) ;; *)
  echo "ERROR: --dataset must be all, cifar10, or celeba" >&2; exit 2 ;;
esac
[[ "$EPOCH" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --epoch must be a positive integer" >&2; exit 2;
}
[[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || {
  echo "ERROR: --batch-size must be 0 or a positive integer" >&2; exit 2;
}

mkdir -p results
LOG_FILE="results/tournament_run_$(date +%Y%m%d_%H%M%S)_pid$$.log"
LOCK_FILE="results/.lock"
LOCK_TOKEN=""
LOCK_OWNED=false
exec > >(tee -a "$LOG_FILE") 2>&1

release_lock() {
  if [[ "$LOCK_OWNED" == true && -f "$LOCK_FILE" ]]; then
    local current
    current="$(cat "$LOCK_FILE")"
    if [[ "$current" == "$LOCK_TOKEN" ]]; then
      rm -f -- "$LOCK_FILE"
    fi
  fi
  LOCK_OWNED=false
}
trap release_lock EXIT INT TERM

acquire_lock() {
  local description=$1
  LOCK_TOKEN="pid=$$;command=run_full_tournament.sh;step=$description"
  if ! (set -o noclobber; printf '%s\n' "$LOCK_TOKEN" > "$LOCK_FILE") 2>/dev/null; then
    echo "ERROR: GPU lock already exists: $LOCK_FILE" >&2
    echo "Wait for the active job, or remove it only after confirming it is stale." >&2
    exit 1
  fi
  LOCK_OWNED=true
}

run_step() {
  local use_lock=$1 description=$2
  shift 2
  echo ""
  echo "=============================================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $description"
  printf '  '
  printf '%q ' "$@"
  echo ""
  echo "=============================================================="
  if [[ "$DRY_RUN" == true ]]; then
    echo "  (dry-run, not executing)"
    return 0
  fi

  if [[ "$use_lock" == true ]]; then acquire_lock "$description"; fi
  set +e
  "$@"
  local status=$?
  set -e
  if [[ "$use_lock" == true ]]; then release_lock; fi
  if [[ $status -ne 0 ]]; then
    echo "ERROR: step failed with exit code $status: $description" >&2
    return "$status"
  fi
}

require_file() {
  local path=$1 purpose=$2
  if [[ -f "$path" ]]; then return 0; fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "[planned prerequisite] $purpose: $path"
    return 0
  fi
  echo "ERROR: missing $purpose: $path" >&2
  exit 1
}

run_algorithm() {
  local algorithm=$1 config=$2 run_name=$3 class_name=$4 description=$5
  local checkpoint="results/${run_name}/checkpoints/${class_name}_epoch${EPOCH}.pt"
  if [[ -f "$checkpoint" ]]; then
    echo ""
    echo "[skip] $description - checkpoint already exists: $checkpoint"
    return 0
  fi
  require_file "$config" "config"
  local train_args=(train.py --algorithm "$algorithm" --config "$config")
  if [[ "$BATCH_SIZE" -gt 0 ]]; then
    train_args+=(--batch-size "$BATCH_SIZE")
  fi
  run_step true "$description" "$PYTHON" "${train_args[@]}"
}

run_dataset() {
  local name=$1
  local fm_config fm_lognorm_config mf_config consistency_config
  local mf_distill_config reflow_config
  if [[ "$name" == "celeba" ]]; then
    fm_config="config/fm_celeba64.json"
    fm_lognorm_config="config/fm_lognorm_celeba64.json"
    mf_config="config/mf_celeba64.json"
    consistency_config="config/consistency_celeba64.json"
    mf_distill_config="config/mf_distill_celeba64.json"
    reflow_config="config/reflow_celeba64.json"
  else
    fm_config="config/fm_full.json"
    fm_lognorm_config="config/fm_lognorm_full.json"
    mf_config="config/mf_full.json"
    consistency_config="config/consistency_full.json"
    mf_distill_config="config/mf_distill_full.json"
    reflow_config="config/reflow_full.json"
  fi

  local teacher="results/fm_${name}/checkpoints/FlowMatchingAlgorithm_epoch${EPOCH}.pt"
  local pairs="data/reflow_pairs_${name}.pt"

  echo ""
  echo "##############################################################"
  echo "# DATASET: $name"
  echo "##############################################################"

  run_algorithm fm "$fm_config" "fm_${name}" FlowMatchingAlgorithm \
    "[1/7] Flow Matching - $name"
  run_algorithm fm_lognorm "$fm_lognorm_config" "fm_lognorm_${name}" \
    FlowMatchingLognormAlgorithm "[2/7] FM + Logit-Normal - $name"
  run_algorithm mf "$mf_config" "mf_${name}" MeanFlowAlgorithm \
    "[3/7] Mean Flow - $name"

  require_file "$teacher" "FM teacher checkpoint"
  run_algorithm consistency "$consistency_config" "consistency_${name}" \
    ConsistencyAlgorithm "[4/7] Consistency Models - $name"
  run_algorithm mf_distill "$mf_distill_config" "mf_distill_${name}" \
    MeanFlowDistillAlgorithm "[5/7] Mean Flow Distillation - $name"

  if [[ -f "$pairs" ]]; then
    echo ""
    echo "[skip] [6/7] Reflow pair generation - artifact exists: $pairs"
  else
    # The generator acquires and releases results/.lock itself.
    run_step false "[6/7] Reflow pair generation - $name" \
      "$PYTHON" scripts/generate_reflow_pairs.py \
      --checkpoint "$teacher" --config "$fm_config" \
      --n-pairs 50000 --nfe 50 --output "$pairs"
  fi

  require_file "$pairs" "Reflow pair artifact"
  run_algorithm reflow "$reflow_config" "reflow_${name}" ReflowAlgorithm \
    "[7/7] Rectified Flow Reflow - $name"
}

echo "Logging to $LOG_FILE"
echo "Completed epoch-$EPOCH checkpoints and existing pair artifacts will be skipped."

if [[ "$DATASET" == "all" || "$DATASET" == "cifar10" ]]; then
  run_dataset cifar10
fi
if [[ "$DATASET" == "all" || "$DATASET" == "celeba" ]]; then
  run_dataset celeba
fi

if [[ "$DATASET" == "all" ]]; then DATASETS=(cifar10 celeba); else DATASETS=("$DATASET"); fi
for name in "${DATASETS[@]}"; do
  run_step true "Evaluate epoch-$EPOCH checkpoints - $name" \
    bash scripts/evaluate_all.sh --dataset "$name" --epoch "$EPOCH"
done
run_step false "Aggregate tournament results" \
  "$PYTHON" scripts/aggregate_results.py

echo ""
echo "Tournament workflow complete. Log: $LOG_FILE"
