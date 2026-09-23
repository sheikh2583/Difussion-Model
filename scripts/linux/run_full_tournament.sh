#!/usr/bin/env bash
# Run the complete experiment tournament sequentially on one GPU.
#
# Usage:
#   ./scripts/linux/run_full_tournament.sh --dry-run
#   ./scripts/linux/run_full_tournament.sh
#   ./scripts/linux/run_full_tournament.sh --dataset cifar10
#   ./scripts/linux/run_full_tournament.sh --dataset celeba --batch-size 32
#   ./scripts/linux/run_full_tournament.sh --dataset cifar10 --mode fresh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: project environment not found. Run ./scripts/linux/init.sh first." >&2
  exit 1
fi

DATASET="all"
EPOCH=100
BATCH_SIZE=0
DRY_RUN=false
MODE="continue"
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
    --mode)
      [[ $# -ge 2 ]] || { echo "ERROR: --mode requires a value" >&2; exit 2; }
      MODE="$2"; shift 2 ;;
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
case "$MODE" in continue|fresh) ;; *)
  echo "ERROR: --mode must be continue or fresh" >&2; exit 2 ;;
esac
[[ "$EPOCH" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --epoch must be a positive integer" >&2; exit 2;
}
[[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || {
  echo "ERROR: --batch-size must be 0 or a positive integer" >&2; exit 2;
}

mkdir -p results
LOG_FILE="results/tournament_run_$(date +%Y%m%d_%H%M%S)_pid$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1

source scripts/linux/workflow_guard.sh
IDENTITY_ARGS=(--launcher scripts/linux/run_full_tournament.sh)
for config in config/*_full.json config/*_celeba64.json; do
  [[ -f "$config" ]] && IDENTITY_ARGS+=(--config "$config")
done
workflow_guard_start "run_full_tournament.sh" "$DRY_RUN" "${IDENTITY_ARGS[@]}"
export DIFFUSION_LIFECYCLE_MODE="$MODE"

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

  workflow_guard_verify_source "$DRY_RUN"
  set +e
  "$@"
  local status=$?
  set -e
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

resolve_checkpoint() {
  local run_name=$1 class_name=$2
  "$PYTHON" scripts/checkpoint_path.py \
    --run-dir "results/$run_name" --class-name "$class_name" --epoch "$EPOCH" \
    --planned-mode "$MODE"
}

run_algorithm() {
  local algorithm=$1 config=$2 run_name=$3 class_name=$4 description=$5
  local checkpoint
  checkpoint="$(resolve_checkpoint "$run_name" "$class_name")"
  require_file "$config" "config"
  local train_args=(
    train.py --algorithm "$algorithm" --config "$config"
    --epochs "$EPOCH" --mode "$MODE"
  )
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

  local teacher
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

  teacher="$(resolve_checkpoint "fm_${name}" FlowMatchingAlgorithm)"
  require_file "$teacher" "FM teacher checkpoint"
  run_algorithm consistency "$consistency_config" "consistency_${name}" \
    ConsistencyAlgorithm "[4/7] Consistency Models - $name"
  run_algorithm mf_distill "$mf_distill_config" "mf_distill_${name}" \
    MeanFlowDistillAlgorithm "[5/7] Mean Flow Distillation - $name"

  if [[ "$MODE" == "fresh" && -f "$pairs" ]]; then
    local history_dir="data/history"
    local pair_basename="${pairs##*/}"
    local pair_stem="${pair_basename%.pt}"
    local archived_pairs="$history_dir/${pair_stem}_$(date +%Y%m%d_%H%M%S)_pid$$.pt"
    echo ""
    echo "[preserve] [6/7] Existing Reflow pairs: $pairs -> $archived_pairs"
    if [[ "$DRY_RUN" == false ]]; then
      mkdir -p "$history_dir"
      mv -- "$pairs" "$archived_pairs"
    fi
  fi

  if [[ "$MODE" == "continue" && -f "$pairs" ]]; then
    echo ""
    echo "[skip] [6/7] Reflow pair generation - artifact exists: $pairs"
  else
    # The generator validates the inherited suite lock token.
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
if [[ "$MODE" == "continue" ]]; then
  echo "Continue mode: completed epoch-$EPOCH checkpoints and existing pair artifacts will be skipped."
else
  echo "Fresh mode: existing runs are preserved by train.py and Reflow pairs are archived before regeneration."
fi

if [[ "$DATASET" == "all" || "$DATASET" == "cifar10" ]]; then
  run_dataset cifar10
fi
if [[ "$DATASET" == "all" || "$DATASET" == "celeba" ]]; then
  run_dataset celeba
fi

if [[ "$DATASET" == "all" ]]; then DATASETS=(cifar10 celeba); else DATASETS=("$DATASET"); fi
for name in "${DATASETS[@]}"; do
  run_step true "Evaluate epoch-$EPOCH checkpoints - $name" \
    bash scripts/linux/evaluate_all.sh --dataset "$name" --epoch "$EPOCH"
done
run_step false "Aggregate tournament results" \
  "$PYTHON" scripts/aggregate_results.py

echo ""
echo "Tournament workflow complete. Log: $LOG_FILE"
