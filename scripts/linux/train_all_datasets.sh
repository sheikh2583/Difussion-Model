#!/usr/bin/env bash
# Unattended, dependency-aware CIFAR-10 + CelebA training with per-model logs.
# Jobs inside a suite remain serialized in dependency order.
set -euo pipefail

PLATFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$PLATFORM_DIR")")"
cd "$PROJECT_ROOT"

DATASET="all"
MODE="continue"
CHECKPOINT_EVERY=10
TRAIN_ONLY=false
DRY_RUN=false
CIFAR_BACKBONE="current"
MACHINE_LABEL="${DIFFUSION_MACHINE_LABEL:-}"
LOG_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      [[ $# -ge 2 ]] || { echo "ERROR: --dataset requires a value" >&2; exit 2; }
      DATASET="$2"; shift 2 ;;
    --mode)
      [[ $# -ge 2 ]] || { echo "ERROR: --mode requires a value" >&2; exit 2; }
      MODE="$2"; shift 2 ;;
    --checkpoint-every)
      [[ $# -ge 2 ]] || { echo "ERROR: --checkpoint-every requires a value" >&2; exit 2; }
      CHECKPOINT_EVERY="$2"; shift 2 ;;
    --train-only) TRAIN_ONLY=true; shift ;;
    --cifar-backbone)
      [[ $# -ge 2 ]] || { echo "ERROR: --cifar-backbone requires a value" >&2; exit 2; }
      CIFAR_BACKBONE="$2"; shift 2 ;;
    --machine-label)
      [[ $# -ge 2 ]] || { echo "ERROR: --machine-label requires a value" >&2; exit 2; }
      MACHINE_LABEL="$2"; shift 2 ;;
    --log-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --log-dir requires a value" >&2; exit 2; }
      LOG_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help)
      sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'
      exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$DATASET" in
  all|cifar10|celeba) ;;
  *) echo "ERROR: --dataset must be all, cifar10, or celeba" >&2; exit 2 ;;
esac
case "$MODE" in
  continue|fresh) ;;
  *) echo "ERROR: --mode must be continue or fresh" >&2; exit 2 ;;
esac
case "$CIFAR_BACKBONE" in
  current|legacy) ;;
  *) echo "ERROR: --cifar-backbone must be current or legacy" >&2; exit 2 ;;
esac
[[ "$CHECKPOINT_EVERY" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --checkpoint-every must be a positive integer" >&2
  exit 2
}

if [[ "$DATASET" == "all" ]]; then
  DATASETS=(cifar10 celeba)
else
  DATASETS=("$DATASET")
fi
ALGORITHMS=(fm fm_lognorm mf mf_distill consistency reflow)
declare -A ALGORITHM_CLASSES=(
  [fm]="FlowMatchingAlgorithm"
  [fm_lognorm]="FlowMatchingLognormAlgorithm"
  [mf]="MeanFlowAlgorithm"
  [mf_distill]="MeanFlowDistillAlgorithm"
  [consistency]="ConsistencyAlgorithm"
  [reflow]="ReflowAlgorithm"
)

PYTHON="$PROJECT_ROOT/venv/bin/python"
[[ -x "$PYTHON" ]] || {
  echo "ERROR: project environment not found. Run ./scripts/linux/init.sh first." >&2
  exit 1
}
source scripts/linux/workflow_guard.sh
IDENTITY_ARGS=(
  --launcher scripts/linux/train_all_datasets.sh
  --launcher scripts/linux/train_all.sh
)
if [[ -n "${DIFFUSION_ENTRY_LAUNCHER:-}" ]]; then
  IDENTITY_ARGS+=(--launcher "$DIFFUSION_ENTRY_LAUNCHER")
fi
for config in config/*_full.json config/*_celeba64.json \
  config/mf_v3_exact_jvp_b128.json config/cifar_legacy/*.json; do
  [[ -f "$config" ]] && IDENTITY_ARGS+=(--config "$config")
done
workflow_guard_start "train_all_datasets.sh" "$DRY_RUN" "${IDENTITY_ARGS[@]}"
export DIFFUSION_LIFECYCLE_MODE="$MODE"

RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST_TOKEN="$(hostname | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-')"
HOST_TOKEN="${HOST_TOKEN%-}"
GPU_NAME=""
GPU_MEMORY_MB=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_QUERY=""
  MEMORY_QUERY=""
  if GPU_QUERY="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"; then
    GPU_NAME="$(printf '%s\n' "$GPU_QUERY" | sed -n '1p')"
    if MEMORY_QUERY="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)"; then
      GPU_MEMORY_MB="$(printf '%s\n' "$MEMORY_QUERY" | sed -n '1p' | tr -d ' ')"
    fi
  fi
fi
if [[ -n "$GPU_NAME" ]]; then
  GPU_TOKEN="$(printf '%s' "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-')"
  GPU_TOKEN="${GPU_TOKEN%-}"
  if [[ "$GPU_MEMORY_MB" =~ ^[0-9]+$ ]]; then
    GPU_TOKEN="${GPU_TOKEN}-$(((GPU_MEMORY_MB + 1023) / 1024))gb"
  fi
  DEVICE_LOG_DIR="training_logs/$GPU_TOKEN"
else
  DEVICE_LOG_DIR="training_logs/cpu-or-unknown"
fi
PIXEL_LOG_DIR="${LOG_DIR:-$DEVICE_LOG_DIR/pixel}"
LOG_DEVICE_TOKEN="${GPU_TOKEN:-cpu-or-unknown}"
if [[ -z "$MACHINE_LABEL" ]]; then
  OS_TOKEN="$(uname -s | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-')"
  OS_TOKEN="${OS_TOKEN%-}"
  if [[ -n "$GPU_NAME" ]]; then
    MACHINE_LABEL="${OS_TOKEN}-${HOST_TOKEN}-${GPU_TOKEN}"
  else
    MACHINE_LABEL="${OS_TOKEN}-${HOST_TOKEN}-gpu-unavailable"
  fi
fi
export DIFFUSION_MACHINE_LABEL="$MACHINE_LABEL"
echo "[PLAN] machine_label=$MACHINE_LABEL"
if [[ "$DRY_RUN" == false ]]; then
  mkdir -p "$PIXEL_LOG_DIR"
fi
FAILURES=()

selected_config() {
  local dataset=$1 algorithm=$2
  if [[ "$dataset" == "celeba" ]]; then
    printf 'config/%s_celeba64.json\n' "$algorithm"
  elif [[ "$CIFAR_BACKBONE" == "legacy" ]]; then
    printf 'config/cifar_legacy/%s.json\n' "$algorithm"
  elif [[ "$algorithm" == "mf" ]]; then
    printf 'config/mf_v3_exact_jvp_b128.json\n'
  else
    printf 'config/%s_full.json\n' "$algorithm"
  fi
}

config_run_dir() {
  "$PYTHON" -c '
from pathlib import Path
import sys
from config.config import ExperimentConfig
from utils.run_lifecycle import run_directory
root = Path.cwd().resolve()
path = run_directory(ExperimentConfig.load(sys.argv[1]), root)
try:
    print(path.relative_to(root))
except ValueError:
    print(path)
' "$1"
}

for dataset in "${DATASETS[@]}"; do
  for algorithm in "${ALGORITHMS[@]}"; do
    config_path="$(selected_config "$dataset" "$algorithm")"
    command=(
      "$PROJECT_ROOT/scripts/linux/train_all.sh"
      --dataset "$dataset"
      --only "$algorithm"
      --mode "$MODE"
      --checkpoint-every "$CHECKPOINT_EVERY"
    )
    if [[ "$dataset" == "cifar10" ]]; then
      command+=(--cifar-backbone "$CIFAR_BACKBONE")
    fi
    command+=(--machine-label "$MACHINE_LABEL")
    [[ "$TRAIN_ONLY" == false ]] || command+=(--train-only)
    [[ "$DRY_RUN" == false ]] || command+=(--dry-run)
    if [[ "$dataset" == "cifar10" && "$algorithm" == "mf" && "$CIFAR_BACKBONE" == "current" ]]; then
      command+=(--mf-config config/mf_v3_exact_jvp_b128.json)
    fi

    run_dir="$(config_run_dir "$config_path")"
    planned_checkpoint="$(
      "$PYTHON" scripts/checkpoint_path.py \
        --run-dir "$run_dir" \
        --class-name "${ALGORITHM_CLASSES[$algorithm]}" \
        --epoch 100 --planned-mode "$MODE"
    )"
    export DIFFUSION_CHECKPOINT_SERIES
    DIFFUSION_CHECKPOINT_SERIES="$(basename "$(dirname "$planned_checkpoint")")"
    echo "[PLAN] dataset=$dataset algorithm=$algorithm checkpoint_series=$DIFFUSION_CHECKPOINT_SERIES"
    job_log_dir="$PIXEL_LOG_DIR/$dataset/$algorithm"
    log_relative="$job_log_dir/${dataset}_pixel_${algorithm}_${DIFFUSION_CHECKPOINT_SERIES}_${HOST_TOKEN}_${RUN_TIMESTAMP}.log"
    echo "[PLAN] log=$log_relative"

    if [[ "$DRY_RUN" == true ]]; then
      echo "[DRY-RUN] ${command[*]}"
      "${command[@]}"
      continue
    fi

    workflow_guard_verify_source "$DRY_RUN"

    mkdir -p "$job_log_dir"
    {
      echo "[run] training_type=pixel_diffusion"
      echo "[run] representation_space=pixel"
      echo "[run] dataset=$dataset"
      echo "[run] algorithm=$algorithm"
      "$PYTHON" scripts/print_run_provenance.py \
        --config "$config_path" \
        --machine-label "$MACHINE_LABEL" \
        --log-device-token "$LOG_DEVICE_TOKEN" \
        --log-path "$log_relative"
      echo "[run] started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "[run] command=${command[*]}"
    } | tee "$log_relative"

    set +e
    "${command[@]}" 2>&1 | tee -a "$log_relative"
    status=${PIPESTATUS[0]}
    set -e
    {
      echo "[run] finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "[run] exit_status=$status"
    } | tee -a "$log_relative"
    "$PYTHON" scripts/write_training_log_metadata.py --log "$log_relative" >/dev/null
    echo "[LOG] $log_relative"
    echo "[METADATA] ${log_relative}.meta.json"
    if [[ "$status" -ne 0 ]]; then
      FAILURES+=("${dataset}:${algorithm}:${status}")
      echo "WARNING: $dataset/$algorithm failed; continuing to the next model." >&2
    fi
  done
done

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry-run complete; no logs or model artifacts were created."
  exit 0
fi

if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo "Completed with failed jobs: ${FAILURES[*]}" >&2
  exit 1
fi
echo "All requested CIFAR-10 and CelebA jobs completed successfully."
