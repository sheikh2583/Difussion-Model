#!/usr/bin/env bash
# Train the complete CelebA latent suite with one timestamped log per GPU job.
#
# The six registered latent experiments are:
#   fm, fm_lognorm, mf, mf_distill, consistency, and reflow.
#
# FM is trained first because MF-Distill, Consistency, and Reflow depend on its
# epoch-100 checkpoint. Reflow pairs are generated automatically when missing.
# Jobs are intentionally serialized because they share one GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

MODE="continue"
ONLY="all"
MACHINE_LABEL="${DIFFUSION_MACHINE_LABEL:-}"
CHECKPOINT_EVERY=""
TRAIN_ONLY=false
DRY_RUN=false
LOG_DIR=""
TRACK_LOGS=true

usage() {
  sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'
  cat <<'EOF'

Usage:
  ./scripts/linux/train_celeba_latent.sh [options]

Options:
  --only NAME              all | fm | fm_lognorm | mf | mf_distill |
                           consistency | reflow (default: all)
  --mode MODE              fresh | continue (default: continue)
  --machine-label NAME     Label stored in run metadata
  --checkpoint-every N     Override checkpoint cadence from each config
  --train-only             Skip evaluation and final sampling
  --log-dir PATH           Override the auto-detected device log directory
  --track-logs             Stage each completed log and sidecar (default)
  --no-track-logs          Leave completed logs unstaged
  --dry-run                Print commands without training or writing logs
  --list                   List the six algorithms and exit
  -h, --help               Show this help

Examples:
  ./scripts/linux/train_celeba_latent.sh --dry-run
  ./scripts/linux/train_celeba_latent.sh --only fm \
    --machine-label NDAG-M-Lab-RTX3090 --mode fresh
  ./scripts/linux/train_celeba_latent.sh --mode continue
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      [[ $# -ge 2 ]] || { echo "ERROR: --only requires a value" >&2; exit 2; }
      ONLY="$2"; shift 2 ;;
    --mode)
      [[ $# -ge 2 ]] || { echo "ERROR: --mode requires a value" >&2; exit 2; }
      MODE="$2"; shift 2 ;;
    --machine-label)
      [[ $# -ge 2 ]] || { echo "ERROR: --machine-label requires a value" >&2; exit 2; }
      MACHINE_LABEL="$2"; shift 2 ;;
    --checkpoint-every)
      [[ $# -ge 2 ]] || { echo "ERROR: --checkpoint-every requires a value" >&2; exit 2; }
      CHECKPOINT_EVERY="$2"; shift 2 ;;
    --train-only) TRAIN_ONLY=true; shift ;;
    --log-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --log-dir requires a value" >&2; exit 2; }
      LOG_DIR="$2"; shift 2 ;;
    --track-logs) TRACK_LOGS=true; shift ;;
    --no-track-logs) TRACK_LOGS=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --list)
      printf '%s\n' fm fm_lognorm mf mf_distill consistency reflow
      exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODE" in
  fresh|continue) ;;
  *) echo "ERROR: --mode must be fresh or continue" >&2; exit 2 ;;
esac
case "$ONLY" in
  all|fm|fm_lognorm|mf|mf_distill|consistency|reflow) ;;
  *) echo "ERROR: unsupported --only value: $ONLY" >&2; exit 2 ;;
esac
if [[ -n "$CHECKPOINT_EVERY" && ! "$CHECKPOINT_EVERY" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --checkpoint-every must be a positive integer" >&2
  exit 2
fi

PYTHON="$PROJECT_ROOT/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: project Python not found or not executable: $PYTHON" >&2
  exit 1
fi

declare -A CONFIGS=(
  [fm]="config/fm_celeba_latent.json"
  [fm_lognorm]="config/fm_lognorm_celeba_latent.json"
  [mf]="config/mf_celeba_latent.json"
  [mf_distill]="config/mf_distill_celeba_latent.json"
  [consistency]="config/consistency_celeba_latent.json"
  [reflow]="config/reflow_celeba_latent.json"
)
declare -A EFFECTIVE_CONFIGS=(
  [fm]="config/fm_celeba_latent.json"
  [fm_lognorm]="config/fm_lognorm_celeba_latent.json"
  [mf]="config/mf_celeba_latent.json"
  [mf_distill]="config/mf_distill_celeba_latent.json"
  [consistency]="config/consistency_celeba_latent.json"
  [reflow]="config/reflow_celeba_latent.json"
)
declare -A ALGORITHM_CLASSES=(
  [fm]="FlowMatchingAlgorithm"
  [fm_lognorm]="FlowMatchingLognormAlgorithm"
  [mf]="MeanFlowAlgorithm"
  [mf_distill]="MeanFlowDistillAlgorithm"
  [consistency]="ConsistencyAlgorithm"
  [reflow]="ReflowAlgorithm"
)
ALGORITHMS=(fm fm_lognorm mf mf_distill consistency reflow)
REFLOW_PAIRS="data/reflow_pairs_celeba_latent.pt"
ACCEPTED_CODEC="results/codecs/celeba_vq_f4/accepted_codec.pt"

for algorithm in "${ALGORITHMS[@]}"; do
  if [[ ! -f "${CONFIGS[$algorithm]}" ]]; then
    echo "ERROR: missing latent config: ${CONFIGS[$algorithm]}" >&2
    exit 1
  fi
done

# A parent suite may already own the lock and source manifest.  Direct runs
# become the owner and pass the same token to Reflow generation.
source scripts/linux/workflow_guard.sh
IDENTITY_ARGS=(--launcher scripts/linux/train_celeba_latent.sh)
for algorithm in "${ALGORITHMS[@]}"; do
  IDENTITY_ARGS+=(--config "${CONFIGS[$algorithm]}")
done
workflow_guard_start "train_celeba_latent.sh" "$DRY_RUN" "${IDENTITY_ARGS[@]}"
export DIFFUSION_LIFECYCLE_MODE="$MODE"

PLANNED_FM_CHECKPOINT=""
if [[ "$ONLY" == "all" ]]; then
  PLANNED_FM_CHECKPOINT="$("$PYTHON" scripts/checkpoint_path.py \
    --run-dir results/fm_celeba_latent \
    --class-name FlowMatchingAlgorithm \
    --epoch 100 \
    --planned-mode "$MODE")"
fi

RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST_TOKEN="$(hostname | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-')"
HOST_TOKEN="${HOST_TOKEN%-}"
GPU_NAME=""
GPU_MEMORY_MB=""
GPU_MEMORY_GB=""
GPU_TOKEN=""

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
    GPU_MEMORY_GB="$(((GPU_MEMORY_MB + 1023) / 1024))"
    GPU_TOKEN="${GPU_TOKEN}-${GPU_MEMORY_GB}gb"
  fi
fi

if [[ -z "$LOG_DIR" ]]; then
  if [[ -n "$GPU_TOKEN" ]]; then
    LOG_DIR="training_logs/$GPU_TOKEN/latent"
  else
    LOG_DIR="training_logs/cpu-or-unknown/latent"
  fi
fi

if [[ -z "$MACHINE_LABEL" ]]; then
  OS_TOKEN="$(uname -s | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-')"
  OS_TOKEN="${OS_TOKEN%-}"
  if [[ -n "$GPU_TOKEN" ]]; then
    MACHINE_LABEL="${OS_TOKEN}-${HOST_TOKEN}-${GPU_TOKEN}"
  else
    MACHINE_LABEL="${OS_TOKEN}-${HOST_TOKEN}-gpu-unavailable"
  fi
fi
export DIFFUSION_MACHINE_LABEL="$MACHINE_LABEL"
echo "[PLAN] machine_label=$MACHINE_LABEL"
echo "[PLAN] gpu_name=${GPU_NAME:-unavailable}"
echo "[PLAN] gpu_memory_mb=${GPU_MEMORY_MB:-unavailable}"
echo "[PLAN] track_logs=$TRACK_LOGS"

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_logged() {
  local job_name=$1
  shift
  local -a command=("$@")
  local job_log_dir="$LOG_DIR/$job_name"
  local log_path="$job_log_dir/celeba_latent_${job_name}_${HOST_TOKEN}_${RUN_TIMESTAMP}.log"
  local sidecar_path="${log_path}.meta.json"

  echo "[PLAN] job=$job_name"
  echo "[PLAN] log=$log_path"
  printf '[PLAN] command='
  print_command "${command[@]}"
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi

  mkdir -p "$job_log_dir"
  if [[ -e "$log_path" ]]; then
    echo "ERROR: refusing to overwrite existing training log: $log_path" >&2
    return 1
  fi
  {
    echo "[run] training_type=latent_diffusion"
    echo "[run] representation_space=latent"
    echo "[run] dataset=celeba_latent"
    echo "[run] algorithm=$job_name"
    echo "[run] job=$job_name"
    echo "[run] source_identity_sha256=${DIFFUSION_SOURCE_IDENTITY:-standalone}"
    echo "[run] lifecycle_mode=$MODE"
    echo "[run] machine_label=$MACHINE_LABEL"
    echo "[run] gpu_name=${GPU_NAME:-unavailable}"
    echo "[run] gpu_memory_gb=${GPU_MEMORY_GB:-unavailable}"
    echo "[run] parent_suite_timestamp=${DIFFUSION_PARENT_SUITE_TIMESTAMP:-none}"
    echo "[run] selected_config_sha256=${DIFFUSION_SELECTED_CONFIG_SHA256:-not-applicable}"
    echo "[run] checkpoint_series=${DIFFUSION_CHECKPOINT_SERIES:-resolved-by-train.py}"
    echo "[run] started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '[run] command='
    print_command "${command[@]}"
  } | tee "$log_path"

  set +e
  "${command[@]}" 2>&1 | tee -a "$log_path"
  local status=${PIPESTATUS[0]}
  set -e
  {
    echo "[run] finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[run] exit_status=$status"
  } | tee -a "$log_path"
  "$PYTHON" scripts/write_training_log_metadata.py --log "$log_path" >/dev/null
  if [[ "$TRACK_LOGS" == true ]]; then
    git add -- "$log_path" "$sidecar_path"
    echo "[TRACKED] $log_path"
    echo "[TRACKED] $sidecar_path"
  else
    echo "[UNSTAGED] $log_path"
    echo "[UNSTAGED] $sidecar_path"
  fi
  return "$status"
}

run_training() {
  local algorithm=$1
  local -a command=(
    "$PYTHON" train.py
    --algorithm "$algorithm"
    --config "${EFFECTIVE_CONFIGS[$algorithm]}"
    --mode "$MODE"
  )
  workflow_guard_verify_source "$DRY_RUN"
  if [[ "$DRY_RUN" == false ]]; then
    export DIFFUSION_SELECTED_CONFIG_SHA256
    DIFFUSION_SELECTED_CONFIG_SHA256="$(sha256sum "${EFFECTIVE_CONFIGS[$algorithm]}" | awk '{print $1}')"
  fi
  local resolved_checkpoint
  resolved_checkpoint="$(
    "$PYTHON" scripts/checkpoint_path.py \
      --run-dir "results/${algorithm}_celeba_latent" \
      --class-name "${ALGORITHM_CLASSES[$algorithm]}" \
      --epoch 100 --planned-mode "$MODE"
  )"
  export DIFFUSION_CHECKPOINT_SERIES
  DIFFUSION_CHECKPOINT_SERIES="$(basename "$(dirname "$resolved_checkpoint")")"
  [[ -z "$MACHINE_LABEL" ]] || command+=(--machine-label "$MACHINE_LABEL")
  [[ -z "$CHECKPOINT_EVERY" ]] || command+=(--checkpoint-every "$CHECKPOINT_EVERY")
  [[ "$TRAIN_ONLY" == false ]] || command+=(--train-only)
  run_logged "$algorithm" "${command[@]}"
}

resolve_corrected_fm_checkpoint() {
  if [[ -n "$PLANNED_FM_CHECKPOINT" ]]; then
    printf '%s\n' "$PLANNED_FM_CHECKPOINT"
    return 0
  fi
  "$PYTHON" scripts/checkpoint_path.py \
    --run-dir results/fm_celeba_latent \
    --class-name FlowMatchingAlgorithm \
    --epoch 100
}

prepare_teacher_config() {
  local algorithm=$1
  local teacher_checkpoint
  if ! teacher_checkpoint="$(resolve_corrected_fm_checkpoint)"; then
    echo "ERROR: corrected latent FM epoch-100 checkpoint is not available." >&2
    echo "Resume it first: ./scripts/linux/train_celeba_latent.sh --only fm --mode continue" >&2
    exit 1
  fi
  require_file "$teacher_checkpoint" "corrected latent FM teacher checkpoint"
  if [[ "$DRY_RUN" == true ]]; then
    EFFECTIVE_CONFIGS[$algorithm]="results/launcher_configs/${RUN_TIMESTAMP}/${algorithm}_celeba_latent.json"
    echo "[PLAN] bind ${CONFIGS[$algorithm]} to teacher $teacher_checkpoint"
    return 0
  fi
  local output="results/launcher_configs/${RUN_TIMESTAMP}/${algorithm}_celeba_latent.json"
  "$PYTHON" scripts/prepare_latent_dependency_config.py \
    --source "${CONFIGS[$algorithm]}" \
    --output "$output" \
    --teacher-checkpoint "$teacher_checkpoint" >/dev/null
  EFFECTIVE_CONFIGS[$algorithm]="$output"
}

require_file() {
  local path=$1
  local purpose=$2
  if [[ -f "$path" ]]; then
    echo "[OK] $purpose: $path"
    return 0
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "[PLANNED] $purpose must exist before execution: $path"
    return 0
  fi
  echo "ERROR: missing $purpose: $path" >&2
  exit 1
}

if [[ "$DRY_RUN" == false ]]; then
  require_file "$ACCEPTED_CODEC" "accepted VQ-f4 codec"
else
  echo "[PLANNED] accepted VQ-f4 codec: $ACCEPTED_CODEC"
fi

if [[ "$ONLY" == "all" || "$ONLY" == "fm" ]]; then
  run_training fm
fi
if [[ "$ONLY" == "all" || "$ONLY" == "fm_lognorm" ]]; then
  run_training fm_lognorm
fi
if [[ "$ONLY" == "all" || "$ONLY" == "mf" ]]; then
  run_training mf
fi

if [[ "$ONLY" == "all" || "$ONLY" == "mf_distill" ]]; then
  prepare_teacher_config mf_distill
  run_training mf_distill
fi
if [[ "$ONLY" == "all" || "$ONLY" == "consistency" ]]; then
  prepare_teacher_config consistency
  run_training consistency
fi
if [[ "$ONLY" == "all" || "$ONLY" == "reflow" ]]; then
  if ! FM_CHECKPOINT="$(resolve_corrected_fm_checkpoint)"; then
    echo "ERROR: corrected latent FM epoch-100 checkpoint is not available." >&2
    echo "Resume it first: ./scripts/linux/train_celeba_latent.sh --only fm --mode continue" >&2
    exit 1
  fi
  require_file "$FM_CHECKPOINT" "corrected latent FM teacher checkpoint"
  if [[ ! -f "$REFLOW_PAIRS" ]]; then
    workflow_guard_verify_source "$DRY_RUN"
    export DIFFUSION_SELECTED_CONFIG_SHA256="$(sha256sum "${CONFIGS[fm]}" | awk '{print $1}')"
    export DIFFUSION_CHECKPOINT_SERIES="$(basename "$(dirname "$FM_CHECKPOINT")")"
    run_logged reflow_pairs \
      "$PYTHON" scripts/generate_reflow_pairs_latent.py \
      --checkpoint "$FM_CHECKPOINT" \
      --config "${CONFIGS[fm]}" \
      --output "$REFLOW_PAIRS" \
      --n-pairs 50000 \
      --nfe 50
  else
    echo "[OK] latent Reflow pairs: $REFLOW_PAIRS"
  fi
  run_training reflow
fi

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry-run complete; no training, pair generation, or log writes occurred."
else
  echo "CelebA latent workflow complete. Partitioned logs: $LOG_DIR/<algorithm>/"
fi
