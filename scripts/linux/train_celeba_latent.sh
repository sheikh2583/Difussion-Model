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
ALGORITHMS=(fm fm_lognorm mf mf_distill consistency reflow)
FM_CHECKPOINT="results/fm_celeba_latent/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt"
REFLOW_PAIRS="data/reflow_pairs_celeba_latent.pt"
ACCEPTED_CODEC="results/codecs/celeba_vq_f4/accepted_codec.pt"

for algorithm in "${ALGORITHMS[@]}"; do
  if [[ ! -f "${CONFIGS[$algorithm]}" ]]; then
    echo "ERROR: missing latent config: ${CONFIGS[$algorithm]}" >&2
    exit 1
  fi
done

if [[ "$ONLY" == "all" ]]; then
  PLANNED_FM_CHECKPOINT="$("$PYTHON" scripts/checkpoint_path.py \
    --run-dir results/fm_celeba_latent \
    --class-name FlowMatchingAlgorithm \
    --epoch 100 \
    --planned-mode "$MODE")"
  if [[ "$PLANNED_FM_CHECKPOINT" != "$FM_CHECKPOINT" ]]; then
    echo "ERROR: the latent teacher configs are pinned to $FM_CHECKPOINT," >&2
    echo "but --mode $MODE would train or resume FM at $PLANNED_FM_CHECKPOINT." >&2
    echo "Run algorithms individually with --only, or align the configs explicitly." >&2
    exit 1
  fi
fi

RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST_TOKEN="$(hostname | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-')"
HOST_TOKEN="${HOST_TOKEN%-}"

if [[ -z "$LOG_DIR" ]]; then
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
    LOG_DIR="training_logs/$GPU_TOKEN/latent"
  else
    LOG_DIR="training_logs/cpu-or-unknown/latent"
  fi
fi

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_logged() {
  local job_name=$1
  shift
  local -a command=("$@")
  local log_path="$LOG_DIR/celeba_latent_${job_name}_${HOST_TOKEN}_${RUN_TIMESTAMP}.log"

  echo "[PLAN] job=$job_name"
  echo "[PLAN] log=$log_path"
  printf '[PLAN] command='
  print_command "${command[@]}"
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi

  mkdir -p "$LOG_DIR"
  {
    echo "[run] training_type=latent_diffusion"
    echo "[run] representation_space=latent"
    echo "[run] dataset=celeba_latent"
    echo "[run] algorithm=$job_name"
    echo "[run] job=$job_name"
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
  return "$status"
}

run_training() {
  local algorithm=$1
  local -a command=(
    "$PYTHON" train.py
    --algorithm "$algorithm"
    --config "${CONFIGS[$algorithm]}"
    --mode "$MODE"
  )
  [[ -z "$MACHINE_LABEL" ]] || command+=(--machine-label "$MACHINE_LABEL")
  [[ -z "$CHECKPOINT_EVERY" ]] || command+=(--checkpoint-every "$CHECKPOINT_EVERY")
  [[ "$TRAIN_ONLY" == false ]] || command+=(--train-only)
  run_logged "$algorithm" "${command[@]}"
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
  require_file "$FM_CHECKPOINT" "latent FM teacher checkpoint"
  run_training mf_distill
fi
if [[ "$ONLY" == "all" || "$ONLY" == "consistency" ]]; then
  require_file "$FM_CHECKPOINT" "latent FM teacher checkpoint"
  run_training consistency
fi
if [[ "$ONLY" == "all" || "$ONLY" == "reflow" ]]; then
  require_file "$FM_CHECKPOINT" "latent FM teacher checkpoint"
  if [[ ! -f "$REFLOW_PAIRS" ]]; then
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
  echo "CelebA latent workflow complete. Logs: $LOG_DIR"
fi
