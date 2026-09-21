#!/usr/bin/env bash
# Train the self-trained factor-4 CelebA KL-VAE fallback on one GPU.
#
# Recommended start (RTX 3090 24 GB):
#   ./scripts/linux/train_scratch_codec.sh
#
# Preview without training:
#   ./scripts/linux/train_scratch_codec.sh --dry-run
#
# Start over only in a new work directory (existing artifacts are preserved):
#   ./scripts/linux/train_scratch_codec.sh --mode fresh \
#     --work-dir results/scratch_vae_experiment_2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/venv/bin/python"
else
  echo "ERROR: project environment not found. Run ./scripts/linux/init.sh first." >&2
  exit 1
fi

MODE="continue"
DRY_RUN=false
DATA_ROOT="data/raw"
WORK_DIR="results/scratch_vae"
OUTPUT="results/codecs/accepted_scratch_kl_vae.pt"
BATCH_SIZE=128
EPOCHS=60
LEARNING_RATE="1e-4"
WEIGHT_DECAY="1e-4"
NUM_WORKERS=4
SEED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:?--mode requires continue or fresh}"; shift 2 ;;
    --data-root) DATA_ROOT="${2:?--data-root requires a path}"; shift 2 ;;
    --work-dir) WORK_DIR="${2:?--work-dir requires a path}"; shift 2 ;;
    --output) OUTPUT="${2:?--output requires a path}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?--batch-size requires a value}"; shift 2 ;;
    --epochs) EPOCHS="${2:?--epochs requires a value}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?--num-workers requires a value}"; shift 2 ;;
    --seed) SEED="${2:?--seed requires a value}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$MODE" in continue|fresh) ;; *)
  echo "ERROR: --mode must be continue or fresh" >&2; exit 2 ;;
esac
for value in "$BATCH_SIZE" "$EPOCHS" "$NUM_WORKERS" "$SEED"; do
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "ERROR: numeric arguments must be non-negative integers" >&2; exit 2;
  }
done
[[ "$BATCH_SIZE" -gt 0 && "$EPOCHS" -gt 0 ]] || {
  echo "ERROR: batch size and epochs must be positive" >&2; exit 2;
}

CHECKPOINT_DIR="$WORK_DIR/checkpoints"
shopt -s nullglob
CHECKPOINTS=("$CHECKPOINT_DIR"/scratch_vae_epoch*.pt)
shopt -u nullglob

RESUME_ARGS=()
if [[ "$MODE" == "continue" && ${#CHECKPOINTS[@]} -gt 0 ]]; then
  RESUME_ARGS=(--resume auto)
elif [[ "$MODE" == "fresh" && ${#CHECKPOINTS[@]} -gt 0 ]]; then
  echo "ERROR: fresh mode refuses to overwrite checkpoints in $CHECKPOINT_DIR" >&2
  echo "Choose a new --work-dir, or use --mode continue." >&2
  exit 1
fi

COMMAND=(
  "$PYTHON" -m codec.train_scratch_vae
  --data-root "$DATA_ROOT"
  --work-dir "$WORK_DIR"
  --output "$OUTPUT"
  --device cuda
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --kl-start 1e-5
  --kl-end 1e-4
  --kl-warmup-epochs 20
  --gradient-clip-norm 1.0
  --validate-every 5
  --validation-samples 5000
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --amp
  "${RESUME_ARGS[@]}"
)

printf 'Selected scratch-codec command:\n  '
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" == true ]]; then
  exit 0
fi

LOCK_FILE="results/.lock"
mkdir -p results "$WORK_DIR/logs"
LOCK_TOKEN="pid=$$;command=train_scratch_codec.sh;work_dir=$WORK_DIR"
if ! (set -o noclobber; printf '%s\n' "$LOCK_TOKEN" > "$LOCK_FILE") 2>/dev/null; then
  echo "ERROR: GPU lock already exists: $LOCK_FILE" >&2
  cat "$LOCK_FILE" >&2
  exit 1
fi
release_lock() {
  if [[ -f "$LOCK_FILE" && "$(cat "$LOCK_FILE")" == "$LOCK_TOKEN" ]]; then
    rm -f -- "$LOCK_FILE"
  fi
}
trap release_lock EXIT INT TERM

LOG_FILE="$WORK_DIR/logs/train_$(date +%Y%m%d_%H%M%S)_pid$$.log"
echo "Writing complete output to $LOG_FILE"
"${COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE"
