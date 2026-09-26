#!/usr/bin/env bash
# Train the self-trained factor-4 CelebA KL-VAE encoder/decoder on one GPU.
#
# Start a new scratch KL-VAE encoder/decoder run (no checkpoint resume):
#   ./scripts/linux/train_scratch_codec.sh
#
# Run staged hyperparameter sweep:
#   ./scripts/linux/train_scratch_codec.sh --stage A
#   ./scripts/linux/train_scratch_codec.sh --stage B --lambda-kl 1e-6
#   ./scripts/linux/train_scratch_codec.sh --stage C --lambda-kl 1e-6
#   ./scripts/linux/train_scratch_codec.sh --stage D --lambda-kl 1e-6
#
# Preview without training:
#   ./scripts/linux/train_scratch_codec.sh --dry-run
#
# This launcher is fresh-only. If its work directory already contains a
# checkpoint, it exits without modifying that experiment. Choose a new
# --work-dir and matching --output for another fresh run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: project environment not found. Run ./scripts/linux/init.sh first." >&2
  exit 1
fi

STAGE=""
MODE="fresh"
DRY_RUN=false
DATA_ROOT="data/raw"
WORK_DIR="results/codecs/scratch_kl_vae_linux_fresh"
OUTPUT="results/codecs/scratch_kl_vae_linux_fresh/accepted_codec.pt"
BATCH_SIZE=64
EPOCHS=200
LEARNING_RATE="1e-4"
WEIGHT_DECAY="1e-4"
KL_START="1e-5"
KL_END="1e-4"
KL_WARMUP_EPOCHS=20
GRADIENT_CLIP_NORM=1.0
VALIDATE_EVERY=20
VALIDATION_SAMPLES=5000
NUM_WORKERS=3
SEED=0

# New perceptual + adversarial args
LAMBDA_KL="1e-6"
LAMBDA_PERC="0.1"
LAMBDA_ADV="0.1"
ADV_START_EPOCH=5
DISC_LR="4e-4"
SWEEP_MODE=false
SWEEP_EPOCHS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="${2:?--stage requires A|B|C|D}"; shift 2 ;;
    --mode) MODE="${2:?--mode requires continue or fresh}"; shift 2 ;;
    --data-root) DATA_ROOT="${2:?--data-root requires a path}"; shift 2 ;;
    --work-dir) WORK_DIR="${2:?--work-dir requires a path}"; shift 2 ;;
    --output) OUTPUT="${2:?--output requires a path}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?--batch-size requires a value}"; shift 2 ;;
    --epochs) EPOCHS="${2:?--epochs requires a value}"; shift 2 ;;
    --learning-rate) LEARNING_RATE="${2:?--learning-rate requires a value}"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="${2:?--weight-decay requires a value}"; shift 2 ;;
    --kl-start) KL_START="${2:?--kl-start requires a value}"; shift 2 ;;
    --kl-end) KL_END="${2:?--kl-end requires a value}"; shift 2 ;;
    --kl-warmup-epochs) KL_WARMUP_EPOCHS="${2:?--kl-warmup-epochs requires a value}"; shift 2 ;;
    --gradient-clip-norm) GRADIENT_CLIP_NORM="${2:?--gradient-clip-norm requires a value}"; shift 2 ;;
    --validate-every) VALIDATE_EVERY="${2:?--validate-every requires a value}"; shift 2 ;;
    --validation-samples) VALIDATION_SAMPLES="${2:?--validation-samples requires a value}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?--num-workers requires a value}"; shift 2 ;;
    --seed) SEED="${2:?--seed requires a value}"; shift 2 ;;
    --lambda-kl) LAMBDA_KL="${2:?--lambda-kl requires a value}"; shift 2 ;;
    --lambda-perc) LAMBDA_PERC="${2:?--lambda-perc requires a value}"; shift 2 ;;
    --lambda-adv) LAMBDA_ADV="${2:?--lambda-adv requires a value}"; shift 2 ;;
    --adv-start-epoch) ADV_START_EPOCH="${2:?--adv-start-epoch requires a value}"; shift 2 ;;
    --disc-lr) DISC_LR="${2:?--disc-lr requires a value}"; shift 2 ;;
    --sweep-mode) SWEEP_MODE=true; shift ;;
    --sweep-epochs) SWEEP_EPOCHS="${2:?--sweep-epochs requires a value}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Apply stage presets
case "$STAGE" in
  A)
    # Stage A: Find safe lambda_kl (5 epochs, no perceptual/adv)
    LAMBDA_PERC="0.0"
    LAMBDA_ADV="0.0"
    SWEEP_MODE=true
    SWEEP_EPOCHS=5
    EPOCHS=5
    VALIDATE_EVERY=9999
    ;;
  B)
    # Stage B: Add perceptual loss (10 epochs, no adv)
    LAMBDA_ADV="0.0"
    ADV_START_EPOCH=9999
    SWEEP_MODE=true
    SWEEP_EPOCHS=10
    EPOCHS=10
    VALIDATE_EVERY=9999
    ;;
  C)
    # Stage C: Full loss probe (20 epochs)
    ADV_START_EPOCH=5
    SWEEP_MODE=false
    EPOCHS=20
    VALIDATE_EVERY=5
    ;;
  D)
    # Stage D: Full 150-epoch run
    ADV_START_EPOCH=5
    SWEEP_MODE=false
    EPOCHS=150
    VALIDATE_EVERY=20
    ;;
  "")
    # No stage - use explicit args (legacy behavior)
    ;;
  *)
    echo "ERROR: --stage must be A, B, C, or D" >&2; exit 2 ;;
esac

case "$MODE" in fresh) ;; *)
  echo "ERROR: this launcher only supports --mode fresh; use a separate resume command for existing runs" >&2; exit 2 ;;
esac
for value in "$BATCH_SIZE" "$EPOCHS" "$NUM_WORKERS" "$SEED" \
  "$KL_WARMUP_EPOCHS" "$VALIDATE_EVERY" "$VALIDATION_SAMPLES" \
  "$ADV_START_EPOCH" "$SWEEP_EPOCHS"; do
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "ERROR: numeric arguments must be non-negative integers" >&2; exit 2;
  }
done
[[ "$BATCH_SIZE" -gt 0 && "$EPOCHS" -gt 0 && "$KL_WARMUP_EPOCHS" -gt 0 \
  && "$VALIDATE_EVERY" -gt 0 && "$VALIDATION_SAMPLES" -gt 0 \
  && "$ADV_START_EPOCH" -gt 0 && "$SWEEP_EPOCHS" -gt 0 ]] || {
  echo "ERROR: batch size, epochs, warmup, validation interval, sample count, adv_start_epoch, sweep_epochs must be positive" >&2; exit 2;
}

if [[ -d "$WORK_DIR" ]] && [[ -n "$(find "$WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "ERROR: fresh mode requires an empty or nonexistent work directory: $WORK_DIR" >&2
  echo "Choose a new --work-dir and matching --output to start a fresh experiment." >&2
  exit 1
fi
if [[ -e "$OUTPUT" ]]; then
  echo "ERROR: fresh mode refuses to overwrite the existing output: $OUTPUT" >&2
  echo "Choose a new --output path." >&2
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
  --kl-start "$KL_START"
  --kl-end "$KL_END"
  --kl-warmup-epochs "$KL_WARMUP_EPOCHS"
  --gradient-clip-norm "$GRADIENT_CLIP_NORM"
  --validate-every "$VALIDATE_EVERY"
  --validation-samples "$VALIDATION_SAMPLES"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --amp
  --lambda-kl "$LAMBDA_KL"
  --lambda-perc "$LAMBDA_PERC"
  --lambda-adv "$LAMBDA_ADV"
  --adv-start-epoch "$ADV_START_EPOCH"
  --disc-lr "$DISC_LR"
)

if [[ "$SWEEP_MODE" == true ]]; then
  COMMAND+=(--sweep-mode --sweep-epochs "$SWEEP_EPOCHS")
fi

printf 'Selected scratch-codec command:\n  '
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" == true ]]; then
  exit 0
fi

source scripts/linux/workflow_guard.sh
workflow_guard_start "train_scratch_codec.sh" false \
  --launcher scripts/linux/train_scratch_codec.sh
CODEC_LOG_DIR="$(workflow_device_log_dir codec)"
mkdir -p results "$CODEC_LOG_DIR"

LOG_FILE="$CODEC_LOG_DIR/celeba_codec_scratch_kl_vae_$(date +%Y%m%d_%H%M%S)_pid$$.log"
echo "Writing complete output to $LOG_FILE"
{
  echo "[run] training_type=scratch_codec"
  echo "[run] representation_space=codec"
  echo "[run] dataset=celeba"
  echo "[run] algorithm=scratch_kl_vae"
  echo "[run] started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '[run] command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} | tee -a "$LOG_FILE"
set +e
"${COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
{
  echo "[run] finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[run] exit_status=$status"
} | tee -a "$LOG_FILE"
exit "$status"
