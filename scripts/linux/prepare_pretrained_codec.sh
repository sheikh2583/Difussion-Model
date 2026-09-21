#!/usr/bin/env bash
# Download and validate the frozen factor-8 CelebA codec.
#
# Preview only:
#   ./scripts/linux/prepare_pretrained_codec.sh --dry-run
#
# Operator run:
#   ./scripts/linux/prepare_pretrained_codec.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/venv/bin/python"
[[ -x "$PYTHON" ]] || {
  echo "ERROR: project environment not found. Run ./scripts/linux/init.sh first." >&2
  exit 1
}

DRY_RUN=false
SKIP_DOWNLOAD=false
DATA_ROOT="data/raw"
SOURCE_DIR="data/pretrained/sd-vae-ft-mse"
OUTPUT_DIR="results/codec_celeba_pretrained"
REVISION="main"
BATCH_SIZE=32
NUM_WORKERS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="${2:?--data-root requires a path}"; shift 2 ;;
    --source-dir) SOURCE_DIR="${2:?--source-dir requires a path}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a path}"; shift 2 ;;
    --revision) REVISION="${2:?--revision requires a revision}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?--batch-size requires a value}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?--num-workers requires a value}"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

DOWNLOAD=(
  "$PYTHON" -m codec.download_pretrained_vae
  --repo-id stabilityai/sd-vae-ft-mse
  --revision "$REVISION"
  --output-dir "$SOURCE_DIR"
)
VALIDATE=(
  "$PYTHON" codec/validate_codec.py
  --codec-source stabilityai/sd-vae-ft-mse
  --codec-source-path "$SOURCE_DIR"
  --codec-source-revision auto
  --celeba-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --device cuda
)

if [[ "$SKIP_DOWNLOAD" == false ]]; then
  printf 'Download command:\n  '; printf '%q ' "${DOWNLOAD[@]}"; printf '\n'
fi
printf 'Validation command:\n  '; printf '%q ' "${VALIDATE[@]}"; printf '\n'
[[ "$DRY_RUN" == true ]] && exit 0

mkdir -p "$OUTPUT_DIR/logs"
LOG_FILE="$OUTPUT_DIR/logs/validate_$(date +%Y%m%d_%H%M%S)_pid$$.log"
if [[ "$SKIP_DOWNLOAD" == false ]]; then
  "${DOWNLOAD[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

LOCK_FILE="results/.lock"
LOCK_TOKEN="pid=$$;command=prepare_pretrained_codec.sh;output_dir=$OUTPUT_DIR"
if ! (set -o noclobber; printf '%s\n' "$LOCK_TOKEN" > "$LOCK_FILE") 2>/dev/null; then
  echo "ERROR: GPU lock already exists: $LOCK_FILE" | tee -a "$LOG_FILE" >&2
  exit 1
fi
release_lock() {
  if [[ -f "$LOCK_FILE" && "$(cat "$LOCK_FILE")" == "$LOCK_TOKEN" ]]; then
    rm -f -- "$LOCK_FILE"
  fi
}
trap release_lock EXIT INT TERM

{
  echo "[run] training_type=codec_validation"
  echo "[run] dataset=celeba"
  echo "[run] algorithm=pretrained_autoencoderkl"
  echo "[run] started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "$LOG_FILE"
set +e
"${VALIDATE[@]}" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
{
  echo "[run] finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[run] exit_status=$status"
} | tee -a "$LOG_FILE"
exit "$status"
