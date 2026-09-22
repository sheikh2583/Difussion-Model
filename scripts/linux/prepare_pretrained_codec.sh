#!/usr/bin/env bash
# Download and validate the frozen face-specific CelebA-HQ VQ-f4 codec.
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
SOURCE_ROOT="data/pretrained/ldm-celebahq-256"
OUTPUT_DIR="results/codecs/celeba_vq_f4"
REVISION="main"
BATCH_SIZE=32
NUM_WORKERS=4
ACCEPT_QUALITY_FAILURE=false
ACCEPTANCE_REASON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="${2:?--data-root requires a path}"; shift 2 ;;
    --source-dir) SOURCE_ROOT="${2:?--source-dir requires a path}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a path}"; shift 2 ;;
    --revision) REVISION="${2:?--revision requires a revision}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?--batch-size requires a value}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?--num-workers requires a value}"; shift 2 ;;
    --accept-quality-failure) ACCEPT_QUALITY_FAILURE=true; shift ;;
    --acceptance-reason) ACCEPTANCE_REASON="${2:?--acceptance-reason requires text}"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

DOWNLOAD=(
  "$PYTHON" -m codec.download_pretrained_vq
  --repo-id CompVis/ldm-celebahq-256
  --revision "$REVISION"
  --output-dir "$SOURCE_ROOT"
)
VALIDATE=(
  "$PYTHON" codec/validate_codec.py
  --codec-source CompVis/ldm-celebahq-256
  --codec-source-path "$SOURCE_ROOT/vqvae"
  --codec-source-revision auto
  --celeba-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --device cuda
)
if [[ "$ACCEPT_QUALITY_FAILURE" == true ]]; then
  [[ -n "$ACCEPTANCE_REASON" ]] || {
    echo "ERROR: --accept-quality-failure requires --acceptance-reason." >&2
    exit 2
  }
  VALIDATE+=(--accept-quality-failure --acceptance-reason "$ACCEPTANCE_REASON")
elif [[ -n "$ACCEPTANCE_REASON" ]]; then
  echo "ERROR: --acceptance-reason requires --accept-quality-failure." >&2
  exit 2
fi

source scripts/linux/workflow_guard.sh
workflow_guard_start "prepare_pretrained_codec.sh" "$DRY_RUN" \
  --launcher scripts/linux/prepare_pretrained_codec.sh

if [[ "$SKIP_DOWNLOAD" == false ]]; then
  printf 'Download command:\n  '; printf '%q ' "${DOWNLOAD[@]}"; printf '\n'
fi
printf 'Validation command:\n  '; printf '%q ' "${VALIDATE[@]}"; printf '\n'
[[ "$DRY_RUN" == true ]] && exit 0

CODEC_LOG_DIR="$(workflow_device_log_dir codec)"
mkdir -p "$CODEC_LOG_DIR"
LOG_FILE="$CODEC_LOG_DIR/celeba_codec_pretrained_vq_f4_$(date +%Y%m%d_%H%M%S)_pid$$.log"
if [[ "$SKIP_DOWNLOAD" == false ]]; then
  "${DOWNLOAD[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

{
  echo "[run] training_type=codec_validation"
  echo "[run] representation_space=codec"
  echo "[run] dataset=celeba"
  echo "[run] algorithm=pretrained_vq_f4"
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
