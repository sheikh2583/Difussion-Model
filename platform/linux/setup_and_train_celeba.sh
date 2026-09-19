#!/usr/bin/env bash
# setup_and_train_celeba.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-shot CelebA setup + unattended training with auto-logging and git push.
#
# Usage:
#   bash platform/linux/setup_and_train_celeba.sh [--mode continue|fresh]
#                                                  [--checkpoint-every N]
#                                                  [--dry-run]
#
# What this does:
#   1. Extracts img_align_celeba.zip  (if not already extracted)
#   2. Downloads the 5 small annotation files via gdown  (if missing)
#   3. Verifies torchvision can open the dataset
#   4. Runs train_all_datasets.sh --dataset celeba
#      → trains fm / fm_lognorm / mf / mf_distill / consistency / reflow
#      → writes per-model logs to training_logs/
#      → auto-commits and git-pushes the logs when done
#
# Prerequisites:
#   - venv must exist  (run ./init_all.sh first for Python deps)
#   - gdown must be installed  (script installs it if missing)
#   - img_align_celeba.zip must be in data/raw/celeba/  (download separately
#     if Google Drive quota blocks the automated path)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PLATFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$PLATFORM_DIR")")"
cd "$PROJECT_ROOT"

CELEBA_DIR="data/raw/celeba"
ZIP_PATH="$CELEBA_DIR/img_align_celeba.zip"
IMAGES_DIR="$CELEBA_DIR/img_align_celeba"
MODE="continue"
CHECKPOINT_EVERY=10
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || { echo "ERROR: --mode requires a value" >&2; exit 2; }
      MODE="$2"; shift 2 ;;
    --checkpoint-every)
      [[ $# -ge 2 ]] || { echo "ERROR: --checkpoint-every requires a value" >&2; exit 2; }
      CHECKPOINT_EVERY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help)
      sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'
      exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$MODE" in
  continue|fresh) ;;
  *) echo "ERROR: --mode must be continue or fresh" >&2; exit 2 ;;
esac

PYTHON="$PROJECT_ROOT/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: venv not found. Run ./init_all.sh first." >&2
  exit 1
fi

echo "═══════════════════════════════════════════════════"
echo " CelebA setup + training  (mode=$MODE)"
echo "═══════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────
# Step 1 — ensure gdown is available
# ─────────────────────────────────────────────────────
echo ""
echo "[step 1/4] Checking gdown..."
if ! "$PYTHON" -m gdown --version >/dev/null 2>&1; then
  echo "  gdown not found — installing..."
  "$PYTHON" -m pip install -q gdown
  echo "  [OK] gdown installed"
else
  echo "  [OK] gdown already available"
fi

# ─────────────────────────────────────────────────────
# Step 2 — extract img_align_celeba.zip
# ─────────────────────────────────────────────────────
echo ""
echo "[step 2/4] Checking image archive..."
mkdir -p "$CELEBA_DIR"

if [[ -d "$IMAGES_DIR" ]]; then
  COUNT=$(find "$IMAGES_DIR" -name "*.jpg" | wc -l)
  echo "  [OK] img_align_celeba/ already extracted ($COUNT images)"
else
  if [[ ! -f "$ZIP_PATH" ]]; then
    echo "  img_align_celeba.zip not found — attempting download (~1.4 GB)..."
    if [[ "$DRY_RUN" == true ]]; then
      echo "  [DRY-RUN] would download img_align_celeba.zip"
    else
      "$PYTHON" -m gdown 0B7EVK8r0v71pZjFTYXZWM3FlRnM -O "$ZIP_PATH" || {
        echo ""
        echo "ERROR: Download failed (Google Drive quota likely exceeded)." >&2
        echo "  Manual fix:" >&2
        echo "  1. Download img_align_celeba.zip from:" >&2
        echo "     https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" >&2
        echo "  2. Place it at: $PROJECT_ROOT/$ZIP_PATH" >&2
        echo "  3. Re-run this script." >&2
        exit 1
      }
    fi
  fi

  if [[ "$DRY_RUN" == false ]]; then
    echo "  Extracting $ZIP_PATH  (this takes ~2 min)..."
    unzip -q "$ZIP_PATH" -d "$CELEBA_DIR"
    COUNT=$(find "$IMAGES_DIR" -name "*.jpg" | wc -l)
    echo "  [OK] Extracted $COUNT images to $IMAGES_DIR"
  else
    echo "  [DRY-RUN] would extract $ZIP_PATH"
  fi
fi

# ─────────────────────────────────────────────────────
# Step 3 — download missing annotation files
# ─────────────────────────────────────────────────────
echo ""
echo "[step 3/4] Checking annotation files..."

declare -A ANNOTATIONS=(
  ["list_attr_celeba.txt"]="0B7EVK8r0v71pblRyaVFSWGxPY0U"
  ["list_bbox_celeba.txt"]="0B7EVK8r0v71pd0eFdThQdDB2aEk"
  ["identity_CelebA.txt"]="1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS"
  ["list_landmarks_align_celeba.txt"]="0B7EVK8r0v71pbThiMVRxWXZ4dU0"
  ["list_eval_partition.txt"]="0B7EVK8r0v71pY0NSMzRuSXJEVkk"
)

ALL_PRESENT=true
for filename in "${!ANNOTATIONS[@]}"; do
  filepath="$CELEBA_DIR/$filename"
  if [[ -f "$filepath" ]]; then
    echo "  [OK] $filename"
  else
    ALL_PRESENT=false
    echo "  [MISSING] $filename — downloading..."
    if [[ "$DRY_RUN" == false ]]; then
      "$PYTHON" -m gdown "${ANNOTATIONS[$filename]}" -O "$filepath" || {
        echo "ERROR: Could not download $filename" >&2
        echo "  Download it manually from:" >&2
        echo "  https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" >&2
        exit 1
      }
      echo "  [OK] $filename downloaded"
    else
      echo "  [DRY-RUN] would download $filename"
    fi
  fi
done

if [[ "$ALL_PRESENT" == true ]]; then
  echo "  [OK] All annotation files present"
fi

# ─────────────────────────────────────────────────────
# Step 4 — verify torchvision can open the dataset
# ─────────────────────────────────────────────────────
echo ""
echo "[step 4/4] Verifying dataset with torchvision..."
if [[ "$DRY_RUN" == false ]]; then
  VERIFY_OUT=$("$PYTHON" - <<'PYEOF'
from torchvision import datasets
try:
    ds = datasets.CelebA("data/raw", split="train", download=False)
    print(f"OK:{len(ds)}")
except Exception as e:
    print(f"ERR:{e}")
PYEOF
)
  if [[ "$VERIFY_OUT" == OK:* ]]; then
    N="${VERIFY_OUT#OK:}"
    echo "  [OK] torchvision sees $N training images — dataset is ready"
  else
    echo "ERROR: torchvision verification failed:" >&2
    echo "  ${VERIFY_OUT#ERR:}" >&2
    echo "" >&2
    echo "  Common fix: ensure img_align_celeba/ folder is inside data/raw/celeba/" >&2
    exit 1
  fi
else
  echo "  [DRY-RUN] skipping torchvision verification"
fi

# ─────────────────────────────────────────────────────
# Launch unattended training
# ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo " Launching CelebA training pipeline..."
echo " Logs → training_logs/  |  Auto git-push on finish"
echo "═══════════════════════════════════════════════════"
echo ""

TRAIN_CMD=(
  bash platform/linux/train_all_datasets.sh
  --dataset celeba
  --mode "$MODE"
  --checkpoint-every "$CHECKPOINT_EVERY"
)
[[ "$DRY_RUN" == false ]] || TRAIN_CMD+=(--dry-run)

exec "${TRAIN_CMD[@]}"
