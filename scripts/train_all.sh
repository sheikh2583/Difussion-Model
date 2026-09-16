#!/usr/bin/env bash
# train_all.sh — train all 6 algorithms in correct dependency order.
#
# Dependency order (must not change):
#   1. fm           — baseline; teacher for mf_distill, consistency, reflow
#   2. fm_lognorm   — independent
#   3. mf           — independent
#   4. mf_distill   — needs fm checkpoint
#   5. consistency  — needs fm checkpoint
#   6. reflow       — needs fm checkpoint + generate_reflow_pairs.py run first
#
# Usage:
#   ./train_all.sh                          # train all
#   ./train_all.sh --only mf_distill        # train one
#   ./train_all.sh --skip-reflow            # skip one
#   ./train_all.sh --dataset celeba         # use CelebA configs instead
#
# See IMPLEMENTATION_LOG.md for phase details and troubleshooting.

set -euo pipefail

SKIP_FM=false; SKIP_FM_LN=false; SKIP_MF=false
SKIP_MF_DISTILL=false; SKIP_CONSISTENCY=false; SKIP_REFLOW=false
ONLY=""
DATASET="cifar10"   # or "celeba"

for arg in "$@"; do
  case $arg in
    --skip-fm)          SKIP_FM=true ;;
    --skip-fm-lognorm)  SKIP_FM_LN=true ;;
    --skip-mf)          SKIP_MF=true ;;
    --skip-mf-distill)  SKIP_MF_DISTILL=true ;;
    --skip-consistency) SKIP_CONSISTENCY=true ;;
    --skip-reflow)      SKIP_REFLOW=true ;;
    --only)             shift; ONLY="$1" ;;
    --dataset)          shift; DATASET="$1" ;;
  esac
done

# Config suffix based on dataset
if [ "$DATASET" = "celeba" ]; then
  SUFFIX="celeba64"
else
  SUFFIX="full"
fi

run() {
  local algo=$1; local cfg=$2
  if [ -n "$ONLY" ] && [ "$ONLY" != "$algo" ]; then return; fi
  echo ""; echo "═══ Training: $algo  (config: $cfg) ═══"
  python train.py --algorithm "$algo" --config "$cfg"
}

check_ckpt() {
  local path=$1; local algo=$2
  if [ ! -f "$path" ] && [ -z "$ONLY" ]; then
    echo "⚠  Checkpoint not found for $algo: $path"
    echo "   Train $algo first or update the teacher_checkpoint in the config."
    exit 1
  fi
}

FM_CKPT="results/fm_${SUFFIX}/checkpoints/FlowMatchingAlgorithm_epoch100.pt"
# For CIFAR-10 use existing checkpoint naming
if [ "$DATASET" = "cifar10" ]; then
  FM_CKPT="results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch100.pt"
fi

# 1–3: independent runs
[ "$SKIP_FM" = false ]    && run fm         "config/fm_${SUFFIX}.json"
[ "$SKIP_FM_LN" = false ] && run fm_lognorm "config/fm_lognorm_${SUFFIX}.json"
[ "$SKIP_MF" = false ]    && run mf         "config/mf_${SUFFIX}.json"

# 4: needs FM checkpoint
if [ "$SKIP_MF_DISTILL" = false ]; then
  check_ckpt "$FM_CKPT" fm
  run mf_distill "config/mf_distill_full.json"
fi

# 5: needs FM checkpoint
if [ "$SKIP_CONSISTENCY" = false ]; then
  check_ckpt "$FM_CKPT" fm
  run consistency "config/consistency_full.json"
fi

# 6: needs reflow pairs generated first
if [ "$SKIP_REFLOW" = false ]; then
  PAIRS="data/reflow_pairs_${DATASET}.pt"
  if [ ! -f "$PAIRS" ]; then
    echo ""; echo "⚠  Reflow pairs not found at $PAIRS"
    echo "   Run first:"
    echo "   python scripts/generate_reflow_pairs.py \\"
    echo "     --checkpoint $FM_CKPT \\"
    echo "     --config config/fm_${SUFFIX}.json \\"
    echo "     --n-pairs 50000 --output $PAIRS"
    if [ -z "$ONLY" ]; then exit 1; fi
  fi
  run reflow "config/reflow_full.json"
fi

echo ""; echo "✅  All training complete. Run ./evaluate_all.sh to score all methods."
