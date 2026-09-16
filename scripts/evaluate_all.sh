#!/usr/bin/env bash
# evaluate_all.sh — evaluate final checkpoint for each algorithm, make plots.
# Skips any algorithm whose checkpoint doesn't exist yet (warns but continues).

set -euo pipefail
EPOCH=100

eval_one() {
  local algo=$1; local ckpt=$2; local cfg=$3
  if [ ! -f "$ckpt" ]; then
    echo "⚠  Skipping $algo — checkpoint not found: $ckpt"
    return
  fi
  echo ""; echo "─── Evaluating: $algo ───"
  python evaluate.py --algorithm "$algo" --checkpoint "$ckpt" --config "$cfg" --make-plots
}

eval_one fm \
  "results/fm_cifar10/checkpoints/FlowMatchingAlgorithm_epoch${EPOCH}.pt" \
  "config/fm_full.json"

eval_one fm_lognorm \
  "results/fm_lognorm_cifar10/checkpoints/FlowMatchingLognormAlgorithm_epoch${EPOCH}.pt" \
  "config/fm_lognorm_full.json"

eval_one mf \
  "results/mf_cifar10/checkpoints/MeanFlowAlgorithm_epoch${EPOCH}.pt" \
  "config/mf_full.json"

eval_one mf_distill \
  "results/mf_distill_cifar10/checkpoints/MeanFlowDistillAlgorithm_epoch${EPOCH}.pt" \
  "config/mf_distill_full.json"

eval_one consistency \
  "results/consistency_cifar10/checkpoints/ConsistencyAlgorithm_epoch${EPOCH}.pt" \
  "config/consistency_full.json"

eval_one reflow \
  "results/reflow_cifar10/checkpoints/ReflowAlgorithm_epoch${EPOCH}.pt" \
  "config/reflow_full.json"

echo ""; echo "✅  Evaluation complete."
echo "    Plots:   results/<experiment>/metrics/plots/"
echo "    Metrics: results/<experiment>/metrics/<experiment>.jsonl"
