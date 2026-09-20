#!/usr/bin/env bash
# Unattended, dependency-aware CIFAR-10 + CelebA training with per-model logs.
set -euo pipefail

PLATFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$PLATFORM_DIR")")"
cd "$PROJECT_ROOT"

DATASET="all"
MODE="continue"
CHECKPOINT_EVERY=10
TRAIN_ONLY=false
DRY_RUN=false

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

RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST_TOKEN="$(hostname | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-')"
HOST_TOKEN="${HOST_TOKEN%-}"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  GPU_MEMORY_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
else
  GPU_NAME=""
  GPU_MEMORY_MB=""
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
if [[ "$DRY_RUN" == false ]]; then
  mkdir -p "$DEVICE_LOG_DIR"
fi
LOG_FILES=()
FAILURES=()

for dataset in "${DATASETS[@]}"; do
  for algorithm in "${ALGORITHMS[@]}"; do
    command=(
      "$PROJECT_ROOT/scripts/linux/train_all.sh"
      --dataset "$dataset"
      --only "$algorithm"
      --mode "$MODE"
      --checkpoint-every "$CHECKPOINT_EVERY"
    )
    [[ "$TRAIN_ONLY" == false ]] || command+=(--train-only)
    [[ "$DRY_RUN" == false ]] || command+=(--dry-run)
    if [[ "$dataset" == "cifar10" && "$algorithm" == "mf" ]]; then
      command+=(--mf-config config/mf_v3_exact_jvp_b128.json)
    fi

    if [[ "$DRY_RUN" == true ]]; then
      echo "[DRY-RUN] ${command[*]}"
      "${command[@]}"
      continue
    fi

    log_relative="$DEVICE_LOG_DIR/${dataset}_${algorithm}_${HOST_TOKEN}_${RUN_TIMESTAMP}.log"
    LOG_FILES+=("$log_relative")
    {
      echo "[run] dataset=$dataset algorithm=$algorithm"
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
    if [[ "$status" -ne 0 ]]; then
      FAILURES+=("${dataset}:${algorithm}:${status}")
      echo "WARNING: $dataset/$algorithm failed; continuing to the next model." >&2
    fi
  done
done

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry-run complete; no logs, commits, or pushes were created."
  exit 0
fi

# Commit all per-model logs together only after every requested job has been
# attempted. This keeps the source code identity identical across model runs.
git add -- "${LOG_FILES[@]}"
if git commit -m "logs: record unattended dataset training $RUN_TIMESTAMP" -- "${LOG_FILES[@]}"; then
  CURRENT_BRANCH="$(git branch --show-current)"
  if [[ -n "$CURRENT_BRANCH" ]]; then
    GIT_TERMINAL_PROMPT=0 git push origin "$CURRENT_BRANCH" || {
      echo "WARNING: logs are committed locally, but Git push failed." >&2
      echo "Run later: git push origin $CURRENT_BRANCH" >&2
    }
  fi
else
  echo "WARNING: logs remain on disk, but their Git commit failed." >&2
fi

if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo "Completed with failed jobs: ${FAILURES[*]}" >&2
  exit 1
fi
echo "All requested CIFAR-10 and CelebA jobs completed successfully."
