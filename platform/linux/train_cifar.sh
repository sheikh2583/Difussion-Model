#!/usr/bin/env bash
set -euo pipefail
PLATFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$PLATFORM_DIR")")"

# Help and dry-runs must stay side-effect free: do not create or commit logs.
for argument in "$@"; do
  if [[ "$argument" == "--dry-run" || "$argument" == "--help" || "$argument" == "-h" ]]; then
    exec "$PROJECT_ROOT/scripts/train_all.sh" \
      --dataset cifar10 \
      --mode continue \
      --checkpoint-every 10 \
      --mf-config config/mf_v3_exact_jvp_b128.json \
      "$@"
  fi
done

# The outer invocation captures the complete terminal stream. The inner
# invocation performs the actual dependency-aware training without recursion.
if [[ "${DIFFUSION_CIFAR_LOGGING_ACTIVE:-0}" != "1" ]]; then
  cd "$PROJECT_ROOT"
  RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  HOST_TOKEN="$(hostname | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-')"
  HOST_TOKEN="${HOST_TOKEN%-}"
  LOG_RELATIVE="training_logs/cifar10_${HOST_TOKEN}_${RUN_TIMESTAMP}.log"
  LOG_PATH="$PROJECT_ROOT/$LOG_RELATIVE"
  mkdir -p "$PROJECT_ROOT/training_logs"

  printf '[log] Capturing complete CIFAR-10 terminal output: %s\n' "$LOG_RELATIVE" \
    | tee "$LOG_PATH"
  set +e
  DIFFUSION_CIFAR_LOGGING_ACTIVE=1 "$0" "$@" 2>&1 | tee -a "$LOG_PATH"
  TRAINING_STATUS=${PIPESTATUS[0]}
  set -e
  printf '[log] Workflow exit status: %s\n' "$TRAINING_STATUS" | tee -a "$LOG_PATH"

  # Track only this run's text log. Checkpoints, datasets, generated images,
  # and all other results remain ignored, and unrelated user changes are not
  # staged. A failed push leaves the log safely committed in the local clone.
  git add -- "$LOG_RELATIVE"
  if git commit -m "logs: record CIFAR-10 training $RUN_TIMESTAMP" -- "$LOG_RELATIVE"; then
    CURRENT_BRANCH="$(git branch --show-current)"
    if [[ -n "$CURRENT_BRANCH" ]]; then
      git push origin "$CURRENT_BRANCH" || {
        echo "WARNING: log committed locally, but Git push failed." >&2
        echo "Run later: git push origin $CURRENT_BRANCH" >&2
      }
    fi
  else
    echo "WARNING: could not commit $LOG_RELATIVE; the log remains on disk." >&2
  fi
  exit "$TRAINING_STATUS"
fi

exec "$PROJECT_ROOT/scripts/train_all.sh" \
  --dataset cifar10 \
  --mode continue \
  --checkpoint-every 10 \
  --mf-config config/mf_v3_exact_jvp_b128.json \
  "$@"
