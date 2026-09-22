#!/usr/bin/env bash
# Shared shell adapter for utils/gpu_lock.py and utils/source_identity.py.

WORKFLOW_LOCK_FILE="results/.lock"
WORKFLOW_LOCK_TOKEN=""
WORKFLOW_LOCK_OWNED=false

workflow_guard_release() {
  if [[ "$WORKFLOW_LOCK_OWNED" == true && -n "$WORKFLOW_LOCK_TOKEN" ]]; then
    "$PYTHON" scripts/workflow_guard.py lock-release \
      --lock-file "$WORKFLOW_LOCK_FILE" --token "$WORKFLOW_LOCK_TOKEN"
  fi
  WORKFLOW_LOCK_OWNED=false
}

workflow_guard_start() {
  local command_name=$1 dry_run=$2
  shift 2
  local -a identity_args=("$@")

  if [[ -z "${DIFFUSION_PARENT_SUITE_TIMESTAMP:-}" ]]; then
    export DIFFUSION_PARENT_SUITE_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  if [[ "$dry_run" == true ]]; then
    echo "[DRY-RUN] GPU lock: $WORKFLOW_LOCK_FILE"
    echo "[DRY-RUN] training source identity will be frozen before execution"
    return 0
  fi

  local inherited_token="${DIFFUSION_GPU_LOCK_TOKEN:-}"
  local -a lock_args=(
    lock-acquire --lock-file "$WORKFLOW_LOCK_FILE" --command "$command_name"
    --owner-pid "$$"
  )
  if [[ -n "$inherited_token" ]]; then
    lock_args+=(--token "$inherited_token")
  fi
  IFS=$'\t' read -r WORKFLOW_LOCK_TOKEN WORKFLOW_LOCK_OWNED < <(
    "$PYTHON" scripts/workflow_guard.py "${lock_args[@]}"
  )
  export DIFFUSION_GPU_LOCK_TOKEN="$WORKFLOW_LOCK_TOKEN"
  trap workflow_guard_release EXIT
  trap 'workflow_guard_release; exit 130' INT
  trap 'workflow_guard_release; exit 143' TERM

  if [[ -n "${DIFFUSION_SOURCE_MANIFEST:-}" ]]; then
    "$PYTHON" scripts/workflow_guard.py source-verify \
      --manifest "$DIFFUSION_SOURCE_MANIFEST" >/dev/null
  else
    local manifest="results/source_manifests/${DIFFUSION_PARENT_SUITE_TIMESTAMP}_pid$$.json"
    export DIFFUSION_SOURCE_IDENTITY
    DIFFUSION_SOURCE_IDENTITY="$(
      "$PYTHON" scripts/workflow_guard.py source-freeze \
        --manifest "$manifest" "${identity_args[@]}"
    )"
    export DIFFUSION_SOURCE_MANIFEST="$manifest"
  fi
}

workflow_guard_verify_source() {
  local dry_run=$1
  if [[ "$dry_run" == false ]]; then
    "$PYTHON" scripts/workflow_guard.py source-verify \
      --manifest "$DIFFUSION_SOURCE_MANIFEST" >/dev/null
  fi
}

workflow_device_log_dir() {
  local category=$1 gpu_name="" gpu_memory_mb="" gpu_token=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n '1p')"
    gpu_memory_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sed -n '1p' | tr -d ' ')"
  fi
  if [[ -n "$gpu_name" ]]; then
    gpu_token="$(printf '%s' "$gpu_name" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-')"
    gpu_token="${gpu_token%-}"
    if [[ "$gpu_memory_mb" =~ ^[0-9]+$ ]]; then
      gpu_token="${gpu_token}-$(((gpu_memory_mb + 1023) / 1024))gb"
    fi
    printf 'training_logs/%s/%s\n' "$gpu_token" "$category"
  else
    printf 'training_logs/cpu-or-unknown/%s\n' "$category"
  fi
}
