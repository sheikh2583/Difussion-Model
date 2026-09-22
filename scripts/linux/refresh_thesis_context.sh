#!/usr/bin/env bash
# Periodically rebuild the canonical summary and verified Claude handoff ZIP.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INTERVAL=300
ITERATIONS=0
ALLOW_RUNNING=false
DRY_RUN=false
CONTEXT_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./scripts/linux/refresh_thesis_context.sh [OPTIONS]

Repeatedly rebuild THESIS_SUMMARY.md, results/aggregate, the training-log
catalog, and the verified thesis_context.zip. No training, sampling, or
evaluation is started.

Options:
  --interval SECONDS   Delay between attempts (default: 300; minimum: 10)
  --iterations COUNT   Stop after COUNT attempts; 0 means until interrupted
  --once               Equivalent to --iterations 1
  --allow-running      Permit read-only snapshots while training is active
  --dry-run            Validate and print one cycle without writing
  --output PATH        Forward a non-default ZIP output path
  --max-mib SIZE       Forward the uncompressed archive size ceiling
  -h, --help           Show this help

Without --allow-running, an active trainer makes that attempt fail safely; the
watcher retries later. Press Ctrl-C to stop after the current command exits.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value." >&2; exit 2; }
            INTERVAL="$2"; shift 2 ;;
        --iterations)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value." >&2; exit 2; }
            ITERATIONS="$2"; shift 2 ;;
        --once) ITERATIONS=1; shift ;;
        --allow-running) ALLOW_RUNNING=true; CONTEXT_ARGS+=("$1"); shift ;;
        --dry-run) DRY_RUN=true; CONTEXT_ARGS+=("$1"); ITERATIONS=1; shift ;;
        --output|--max-mib)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value." >&2; exit 2; }
            CONTEXT_ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$INTERVAL" =~ ^[0-9]+$ ]] && (( INTERVAL >= 10 )) || {
    echo "ERROR: --interval must be an integer of at least 10 seconds." >&2
    exit 2
}
[[ "$ITERATIONS" =~ ^[0-9]+$ ]] || {
    echo "ERROR: --iterations must be a non-negative integer." >&2
    exit 2
}

command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required to prevent duplicate refresh watchers." >&2
    exit 1
}
LOCK_KEY="$(printf '%s' "$PROJECT_ROOT" | cksum | awk '{print $1}')"
LOCK_PATH="${XDG_RUNTIME_DIR:-/tmp}/diffusion-thesis-context-$LOCK_KEY.lock"
exec 9>"$LOCK_PATH"
flock -n 9 || {
    echo "ERROR: another thesis-context refresh watcher owns $LOCK_PATH" >&2
    exit 4
}

attempt=0
last_status=0
while (( ITERATIONS == 0 || attempt < ITERATIONS )); do
    attempt=$((attempt + 1))
    printf '[%s] thesis-context refresh attempt %d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$attempt"
    if "$SCRIPT_DIR/make_thesis_context.sh" "${CONTEXT_ARGS[@]}"; then
        last_status=0
        printf '[%s] refresh verified\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    else
        last_status=$?
        printf '[%s] refresh failed safely (status %d); existing ZIP was preserved\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$last_status" >&2
    fi
    (( ITERATIONS != 0 && attempt >= ITERATIONS )) && break
    sleep "$INTERVAL"
done

exit "$last_status"
