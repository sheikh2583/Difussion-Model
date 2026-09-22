#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$PROJECT_ROOT/venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "ERROR: project environment not found. Run ./scripts/linux/init.sh first." >&2; exit 1; }
SKIP_SUMMARY=false
ALLOW_RUNNING=false
DRY_RUN=false
PACKAGE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-summary) SKIP_SUMMARY=true; shift ;;
        --allow-running)
            ALLOW_RUNNING=true
            PACKAGE_ARGS+=("$1")
            shift ;;
        --dry-run)
            DRY_RUN=true
            PACKAGE_ARGS+=("$1")
            shift ;;
        --output|--max-mib)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value." >&2; exit 2; }
            PACKAGE_ARGS+=("$1" "$2")
            shift 2 ;;
        -h|--help)
            "$PYTHON" "$PROJECT_ROOT/scripts/package_thesis_context.py" --help
            echo "  --skip-summary   Package only after verifying existing canonical summaries"
            exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
    esac
done
cd "$PROJECT_ROOT"
if [[ "$SKIP_SUMMARY" == false ]]; then
    SUMMARY_ARGS=()
    [[ "$ALLOW_RUNNING" == false ]] || SUMMARY_ARGS+=(--allow-running)
    [[ "$DRY_RUN" == false ]] || SUMMARY_ARGS+=(--dry-run)
    "$SCRIPT_DIR/make_summary.sh" "${SUMMARY_ARGS[@]}"
fi
exec "$PYTHON" scripts/package_thesis_context.py "${PACKAGE_ARGS[@]}"
