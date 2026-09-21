#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$PROJECT_ROOT/venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "ERROR: project environment not found. Run ./scripts/linux/init.sh first." >&2; exit 1; }
cd "$PROJECT_ROOT"
exec "$PYTHON" scripts/package_thesis_context.py "$@"
