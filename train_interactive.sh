#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PYTHON="$PROJECT_ROOT/venv/bin/python"
if [[ ! -x "$PROJECT_PYTHON" ]]; then
  echo "ERROR: project environment not found. Run ./init_all.sh first." >&2
  exit 1
fi
cd "$PROJECT_ROOT"
exec "$PROJECT_PYTHON" scripts/interactive_train.py "$@"
