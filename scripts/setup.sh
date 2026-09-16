#!/usr/bin/env bash
# setup.sh — thin wrapper around bootstrap.py
# Usage: ./scripts/setup.sh [--yes] [--gpu cuda118|cuda121|rocm|cpu]
#                             [--datasets cifar10|celeba|all|none]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
if command -v python >/dev/null 2>&1; then
  exec python bootstrap.py "$@"
elif command -v python3 >/dev/null 2>&1; then
  exec python3 bootstrap.py "$@"
else
  echo "Python 3.9+ is required. Install Python and rerun this script." >&2
  exit 1
fi
