#!/usr/bin/env bash
# setup.sh — thin wrapper around bootstrap.py
# Usage: ./scripts/setup.sh [--yes] [--gpu cuda118|cuda121|rocm|cpu]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
exec python bootstrap.py "$@"
