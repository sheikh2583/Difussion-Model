#!/usr/bin/env sh
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [ "${1:-}" = "--all-latent" ]; then
  shift
  cd "$PROJECT_ROOT"
  exec bash scripts/linux/train_celeba_latent.sh --only all "$@"
fi

PROJECT_PYTHON="$PROJECT_ROOT/venv/bin/python"
if [ ! -x "$PROJECT_PYTHON" ]; then
  echo "ERROR: project environment not found. Run ./scripts/linux/init.sh first" >&2
  exit 1
fi
cd "$PROJECT_ROOT"
exec "$PROJECT_PYTHON" scripts/interactive_train.py "$@"
