#!/usr/bin/env sh
# setup.sh — thin wrapper around bootstrap.py
# Usage: ./scripts/linux/setup.sh [--yes] [--gpu cuda118|cuda121|cuda128|rocm|cpu]
#                             [--datasets cifar10|celeba|all|none]
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

python_is_supported() {
  "$1" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' \
    >/dev/null 2>&1
}

python_can_create_venv() {
  "$1" -c 'import ensurepip, venv' >/dev/null 2>&1
}

install_python_support() {
  if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
  elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "ERROR: Python 3.10+ with venv support is required, and sudo is unavailable." >&2
    exit 1
  fi

  echo "Installing Python 3, pip, and virtual-environment support..."
  if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update
    $SUDO apt-get install -y python3 python3-venv python3-pip
  elif command -v dnf >/dev/null 2>&1; then
    $SUDO dnf install -y python3 python3-pip
  elif command -v yum >/dev/null 2>&1; then
    $SUDO yum install -y python3 python3-pip
  elif command -v pacman >/dev/null 2>&1; then
    $SUDO pacman -Sy --needed --noconfirm python python-pip
  elif command -v zypper >/dev/null 2>&1; then
    $SUDO zypper --non-interactive install python3 python3-pip
  elif command -v apk >/dev/null 2>&1; then
    $SUDO apk add python3 py3-pip
  else
    echo "ERROR: no supported package manager found. Install Python 3.10+ and rerun." >&2
    exit 1
  fi
}

find_bootstrap_python() {
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       python_is_supported "$candidate" && \
       python_can_create_venv "$candidate"; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

if [ -x "$VENV_PYTHON" ] && python_is_supported "$VENV_PYTHON"; then
  exec "$VENV_PYTHON" bootstrap.py "$@"
fi

BOOTSTRAP_PYTHON="$(find_bootstrap_python || true)"
if [ -z "$BOOTSTRAP_PYTHON" ]; then
  install_python_support
  BOOTSTRAP_PYTHON="$(find_bootstrap_python || true)"
fi

if [ -z "$BOOTSTRAP_PYTHON" ]; then
  echo "ERROR: Python 3.10+ with venv/ensurepip support is still unavailable." >&2
  echo "Install those packages for your distribution and rerun ./scripts/linux/init.sh." >&2
  exit 1
fi

exec "$BOOTSTRAP_PYTHON" bootstrap.py "$@"
