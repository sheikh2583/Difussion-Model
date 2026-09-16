#!/usr/bin/env bash
# setup.sh — thin wrapper around bootstrap.py
# Usage: ./scripts/setup.sh [--yes] [--gpu cuda118|cuda121|cuda128|rocm|cpu]
#                             [--datasets cifar10|celeba|all|none]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

if [[ -x "$VENV_PYTHON" ]] && \
   "$VENV_PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
  exec "$VENV_PYTHON" bootstrap.py "$@"
elif command -v python >/dev/null 2>&1 && \
   python -c 'import sys; raise SystemExit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
  exec python bootstrap.py "$@"
elif command -v python3 >/dev/null 2>&1 && \
     python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
  exec python3 bootstrap.py "$@"
else
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    SUDO=()
  elif command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
  else
    echo "ERROR: Python is missing and sudo is unavailable for package installation." >&2
    exit 1
  fi

  echo "Python 3.9+ was not found; installing it with the system package manager..."
  if command -v apt-get >/dev/null 2>&1; then
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y python3 python3-venv python3-pip
  elif command -v dnf >/dev/null 2>&1; then
    "${SUDO[@]}" dnf install -y python3 python3-pip
  elif command -v yum >/dev/null 2>&1; then
    "${SUDO[@]}" yum install -y python3 python3-pip
  elif command -v pacman >/dev/null 2>&1; then
    "${SUDO[@]}" pacman -Sy --needed --noconfirm python python-pip
  elif command -v zypper >/dev/null 2>&1; then
    "${SUDO[@]}" zypper --non-interactive install python3 python3-pip
  elif command -v apk >/dev/null 2>&1; then
    "${SUDO[@]}" apk add python3 py3-pip
  else
    echo "ERROR: no supported package manager found. Install Python 3.9+ and rerun." >&2
    exit 1
  fi

  if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
    echo "ERROR: the package manager installed Python older than 3.9." >&2
    exit 1
  fi
  exec python3 bootstrap.py "$@"
fi
