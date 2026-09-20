#!/usr/bin/env sh
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ "$#" -eq 0 ]; then
  exec sh "$SCRIPT_DIR/setup.sh" --yes --datasets cifar10
fi
exec sh "$SCRIPT_DIR/setup.sh" --yes "$@"
