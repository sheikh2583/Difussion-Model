#!/usr/bin/env sh
set -eu
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ "$#" -eq 0 ]; then
  exec "$PROJECT_ROOT/scripts/setup.sh" --yes --datasets cifar10
fi
exec "$PROJECT_ROOT/scripts/setup.sh" --yes "$@"
