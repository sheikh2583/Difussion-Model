#!/usr/bin/env sh
set -eu
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$PROJECT_ROOT/scripts/setup.sh" --yes --datasets all "$@"
