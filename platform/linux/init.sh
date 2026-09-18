#!/usr/bin/env sh
set -eu
PLATFORM_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$PLATFORM_DIR")")"
exec sh "$PROJECT_ROOT/init_all.sh" "$@"

