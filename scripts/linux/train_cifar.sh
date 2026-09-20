#!/usr/bin/env bash
set -euo pipefail
PLATFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PLATFORM_DIR/train_all_datasets.sh" --dataset cifar10 "$@"
