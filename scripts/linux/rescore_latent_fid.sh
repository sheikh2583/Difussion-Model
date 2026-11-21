#!/usr/bin/env bash
# Re-score retained latent checkpoints against the raw-pixel CelebA FID reference.
#
#   ./scripts/linux/rescore_latent_fid.sh --dry-run
#   ./scripts/linux/rescore_latent_fid.sh
#   ./scripts/linux/rescore_latent_fid.sh --experiments fm_celeba_latent
#
# Outputs go to results/aggregate/latent_rescored/; results/<experiment>/ is
# never modified. The run is resumable and holds the shared GPU lock.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "ERROR: run ./scripts/linux/init.sh first." >&2; exit 1; }

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiments)
      shift
      ARGS+=(--experiments)
      while [[ $# -gt 0 && "$1" != --* ]]; do ARGS+=("$1"); shift; done ;;
    --output-dir) ARGS+=(--output-dir "${2:?--output-dir requires a path}"); shift 2 ;;
    --device) ARGS+=(--device "${2:?--device requires a value}"); shift 2 ;;
    --dry-run) ARGS+=(--dry-run); shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

exec "$PYTHON" scripts/rescore_latent_fid.py "${ARGS[@]}"
