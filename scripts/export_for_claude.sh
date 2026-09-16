#!/usr/bin/env bash
# scripts/export_for_claude.sh
# Creates a zip of all essential source files for sharing with Claude web.
# Output: ../thesis_for_claude.zip  (one level above the project root)
#
# Usage:
#   ./scripts/export_for_claude.sh
#   ./scripts/export_for_claude.sh /tmp/snapshot.zip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
OUT="${1:-$(dirname "$ROOT")/thesis_for_claude.zip}"

echo ""
echo "  Scanning source files..."

cd "$ROOT"

zip -r "$OUT" . \
  --exclude "*/venv/*" \
  --exclude "*/__pycache__/*" \
  --exclude "*/.git/*" \
  --exclude "*/results/*" \
  --exclude "*/data/raw/*" \
  --exclude "*/archive/*" \
  --exclude "*.pyc" \
  --exclude "*.pyo" \
  --exclude "*.pt" \
  --exclude "*.zip" \
  --exclude "*.npz" \
  --exclude "*.gif" \
  --exclude "*.png" \
  --exclude "*.jpg" \
  --exclude "*.jpeg" \
  --exclude "*.pptx" \
  --exclude "*.pdf" \
  -q

SIZE=$(du -sh "$OUT" | cut -f1)
echo "  [OK] $OUT  ($SIZE)"
echo ""
