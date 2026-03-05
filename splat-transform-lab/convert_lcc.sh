#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./convert_lcc.sh <input.lcc> [output-prefix]"
  exit 1
fi

INPUT="$1"
OUT="${2:-output}"

echo "[1/3] LCC -> HTML"
splat-transform -w "$INPUT" "$OUT.html"

echo "[2/3] LCC -> PLY"
splat-transform -w "$INPUT" "$OUT.ply"

echo "[3/3] LCC -> SOG"
splat-transform -w "$INPUT" "$OUT.sog"

echo "Done."
