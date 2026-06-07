#!/usr/bin/env bash
# Run baseline_run.sh across several DS4_MTP_MIN_MARGIN values; collect a
# matrix of (class, margin) -> accept_rate and tps. Writes a combined
# pareto-style table.
#
# Usage: margin_sweep.sh [--margins "0,0.5,1,2,4"]

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MARGINS="${1:-0,0.5,1.0,2.0,4.0}"
[[ "${1:-}" == "--margins" ]] && MARGINS="$2"

OUT="$HERE/runs/margin-sweep-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

IFS=',' read -ra MLIST <<< "$MARGINS"

for m in "${MLIST[@]}"; do
    # margin=0 means default (no env var); skip the explicit margin flag
    if awk "BEGIN {exit !($m > 0)}"; then
        "$HERE/baseline_run.sh" --margin "$m" --label "margin${m}" 2>&1 | tail -20
        run_dir=$(ls -1dt "$HERE/runs/margin${m}-"* | head -1)
    else
        "$HERE/baseline_run.sh" --label "margin0-default" 2>&1 | tail -20
        run_dir=$(ls -1dt "$HERE/runs/margin0-default-"* | head -1)
    fi
    cp -r "$run_dir" "$OUT/margin_${m}"
done

# Combined matrix
echo "===== MATRIX: accept_rate(%) and implied_tps ====="
python3 - "$OUT" <<'PY'
import json, glob, sys, os
out = sys.argv[1]
margins = sorted([d for d in os.listdir(out) if d.startswith("margin_")],
                 key=lambda d: float(d.split("_", 1)[1]))
classes = ["prose-continuation","chat-essay","code-generation","analytical-qa","structured-list"]
print(f"\n## Accept rate")
print(f"| class | " + " | ".join(d.split("_",1)[1] for d in margins) + " |")
print(f"| ----- | " + " | ".join("---" for _ in margins) + " |")
for c in classes:
    row = [f"| {c}"]
    for m in margins:
        p = f"{out}/{m}/{c}.json"
        if not os.path.isfile(p): row.append(""); continue
        d = json.load(open(p))
        row.append(f"{d.get('accept_rate',0)*100:.1f}%" if d.get("total_steps") else "—")
    print(" | ".join(row) + " |")

print(f"\n## Implied decode tps (from MTP step times)")
print(f"| class | " + " | ".join(d.split("_",1)[1] for d in margins) + " |")
print(f"| ----- | " + " | ".join("---" for _ in margins) + " |")
for c in classes:
    row = [f"| {c}"]
    for m in margins:
        p = f"{out}/{m}/{c}.json"
        if not os.path.isfile(p): row.append(""); continue
        d = json.load(open(p))
        row.append(f"{d.get('implied_decode_tps',0):.2f}" if d.get("total_steps") else "—")
    print(" | ".join(row) + " |")
PY
