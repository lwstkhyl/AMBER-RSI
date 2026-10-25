#!/bin/bash
# Usage: bash plotting/replot_helvetica.sh RUN_DIR NEW_OUTPUT_DIR HELVETICA_FONT_DIR
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: bash $0 RUN_DIR NEW_OUTPUT_DIR HELVETICA_FONT_DIR" >&2
  exit 2
fi
repo_root=$(cd "$(dirname "$0")/.." && pwd)
"${PYTHON:-python}" - "$repo_root" "$1" "$2" "$3" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

repo, work, out, fonts = [Path(value).expanduser().resolve() for value in sys.argv[1:]]
if out == work or work in out.parents:
    raise SystemExit("Choose a new output directory outside the existing run.")
if not fonts.is_dir():
    raise SystemExit(f"Helvetica font directory not found: {fonts}")
source_config = work / "config.snapshot.json"
if not source_config.is_file():
    raise SystemExit(f"Run configuration not found: {source_config}")
required_tables = [
    "summary/total_results.tsv", "summary/position_predictions.tsv",
    "summary/target_metrics.tsv", "plot_data/distance_per_target.tsv",
    "plot_data/observed_window_summary.tsv", "transfer/summary/condition_metrics.tsv",
    "transfer/summary/editing_window_curve.tsv", "transfer/summary/metrics_by_length.tsv",
]
missing = [name for name in required_tables if not (work / name).is_file()]
if missing:
    raise SystemExit("Missing plotting inputs:\n" + "\n".join(missing))
config = json.loads(source_config.read_text(encoding="utf-8"))
config["figures"].update(
    model_name="RLDpred-BE", font_family="Helvetica",
    min_font_size=12, font_dir=str(fonts),
)
out.mkdir(parents=True, exist_ok=True)
plot_config = out / "config.plot.local.json"
plot_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Plot configuration: {plot_config}", flush=True)
subprocess.run([
    sys.executable, str(repo / "plotting/plot_figures2.py"),
    "--config", str(plot_config), "--work", str(work), "--out", str(out),
], check=True)
PY
