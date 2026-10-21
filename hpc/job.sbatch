#!/bin/bash
# Resources and array limits are supplied from config.json by the common entry.
set -euo pipefail
root=$1
snapshot=$2
work=$3
phase=$4
if [[ $7 == true ]]; then
  [[ -z $5 ]] || module load "$5"
  conda activate "$6"
fi
runtime_exports=$(python3 "$root/pipeline.py" _env "$snapshot")
eval "$runtime_exports"
if [[ $phase == predict ]]; then
  shift 7
  exec python3 "$root/deployment/predict.py" --run "$work/deployment/run01" "$@"
fi
exec python3 "$root/pipeline.py" _worker "$snapshot" "$work" "$phase"
