#!/bin/bash
set -euo pipefail

job_name="eval"
if [[ $# -gt 0 && "$1" != --* ]]; then
  job_name="$1"
  shift
fi

config="configs/load_config.py:eval"
workdir=""
mode="remote_run"
load_from=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      config="$2"
      shift 2
      ;;
    --workdir)
      workdir="$2"
      shift 2
      ;;
    --mode)
      mode="$2"
      shift 2
      ;;
    --load_from)
      load_from="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$workdir" ]]; then
  run_id="$(date '+%Y%m%d_%H%M%S')_${job_name}"
  workdir="runs/user/${run_id}"
fi
mkdir -p "$workdir"

args=()
if [[ -n "$load_from" ]]; then
  args+=(--load_from="$load_from")
fi

python3 main.py \
  --workdir="$workdir" \
  --config="$config" \
  --mode="$mode" \
  "${args[@]}" \
  2>&1 | tee -a "$workdir/output.log"
