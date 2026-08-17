#!/bin/bash
# Submit one independent one-GPU PBS job per temporal case.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 ASSETS_ROOT WORK_DIR" >&2
  exit 2
fi

ASSETS="$(readlink -f "$1")"
mkdir -p "$2"
WORK_DIR="$(readlink -f "$2")"
RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_DIR="$WORK_DIR/job_summaries/qwen"
mkdir -p "$SUMMARY_DIR"

index=0
for case_dir in "$ASSETS"/*__*; do
  [[ -d "$case_dir" ]] || continue
  [[ -f "$case_dir/metadata.json" ]] || continue
  [[ -f "$case_dir/temporal_window_index.json" ]] || continue
  case_id="$(basename "$case_dir")"
  index=$((index + 1))
  summary_path="$SUMMARY_DIR/$case_id.json"
  job_id="$(
    qsub \
      -N "q4b_case_${index}" \
      -v "ASSETS=$ASSETS,WORK_DIR=$WORK_DIR,CASE_FILTER=$case_id,SUMMARY_PATH=$summary_path" \
      "$RUNNER_DIR/run_qwen_temporal_batch.pbs"
  )"
  printf '%s\t%s\t%s\n' "$job_id" "$case_id" "$summary_path"
done

if [[ $index -eq 0 ]]; then
  echo "No case directories found under $ASSETS" >&2
  exit 1
fi

echo "Submitted $index independent Qwen jobs."
