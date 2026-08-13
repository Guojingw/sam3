#!/bin/bash
# Submit one independent one-GPU SAM3 reranking job per temporal case.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 ASSETS_ROOT WORK_DIR" >&2
  exit 2
fi

ASSETS="$(readlink -f "$1")"
mkdir -p "$2"
WORK_DIR="$(readlink -f "$2")"
RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_DIR="$WORK_DIR/job_summaries/sam3"
mkdir -p "$SUMMARY_DIR"

index=0
for case_dir in "$ASSETS"/*__*; do
  [[ -d "$case_dir" ]] || continue
  [[ -f "$case_dir/metadata.json" ]] || continue
  [[ -f "$case_dir/temporal_window_index.json" ]] || continue
  case_id="$(basename "$case_dir")"
  result_path="$case_dir/temporal_analysis_result.json"
  if ! jq -e '
      .schema_version == 11 and
      .pipeline_status == "awaiting_sam3_rerank" and
      ((.sam3_rerank_candidates // []) | length) > 0
    ' "$result_path" >/dev/null 2>&1; then
    echo "Skipping $case_id: no schema-11 SAM3 candidates."
    continue
  fi
  index=$((index + 1))
  summary_path="$SUMMARY_DIR/$case_id.json"
  job_id="$(
    qsub \
      -N "sam3_case_${index}" \
      -v "ASSETS=$ASSETS,WORK_DIR=$WORK_DIR,CASE_FILTER=$case_id,SUMMARY_PATH=$summary_path" \
      "$RUNNER_DIR/run_sam3_temporal_rerank.pbs"
  )"
  printf '%s\t%s\t%s\n' "$job_id" "$case_id" "$summary_path"
done

if [[ $index -eq 0 ]]; then
  echo "No schema-11 cases require SAM3 reranking."
  exit 0
fi

echo "Submitted $index independent SAM3 jobs."
