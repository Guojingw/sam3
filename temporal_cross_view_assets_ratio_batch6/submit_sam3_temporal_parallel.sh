#!/bin/bash
# Submit one independent final-visualization SAM3 job per selected case.

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
  if [[ ! -f "$result_path" ]]; then
    echo "Skipping $case_id: missing Qwen result; run Qwen first."
    continue
  fi
  if ! jq -e '
      .status == "success" and
      .best_segment != null and
      ((.schema_version == 11 and
        .pipeline_status == "awaiting_final_sam3_segmentation") or
       (.schema_version >= 14 and
        .pipeline_status == "complete" and
        ((.final_segmentation.status == "not_requested") or
         ((.final_segmentation.schema_version // 0) < 2))))
    ' "$result_path" >/dev/null 2>&1; then
    schema="$(jq -r '.schema_version // 0' "$result_path" 2>/dev/null || echo 0)"
    pipeline="$(jq -r '.pipeline_status // "legacy"' "$result_path" 2>/dev/null || echo unreadable)"
    echo "Skipping $case_id: schema=$schema pipeline=$pipeline; run current Qwen first unless already complete."
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
  echo "No selected cases requested optional SAM3 masks."
  exit 0
fi

echo "Submitted $index independent final-mask jobs."
