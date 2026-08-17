#!/usr/bin/env python3
"""Classify every temporal case by the next pipeline stage it requires."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def classify(case_dir: Path) -> tuple[str, str]:
    result_path = case_dir / "temporal_analysis_result.json"
    if not result_path.is_file():
        return "needs_qwen", "missing temporal_analysis_result.json"
    try:
        result = read_json(result_path)
    except Exception as error:
        return "needs_qwen", f"unreadable result: {error}"

    schema = int(result.get("schema_version", 0) or 0)
    status = str(result.get("status", ""))
    pipeline = str(result.get("pipeline_status", ""))
    best = result.get("qwen_temporal_selection") or result.get("best_segment")

    if schema >= 13 and pipeline == "complete":
        segmented = int(
            (result.get("final_segmentation") or {}).get(
                "segmented_frame_count", 0
            )
            or 0
        )
        if status == "success":
            temporal_only = (
                (result.get("final_segmentation") or {}).get("status")
                == "not_requested"
            )
            return (
                "complete",
                f"schema={schema} temporal_selection_complete "
                f"sam3_optional={str(temporal_only).lower()} "
                f"segmented_frames={segmented}",
            )
        return "complete_uncertain", "Qwen found no verified temporal window"
    if (
        schema == 11
        and pipeline == "awaiting_final_sam3_segmentation"
        and status == "success"
        and best
    ):
        return "needs_sam3", f"window={best.get('window_id', 'unknown')}"
    if status == "failed":
        return "needs_qwen", f"previous failure: {result.get('error', '')}"
    return (
        "needs_qwen",
        f"legacy/incomplete schema={schema} pipeline={pipeline or '-'}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = []
    for case_dir in sorted(args.assets_root.resolve().glob("*__*")):
        if not (case_dir / "metadata.json").is_file():
            continue
        stage, detail = classify(case_dir)
        rows.append({"case_id": case_dir.name, "stage": stage, "detail": detail})

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for row in rows:
            print(f"{row['stage']}\t{row['case_id']}\t{row['detail']}")
        counts = {
            stage: sum(row["stage"] == stage for row in rows)
            for stage in (
                "needs_qwen",
                "needs_sam3",
                "complete",
                "complete_uncertain",
            )
        }
        print(
            "SUMMARY\t"
            + " ".join(f"{key}={value}" for key, value in counts.items())
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
