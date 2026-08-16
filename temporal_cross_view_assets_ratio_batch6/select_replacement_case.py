#!/usr/bin/env python3
"""Rank unseen dataset objects for replacing an ambiguous temporal case."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

generator: Any = None


def existing_take_ids(assets_root: Path | None) -> set[str]:
    """Read take IDs already represented in an existing assets batch."""
    take_ids: set[str] = set()
    if not assets_root or not assets_root.is_dir():
        return take_ids
    for metadata_path in assets_root.glob("*__*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        take_id = str(metadata.get("take_id", "")).strip()
        if take_id:
            take_ids.add(take_id)
    return take_ids


def diverse_recommendations(
    rows: list[dict[str, Any]],
    limit: int,
    one_per_take: bool = True,
    one_per_object: bool = True,
) -> list[dict[str, Any]]:
    """Keep the best case per take and object unless explicitly disabled."""
    selected: list[dict[str, Any]] = []
    seen_takes: set[str] = set()
    seen_objects: set[str] = set()
    for row in rows:
        take_id = str(row["take_id"])
        object_name = str(row["object_name"])
        if one_per_take and take_id in seen_takes:
            continue
        if one_per_object and object_name in seen_objects:
            continue
        selected.append(row)
        seen_takes.add(take_id)
        seen_objects.add(object_name)
        if len(selected) >= max(1, limit):
            break
    return selected


def largest_contiguous_run(
    frame_ids: list[int], max_gap_factor: float = 2.5
) -> int:
    if not frame_ids:
        return 0
    ordered = sorted(set(frame_ids))
    gaps = [right - left for left, right in zip(ordered, ordered[1:])]
    positive = sorted(gap for gap in gaps if gap > 0)
    if not positive:
        return 1
    middle = len(positive) // 2
    median_gap = (
        float(positive[middle])
        if len(positive) % 2
        else (positive[middle - 1] + positive[middle]) / 2.0
    )
    threshold = median_gap * max_gap_factor
    longest = current = 1
    for gap in gaps:
        if gap <= threshold:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return max(longest, current)


def target_frame_statistics(take_dir: Path, prefix: str) -> dict[str, Any]:
    total = 0
    largest_run = 0
    viable_camera_count = 0
    for path in take_dir.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        frame_map, _ = generator.build_frame_map(path)
        camera_count = len(frame_map)
        camera_run = largest_contiguous_run(list(frame_map))
        total += camera_count
        largest_run = max(largest_run, camera_run)
        minimum_window = max(2, math.floor(camera_count * 0.20))
        if camera_run >= minimum_window:
            viable_camera_count += 1
    return {
        "target_frame_count": total,
        "largest_contiguous_target_run": largest_run,
        "viable_target_camera_count": viable_camera_count,
    }


def score_case(
    case: Any,
    source_prefix: str,
    target_prefix: str,
) -> dict[str, Any] | None:
    annotation = generator.load_json(case.annotation_path)
    ranked, _, warnings = generator.rank_source_masks(
        annotation, case, source_prefix
    )
    valid = [item for item in ranked if generator.usable_source(item)]
    if not valid:
        return None
    ratios = [float(item.mask_area_ratio) for item in valid]
    target = target_frame_statistics(case.take_dir, target_prefix)
    target_frames = int(target["target_frame_count"])
    # Favor a large, repeatedly visible source mask and a substantial target
    # timeline. This is a data-quality ranking, not a semantic success claim.
    quality = (
        min(max(ratios) / 0.05, 1.0) * 0.55
        + min(median(ratios) / 0.02, 1.0) * 0.20
        + min(len(valid) / 20.0, 1.0) * 0.15
        + min(target_frames / 200.0, 1.0) * 0.10
    )
    return {
        "case_id": case.case_id,
        "take_id": case.take_id,
        "object_name": case.object_name,
        "quality_score": round(quality, 4),
        "best_source_mask_ratio": round(max(ratios), 8),
        "median_source_mask_ratio": round(median(ratios), 8),
        "valid_source_mask_frames": len(valid),
        "target_frame_count": target_frames,
        "largest_contiguous_target_run": target[
            "largest_contiguous_target_run"
        ],
        "viable_target_camera_count": target["viable_target_camera_count"],
        "warning_count": len(warnings),
    }


def quality_rejection_reasons(
    row: dict[str, Any],
    min_mask_ratio: float,
    min_source_frames: int,
    min_target_frames: int,
) -> list[str]:
    reasons = []
    if row["best_source_mask_ratio"] < min_mask_ratio:
        reasons.append("source_mask_ratio_below_minimum")
    if row["valid_source_mask_frames"] < min_source_frames:
        reasons.append("source_mask_frames_below_minimum")
    if row["target_frame_count"] < min_target_frames:
        reasons.append("target_frames_below_minimum")
    if row["viable_target_camera_count"] < 1:
        reasons.append("no_contiguous_20_percent_target_window")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--source-prefix", default="aria")
    parser.add_argument("--target-prefix", default="cam")
    parser.add_argument("--min-mask-ratio", type=float, default=0.01)
    parser.add_argument("--min-source-frames", type=int, default=3)
    parser.add_argument("--min-target-frames", type=int, default=100)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print scan progress after this many discovered cases.",
    )
    parser.add_argument("--exclude-object", action="append", default=[])
    parser.add_argument("--exclude-take", action="append", default=[])
    parser.add_argument(
        "--allow-existing-takes",
        action="store_true",
        help="Do not exclude take IDs already represented by --assets-root.",
    )
    parser.add_argument(
        "--allow-multiple-per-take",
        action="store_true",
        help="Allow more than one recommended object from the same scene/take.",
    )
    parser.add_argument(
        "--allow-duplicate-objects",
        action="store_true",
        help="Allow the same object label to be recommended in multiple takes.",
    )
    parser.add_argument(
        "--avoid-term",
        action="append",
        default=[],
        help="Case-insensitive name fragment to exclude (repeatable).",
    )
    parser.add_argument(
        "--no-default-avoid-terms",
        action="store_true",
        help="Include mug/stainless labels; exact exclusions still apply.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    global generator
    try:
        generator = importlib.import_module("generate_temporal_cross_view_assets")
    except ModuleNotFoundError as error:
        parser.error(
            f"missing generator dependency {error.name!r}; run this with the "
            "same conda environment used to generate the temporal assets"
        )

    existing = set()
    if args.assets_root and args.assets_root.is_dir():
        existing = {
            path.name for path in args.assets_root.glob("*__*") if path.is_dir()
        }
    excluded_objects = set(args.exclude_object)
    excluded_takes = set(args.exclude_take)
    represented_takes = existing_take_ids(args.assets_root)
    if not args.allow_existing_takes:
        excluded_takes.update(represented_takes)
    avoid_terms = [] if args.no_default_avoid_terms else ["mug", "stainless"]
    avoid_terms.extend(term.lower() for term in args.avoid_term)

    rows = []
    near_misses = []
    rejection_counts: Counter[str] = Counter()
    cases = generator.discover_cases(args.data_root, "annotation.json", [])
    print(
        f"Discovered {len(cases)} object cases under {args.data_root}; "
        f"excluded_takes={len(excluded_takes)} existing_cases={len(existing)}",
        file=sys.stderr,
        flush=True,
    )
    for position, case in enumerate(cases, start=1):
        if position == 1 or (
            args.progress_every > 0 and position % args.progress_every == 0
        ):
            print(
                f"[scan {position}/{len(cases)}] eligible_so_far={len(rows)} "
                f"current={case.case_id}",
                file=sys.stderr,
                flush=True,
            )
        if case.case_id in existing:
            rejection_counts["case_already_in_assets"] += 1
            continue
        if case.object_name in excluded_objects:
            rejection_counts["explicitly_excluded_object"] += 1
            continue
        if case.take_id in excluded_takes:
            rejection_counts["take_already_represented_or_excluded"] += 1
            continue
        if any(term in case.object_name.lower() for term in avoid_terms):
            rejection_counts["object_name_avoid_term"] += 1
            continue
        try:
            row = score_case(case, args.source_prefix, args.target_prefix)
        except Exception as error:
            rejection_counts["case_scan_error"] += 1
            print(
                f"[WARN] {case.case_id}: {error}",
                file=sys.stderr,
                flush=True,
            )
            continue
        if not row:
            rejection_counts["no_usable_source_mask"] += 1
            continue
        reasons = quality_rejection_reasons(
            row,
            args.min_mask_ratio,
            args.min_source_frames,
            args.min_target_frames,
        )
        if reasons:
            rejection_counts.update(reasons)
            near_misses.append({**row, "rejection_reasons": reasons})
            continue
        rows.append(row)
        rejection_counts["eligible"] += 1

    print(
        f"Scan complete: discovered={len(cases)} eligible={len(rows)}",
        file=sys.stderr,
        flush=True,
    )

    rows.sort(
        key=lambda row: (
            row["quality_score"],
            row["best_source_mask_ratio"],
            row["valid_source_mask_frames"],
        ),
        reverse=True,
    )
    output = diverse_recommendations(
        rows,
        args.top_k,
        one_per_take=not args.allow_multiple_per_take,
        one_per_object=not args.allow_duplicate_objects,
    )
    near_misses.sort(
        key=lambda row: (
            -len(row["rejection_reasons"]),
            row["quality_score"],
            row["best_source_mask_ratio"],
        ),
        reverse=True,
    )
    diverse_near_misses = diverse_recommendations(near_misses, 20)
    eligible_take_summaries = []
    for take_id in sorted({row["take_id"] for row in rows}):
        take_rows = [row for row in rows if row["take_id"] == take_id]
        best = max(take_rows, key=lambda row: row["quality_score"])
        eligible_take_summaries.append(
            {
                "take_id": take_id,
                "eligible_case_count": len(take_rows),
                "best_case_id": best["case_id"],
                "best_quality_score": best["quality_score"],
            }
        )
    payload = {
        "discovered_case_count": len(cases),
        "eligible_case_count": len(rows),
        "eligible_take_count": len({row["take_id"] for row in rows}),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "eligible_take_summaries": eligible_take_summaries,
        "near_miss_take_count": len(
            {row["take_id"] for row in near_misses}
        ),
        "near_misses": diverse_near_misses,
        "excluded_existing_take_ids": sorted(represented_takes),
        "one_recommendation_per_take": not args.allow_multiple_per_take,
        "one_recommendation_per_object": not args.allow_duplicate_objects,
        "recommendations": output,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if output else 1


if __name__ == "__main__":
    raise SystemExit(main())
