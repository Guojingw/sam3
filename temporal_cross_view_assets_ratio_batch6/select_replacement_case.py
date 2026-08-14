#!/usr/bin/env python3
"""Rank unseen dataset objects for replacing an ambiguous temporal case."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
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


def target_frame_count(take_dir: Path, prefix: str) -> int:
    return sum(
        len(generator.image_files(path))
        for path in take_dir.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )


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
    target_frames = target_frame_count(case.take_dir, target_prefix)
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
        "warning_count": len(warnings),
    }


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
        default=["mug", "stainless"],
        help="Case-insensitive name fragment to exclude (repeatable).",
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
    avoid_terms = [term.lower() for term in args.avoid_term]

    rows = []
    cases = generator.discover_cases(args.data_root, "annotation.json", [])
    for position, case in enumerate(cases, start=1):
        if (
            case.case_id in existing
            or case.object_name in excluded_objects
            or case.take_id in excluded_takes
        ):
            continue
        if any(term in case.object_name.lower() for term in avoid_terms):
            continue
        try:
            row = score_case(case, args.source_prefix, args.target_prefix)
        except Exception as error:
            print(f"[WARN] {case.case_id}: {error}", file=sys.stderr)
            continue
        if not row:
            continue
        if row["best_source_mask_ratio"] < args.min_mask_ratio:
            continue
        if row["valid_source_mask_frames"] < args.min_source_frames:
            continue
        if row["target_frame_count"] < args.min_target_frames:
            continue
        rows.append(row)
        if position % 100 == 0:
            print(f"scanned {position}/{len(cases)}", file=sys.stderr)

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
    payload = {
        "eligible_case_count": len(rows),
        "eligible_take_count": len({row["take_id"] for row in rows}),
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
