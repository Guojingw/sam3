#!/usr/bin/env python3
"""Rerank dense temporal candidates using SAM3 mask tracks."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

import qwen_temporal_runner as temporal


RERANK_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 12
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--minimum-score-margin", type=float, default=0.03)
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def box_xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    return [
        float(box[0]),
        float(box[1]),
        float(box[2]) - float(box[0]),
        float(box[3]) - float(box[1]),
    ]


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = temporal.normalized_box_area(left)
    right_area = temporal.normalized_box_area(right)
    return intersection / max(1e-9, left_area + right_area - intersection)


def mask_bbox(mask: np.ndarray) -> list[float] | None:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None
    height, width = mask.shape
    return [
        float(columns.min() / width),
        float(rows.min() / height),
        float((columns.max() + 1) / width),
        float((rows.max() + 1) / height),
    ]


def output_arrays(outputs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_object_ids = outputs.get("out_obj_ids", [])
    raw_masks = outputs.get("out_binary_masks", [])
    if hasattr(raw_object_ids, "detach"):
        raw_object_ids = raw_object_ids.detach().cpu().numpy()
    if hasattr(raw_masks, "detach"):
        raw_masks = raw_masks.detach().cpu().numpy()
    object_ids = np.asarray(raw_object_ids).reshape(-1)
    masks = np.asarray(raw_masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    return object_ids, masks


def output_mask_for_object(
    outputs: Mapping[str, Any], object_id: int | None
) -> tuple[int | None, np.ndarray | None]:
    object_ids, masks = output_arrays(outputs)
    if len(object_ids) == 0 or len(masks) == 0:
        return object_id, None
    if object_id is None:
        object_id = int(object_ids[0])
    matches = np.nonzero(object_ids.astype(int) == int(object_id))[0]
    if len(matches) == 0:
        return object_id, None
    return object_id, masks[int(matches[0])].astype(bool)


def prompt_aligned_output_mask(
    outputs: Mapping[str, Any], prompt_box: Sequence[float]
) -> tuple[int | None, np.ndarray | None, float]:
    """Select the prompted SAM object, never an arbitrary first object ID."""
    object_ids, masks = output_arrays(outputs)
    candidates = []
    for index, object_id in enumerate(object_ids.astype(int)):
        if index >= len(masks):
            continue
        mask = masks[index].astype(bool)
        box = mask_bbox(mask)
        if box is None:
            continue
        candidates.append((box_iou(box, prompt_box), int(object_id), mask))
    if not candidates:
        return None, None, 0.0
    overlap, object_id, mask = max(candidates, key=lambda item: item[0])
    return object_id, mask, float(overlap)


def materialize_window_frames(
    candidate: Mapping[str, Any],
    catalog: Mapping[tuple[str, int], Path],
    output_dir: Path,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cam = str(candidate["cam"])
    for index, frame_id in enumerate(candidate["frame_ids"]):
        source = catalog[(cam, int(frame_id))].resolve()
        destination = output_dir / f"{index:06d}.jpg"
        if source.suffix.lower() in {".jpg", ".jpeg"}:
            try:
                destination.symlink_to(source)
            except OSError:
                shutil.copy2(source, destination)
        else:
            Image.open(source).convert("RGB").save(destination, quality=96)


def longest_true_run(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def track_candidate_once(
    predictor: Any,
    candidate: Mapping[str, Any],
    catalog: Mapping[tuple[str, int], Path],
    track_root: Path,
    seed_frame_id: int,
    seed_box: Sequence[float],
) -> dict[str, Any]:
    window_id = str(candidate["window_id"])
    frame_ids = [int(value) for value in candidate["frame_ids"]]
    seed_index = frame_ids.index(seed_frame_id)
    frame_dir = track_root / window_id / "frames"
    materialize_window_frames(candidate, catalog, frame_dir)

    session = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(frame_dir),
            "offload_video_to_cpu": True,
        }
    )
    session_id = session["session_id"]
    masks_by_index: dict[int, np.ndarray] = {}
    try:
        prompted = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": seed_index,
                "bounding_boxes": [box_xyxy_to_xywh(seed_box)],
                "bounding_box_labels": [1],
            }
        )
        object_id, seed_mask, prompt_mask_iou = prompt_aligned_output_mask(
            prompted["outputs"], seed_box
        )
        if seed_mask is not None:
            masks_by_index[seed_index] = seed_mask
        if object_id is not None:
            for response in predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "both",
                }
            ):
                object_id, mask = output_mask_for_object(
                    response["outputs"], object_id
                )
                if mask is not None:
                    masks_by_index[int(response["frame_index"])] = mask
    finally:
        predictor.handle_request(
            {"type": "close_session", "session_id": session_id}
        )

    records: list[dict[str, Any]] = []
    masks_by_frame: dict[int, np.ndarray] = {}
    for index, frame_id in enumerate(frame_ids):
        mask = masks_by_index.get(index)
        if mask is None:
            records.append(
                {
                    "frame_id": frame_id,
                    "target_present": False,
                    "mask_area_ratio": 0.0,
                    "mask_bbox_xyxy_normalized": None,
                }
            )
            continue
        area_ratio = float(mask.mean())
        present = 0.0001 <= area_ratio <= 0.75
        box = mask_bbox(mask) if present else None
        if present:
            masks_by_frame[frame_id] = mask
        records.append(
            {
                "frame_id": frame_id,
                "target_present": present,
                "mask_area_ratio": area_ratio if present else 0.0,
                "mask_bbox_xyxy_normalized": box,
            }
        )

    present_values = [bool(item["target_present"]) for item in records]
    present_records = [item for item in records if item["target_present"]]
    areas = [float(item["mask_area_ratio"]) for item in present_records]
    verified_boxes = {
        int(item["frame_id"]): item["bbox_xyxy_normalized"]
        for item in candidate.get("verified_seed_frames", [])
    }
    records_by_frame = {int(item["frame_id"]): item for item in records}
    anchor_ious = []
    for frame_id, verified_box in verified_boxes.items():
        tracked = records_by_frame.get(frame_id)
        if not tracked or not tracked["target_present"]:
            anchor_ious.append(0.0)
            continue
        anchor_ious.append(
            box_iou(tracked["mask_bbox_xyxy_normalized"], verified_box)
        )
    anchor_match_count = sum(value >= 0.10 for value in anchor_ious)
    coverage = len(present_records) / max(1, len(records))
    continuity = longest_true_run(present_values) / max(1, len(records))
    mean_area = float(np.mean(areas)) if areas else 0.0
    area_stability = (
        max(0.0, 1.0 - float(np.std(areas)) / max(1e-9, mean_area))
        if len(areas) >= 2
        else 0.0
    )
    anchor_consistency = float(np.mean(anchor_ious)) if anchor_ious else 0.0
    centers = [
        (
            (item["mask_bbox_xyxy_normalized"][0]
             + item["mask_bbox_xyxy_normalized"][2]) / 2.0,
            (item["mask_bbox_xyxy_normalized"][1]
             + item["mask_bbox_xyxy_normalized"][3]) / 2.0,
        )
        for item in present_records
    ]
    center_jumps = [
        math.dist(centers[index - 1], centers[index])
        for index in range(1, len(centers))
    ]
    motion_stability = (
        max(0.0, 1.0 - float(np.mean(center_jumps)) / 0.25)
        if center_jumps
        else 0.0
    )
    raw_evidence = sum(areas)
    return {
        "rerank_schema_version": RERANK_SCHEMA_VERSION,
        "window_id": window_id,
        "cam": candidate["cam"],
        "start_frame": candidate["start_frame"],
        "end_frame": candidate["end_frame"],
        "requested_video_ratio": candidate["requested_video_ratio"],
        "actual_sampled_frame_ratio": candidate.get("actual_sampled_frame_ratio"),
        "actual_frame_span_ratio": candidate.get("actual_frame_span_ratio"),
        "seed_frame_id": seed_frame_id,
        "seed_bbox_xyxy_normalized": seed_box,
        "tracked_object_id": object_id,
        "frame_results": records,
        "mask_track_metrics": {
            "present_frame_count": len(present_records),
            "tested_frame_count": len(records),
            "coverage_fraction": coverage,
            "longest_continuous_fraction": continuity,
            "mean_mask_area_ratio": mean_area,
            "total_mask_area_evidence": raw_evidence,
            "mask_area_stability": area_stability,
            "verified_anchor_iou": anchor_consistency,
            "verified_anchor_count": len(verified_boxes),
            "verified_anchor_match_count": anchor_match_count,
            "prompt_mask_iou": prompt_mask_iou,
            "bbox_motion_stability": motion_stability,
        },
        "_masks_by_frame": masks_by_frame,
    }


def track_candidate(
    predictor: Any,
    candidate: Mapping[str, Any],
    catalog: Mapping[tuple[str, int], Path],
    track_root: Path,
) -> dict[str, Any]:
    primary = {
        "frame_id": int(candidate["seed_frame_id"]),
        "bbox_xyxy_normalized": candidate["seed_bbox_xyxy_normalized"],
    }
    seeds = [primary, *candidate.get("verified_seed_frames", [])]
    unique_seeds = []
    seen = set()
    for seed in seeds:
        key = (
            int(seed["frame_id"]),
            tuple(float(value) for value in seed["bbox_xyxy_normalized"]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_seeds.append(seed)
        if len(unique_seeds) >= 3:
            break

    trials = []
    for seed in unique_seeds:
        trial = track_candidate_once(
            predictor,
            candidate,
            catalog,
            track_root,
            int(seed["frame_id"]),
            seed["bbox_xyxy_normalized"],
        )
        trials.append(trial)
        metrics = trial["mask_track_metrics"]
        if (
            int(metrics["verified_anchor_match_count"]) >= 2
            and float(metrics["prompt_mask_iou"]) >= 0.10
            and int(metrics["present_frame_count"]) >= 2
        ):
            break

    winner = max(
        trials,
        key=lambda item: (
            int(item["mask_track_metrics"]["verified_anchor_match_count"]),
            float(item["mask_track_metrics"]["verified_anchor_iou"]),
            float(item["mask_track_metrics"]["prompt_mask_iou"]),
            float(item["mask_track_metrics"]["longest_continuous_fraction"]),
            float(item["mask_track_metrics"]["coverage_fraction"]),
        ),
    )
    winner["seed_trials"] = [
        {
            "seed_frame_id": trial["seed_frame_id"],
            "seed_bbox_xyxy_normalized": trial["seed_bbox_xyxy_normalized"],
            "metrics": trial["mask_track_metrics"],
            "selected_for_window": trial is winner,
        }
        for trial in trials
    ]
    return winner


def score_tracks(tracks: list[dict[str, Any]]) -> None:
    maximum_evidence = max(
        (
            float(item["mask_track_metrics"]["total_mask_area_evidence"])
            for item in tracks
        ),
        default=0.0,
    )
    maximum_mean_area = max(
        (
            float(item["mask_track_metrics"]["mean_mask_area_ratio"])
            for item in tracks
        ),
        default=0.0,
    )
    for item in tracks:
        metrics = item["mask_track_metrics"]
        captured = float(metrics["total_mask_area_evidence"]) / max(
            1e-9, maximum_evidence
        )
        scale = float(metrics["mean_mask_area_ratio"]) / max(
            1e-9, maximum_mean_area
        )
        metrics["normalized_captured_mask_evidence"] = captured
        metrics["normalized_target_scale"] = scale
        metrics["sam3_selection_score"] = (
            0.25 * captured
            + 0.20 * float(metrics["coverage_fraction"])
            + 0.20 * float(metrics["longest_continuous_fraction"])
            + 0.15 * scale
            + 0.10 * float(metrics["verified_anchor_iou"])
            + 0.05 * float(metrics["mask_area_stability"])
            + 0.05 * float(metrics["bbox_motion_stability"])
        )


def save_candidate_diagnostic_masks(
    case_dir: Path, candidate: dict[str, Any]
) -> None:
    masks = candidate.get("_masks_by_frame", {})
    if not masks:
        return
    window_id = str(candidate["window_id"])
    output_dir = (
        case_dir / "analysis_outputs" / "sam3_candidate_masks" / window_id
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    present_ids = sorted(int(frame_id) for frame_id in masks)
    display_ids = set(temporal.representative_ids(present_ids, 5))
    if candidate.get("seed_frame_id") is not None:
        display_ids.add(int(candidate["seed_frame_id"]))
    mask_paths = {}
    for frame_id in display_ids:
        mask = masks.get(frame_id)
        if mask is None:
            continue
        path = output_dir / f"frame_{frame_id:06d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        mask_paths[frame_id] = str(path.relative_to(case_dir))
    for item in candidate["frame_results"]:
        path = mask_paths.get(int(item["frame_id"]))
        if path:
            item["mask_path"] = path


def save_selected_masks(
    case_dir: Path, selected: dict[str, Any]
) -> list[dict[str, Any]]:
    output_dir = case_dir / "analysis_outputs" / "sam3_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = [int(item["frame_id"]) for item in selected["frame_results"]]
    display_ids = set(temporal.representative_ids(frame_ids, 5))
    mask_paths: dict[int, str] = {}
    for frame_id, mask in selected.pop("_masks_by_frame").items():
        if frame_id not in display_ids:
            continue
        path = output_dir / f"frame_{frame_id:06d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        mask_paths[frame_id] = str(path.relative_to(case_dir))
    for item in selected["frame_results"]:
        if int(item["frame_id"]) in mask_paths:
            item["mask_path"] = mask_paths[int(item["frame_id"])]
    return selected["frame_results"]


def update_result(
    case_dir: Path,
    result: dict[str, Any],
    tracks: list[dict[str, Any]],
    minimum_score_margin: float,
) -> dict[str, Any]:
    result.pop("sam3_frame_evidence", None)
    score_tracks(tracks)
    tracks.sort(
        key=lambda item: float(
            item["mask_track_metrics"]["sam3_selection_score"]
        ),
        reverse=True,
    )
    for track in tracks:
        save_candidate_diagnostic_masks(case_dir, track)
    serializable = []
    for track in tracks:
        copy = dict(track)
        copy.pop("_masks_by_frame", None)
        serializable.append(copy)
    result["sam3_rerank"] = {
        "schema_version": RERANK_SCHEMA_VERSION,
        "candidate_count": len(tracks),
        "candidates": serializable,
        "selected_window_id": None,
        "score_margin": None,
    }
    if not tracks:
        result["status"] = "uncertain"
        result["best_cam"] = None
        result["best_segment"] = None
        result["global_challenger_comparison"] = None
        result["uncertainty"] = "SAM3 received no valid dense-window seed candidates."
        result["schema_version"] = RESULT_SCHEMA_VERSION
        result["pipeline_status"] = "complete"
        result["confidence"] = 0.0
        return result

    winner = tracks[0]
    runner_up_score = (
        float(tracks[1]["mask_track_metrics"]["sam3_selection_score"])
        if len(tracks) > 1
        else 0.0
    )
    winner_score = float(
        winner["mask_track_metrics"]["sam3_selection_score"]
    )
    margin = winner_score - runner_up_score
    metrics = winner["mask_track_metrics"]
    accepted = bool(
        int(metrics["present_frame_count"]) >= 2
        and float(metrics["prompt_mask_iou"]) >= 0.10
        and float(metrics["verified_anchor_iou"]) >= 0.10
        and int(metrics["verified_anchor_count"]) >= 2
        and int(metrics["verified_anchor_match_count"]) >= 2
        and margin >= minimum_score_margin
    )
    failed_gates = []
    if int(metrics["present_frame_count"]) < 2:
        failed_gates.append("fewer than two tracked mask frames")
    if float(metrics["prompt_mask_iou"]) < 0.10:
        failed_gates.append("SAM3 prompt mask does not overlap its Qwen seed box")
    if int(metrics["verified_anchor_count"]) < 2:
        failed_gates.append("fewer than two independent Qwen anchors")
    if int(metrics["verified_anchor_match_count"]) < 2:
        failed_gates.append("fewer than two SAM3/Qwen anchor matches")
    if float(metrics["verified_anchor_iou"]) < 0.10:
        failed_gates.append("mean anchor IoU below 0.10")
    if margin < minimum_score_margin:
        failed_gates.append(
            f"winner margin {margin:.4f} below {minimum_score_margin:.4f}"
        )
    result["sam3_rerank"]["acceptance_gate"] = {
        "passed": accepted,
        "failed_conditions": failed_gates,
    }
    result["sam3_rerank"]["score_margin"] = margin
    result["global_challenger_comparison"] = {
        "passed": margin >= minimum_score_margin,
        "selected_score": winner_score,
        "challenger_window_id": tracks[1]["window_id"] if len(tracks) > 1 else None,
        "challenger_score": runner_up_score if len(tracks) > 1 else None,
        "score_margin": margin,
    }
    result["schema_version"] = RESULT_SCHEMA_VERSION
    result["pipeline_status"] = "complete"
    result["confidence"] = winner_score if accepted else 0.0
    if not accepted:
        result["status"] = "uncertain"
        result["best_cam"] = None
        result["best_segment"] = None
        result["uncertainty"] = (
            "SAM3 mask-track acceptance failed: " + "; ".join(failed_gates)
        )
        return result

    frame_results = save_selected_masks(case_dir, winner)
    result["sam3_rerank"]["selected_window_id"] = winner["window_id"]
    result["sam3_frame_evidence"] = frame_results
    result["status"] = "success"
    result["best_cam"] = winner["cam"]
    result["uncertainty"] = ""
    result["best_segment"] = {
        "window_id": winner["window_id"],
        "cam": winner["cam"],
        "start_frame": winner["start_frame"],
        "end_frame": winner["end_frame"],
        "requested_video_ratio": winner["requested_video_ratio"],
        "actual_sampled_frame_ratio": winner.get("actual_sampled_frame_ratio"),
        "actual_frame_span_ratio": winner.get("actual_frame_span_ratio"),
        "sam3_mask_track_metrics": winner["mask_track_metrics"],
        "reason_selected": (
            "Highest global dense-window score from SAM3 mask area, coverage, "
            "continuity, scale, stability, and verified-anchor consistency."
        ),
    }
    return result


def process_case(
    predictor: Any,
    case_dir: Path,
    work_root: Path,
    max_candidates: int,
    minimum_score_margin: float,
) -> dict[str, Any]:
    print(f"\n===== SAM3 {case_dir.name} =====", flush=True)
    work_case = work_root / case_dir.name
    result_path = case_dir / "temporal_analysis_result.json"
    result = temporal.read_json(result_path)
    index = temporal.read_json(case_dir / "temporal_window_index.json")
    catalog = temporal.make_frame_catalog(case_dir, index, work_case)
    candidates = [
        item
        for item in result.get("sam3_rerank_candidates", [])[:max_candidates]
        if item.get("seed_frame_id") is not None
        and item.get("seed_bbox_xyxy_normalized")
    ]
    tracks = []
    for position, candidate in enumerate(candidates, start=1):
        print(
            f"[track {position}/{len(candidates)}] {candidate['window_id']}",
            flush=True,
        )
        tracks.append(
            track_candidate(
                predictor,
                candidate,
                catalog,
                work_case / "sam3_tracks",
            )
        )
    result = update_result(case_dir, result, tracks, minimum_score_margin)
    temporal.write_json(result_path, result)
    evidence = temporal.read_json(work_case / "qwen_frame_evidence.json")
    temporal.render_result(case_dir, work_case, result, evidence, catalog, index)
    best = result.get("best_segment")
    return {
        "case_id": case_dir.name,
        "status": result["status"],
        "window_id": best["window_id"] if best else None,
        "start_frame": best["start_frame"] if best else None,
        "end_frame": best["end_frame"] if best else None,
        "uncertainty": result.get("uncertainty", ""),
        "pipeline_status": result.get("pipeline_status"),
    }


def main() -> None:
    args = parse_args()
    if args.max_candidates <= 0:
        raise SystemExit("--max-candidates must be positive")
    if args.minimum_score_margin < 0.0:
        raise SystemExit("--minimum-score-margin cannot be negative")
    root = args.assets_root.resolve()
    work_root = args.work_dir.resolve()
    cases = temporal.case_directories(root, args.case)
    if not cases:
        raise SystemExit("No matching temporal cases found.")

    if args.render_only:
        for case_dir in cases:
            work_case = work_root / case_dir.name
            result = temporal.read_json(case_dir / "temporal_analysis_result.json")
            index = temporal.read_json(case_dir / "temporal_window_index.json")
            catalog = temporal.make_frame_catalog(case_dir, index, work_case)
            evidence = temporal.read_json(work_case / "qwen_frame_evidence.json")
            temporal.render_result(case_dir, work_case, result, evidence, catalog, index)
        return

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))

    import torch
    from sam3.model_builder import build_sam3_video_predictor

    checkpoint = str(args.checkpoint.resolve()) if args.checkpoint else None
    predictor = build_sam3_video_predictor(
        checkpoint_path=checkpoint,
        gpus_to_use=[torch.cuda.current_device()],
        compile=False,
    )
    summaries = []
    try:
        for case_dir in cases:
            try:
                summaries.append(
                    process_case(
                        predictor,
                        case_dir,
                        work_root,
                        args.max_candidates,
                        args.minimum_score_margin,
                    )
                )
            except Exception as error:
                traceback.print_exc()
                summaries.append(
                    {
                        "case_id": case_dir.name,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    finally:
        predictor.shutdown()
    summary_path = (
        args.summary_path.resolve()
        if args.summary_path
        else root / "batch_sam3_temporal_summary.json"
    )
    temporal.write_json(summary_path, summaries)
    if args.summary_path is None:
        temporal.write_json(root / "batch_temporal_analysis_summary.json", summaries)


if __name__ == "__main__":
    main()
