#!/usr/bin/env python3
"""Render final selected-window frames with independent SAM3 masks.

Qwen owns temporal selection. SAM3 is deliberately limited to final visual
segmentation and never changes the selected window or temporal status.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

import qwen_temporal_runner as temporal


FINAL_SEGMENTATION_SCHEMA_VERSION = 2
RESULT_SCHEMA_VERSION = 14
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def box_xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    return [
        float(box[0]),
        float(box[1]),
        float(box[2]) - float(box[0]),
        float(box[3]) - float(box[1]),
    ]


def expand_box(box: Sequence[float], factor: float = 1.35) -> list[float]:
    center_x = (float(box[0]) + float(box[2])) / 2.0
    center_y = (float(box[1]) + float(box[3])) / 2.0
    half_width = (float(box[2]) - float(box[0])) * factor / 2.0
    half_height = (float(box[3]) - float(box[1])) * factor / 2.0
    return [
        max(0.0, center_x - half_width),
        max(0.0, center_y - half_height),
        min(1.0, center_x + half_width),
        min(1.0, center_y + half_height),
    ]


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


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = temporal.normalized_box_area(left)
    right_area = temporal.normalized_box_area(right)
    return intersection / max(1e-9, left_area + right_area - intersection)


def reference_coverage(candidate: Sequence[float], reference: Sequence[float]) -> float:
    intersection = max(
        0.0, min(candidate[2], reference[2]) - max(candidate[0], reference[0])
    ) * max(
        0.0, min(candidate[3], reference[3]) - max(candidate[1], reference[1])
    )
    return intersection / max(1e-9, temporal.normalized_box_area(reference))


def output_arrays(
    outputs: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = []
    for key in ("out_obj_ids", "out_binary_masks", "out_probs"):
        value = outputs.get(key, [])
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        values.append(np.asarray(value))
    object_ids = values[0].reshape(-1)
    masks = values[1]
    probabilities = values[2].reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if len(probabilities) < len(object_ids):
        probabilities = np.pad(
            probabilities,
            (0, len(object_ids) - len(probabilities)),
            constant_values=0.0,
        )
    return object_ids, masks, probabilities


def choose_final_mask(
    outputs: Mapping[str, Any],
    expected_box: Sequence[float] | None,
    require_spatial_match: bool = False,
) -> dict[str, Any] | None:
    """Choose a mask that agrees with Qwen's verified instance geometry."""
    object_ids, masks, probabilities = output_arrays(outputs)
    choices = []
    for index, object_id in enumerate(object_ids.astype(int)):
        if index >= len(masks):
            continue
        mask = masks[index].astype(bool)
        area = float(mask.mean())
        box = mask_bbox(mask)
        if box is None or not 0.0001 <= area <= 0.60:
            continue
        probability = float(probabilities[index]) if index < len(probabilities) else 0.0
        overlap = box_iou(box, expected_box) if expected_box else 0.0
        coverage = reference_coverage(box, expected_box) if expected_box else 0.0
        spatial_match = bool(
            expected_box
            and (coverage >= 0.15 or overlap >= 0.10)
        )
        if require_spatial_match and not spatial_match:
            continue
        # Qwen owns instance identity and location. SAM3 probability only
        # breaks ties between masks that already agree spatially.
        score = 0.55 * coverage + 0.35 * overlap + 0.10 * probability
        choices.append(
            {
                "object_id": int(object_id),
                "mask": mask,
                "mask_bbox_xyxy_normalized": box,
                "mask_area_ratio": area,
                "semantic_probability": probability,
                "expected_box_iou": overlap,
                "expected_box_coverage": coverage,
                "qwen_spatial_match": spatial_match,
                "selection_score": score,
            }
        )
    if not choices:
        return None
    return max(choices, key=lambda item: item["selection_score"])


def materialize_single_frame(source: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "000000.jpg"
    if source.suffix.lower() in {".jpg", ".jpeg"}:
        try:
            destination.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, destination)
    else:
        Image.open(source).convert("RGB").save(destination, quality=97)


def run_single_frame_prompt(
    predictor: Any,
    frame_path: Path,
    frame_dir: Path,
    text_prompt: str | None,
    box_prompt: Sequence[float] | None,
) -> Mapping[str, Any]:
    materialize_single_frame(frame_path, frame_dir)
    session = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(frame_dir),
            "offload_video_to_cpu": True,
        }
    )
    session_id = session["session_id"]
    try:
        request: dict[str, Any] = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
        }
        if text_prompt:
            request["text"] = text_prompt
        if box_prompt:
            request["bounding_boxes"] = [box_xyxy_to_xywh(box_prompt)]
            request["bounding_box_labels"] = [1]
        return predictor.handle_request(request)["outputs"]
    finally:
        predictor.handle_request(
            {"type": "close_session", "session_id": session_id}
        )


def segment_final_frame(
    predictor: Any,
    frame_path: Path,
    frame_dir: Path,
    object_identity: str,
    expected_box: Sequence[float] | None,
) -> tuple[dict[str, Any] | None, str]:
    if expected_box is None:
        return None, "missing_qwen_verified_box"
    box_outputs = run_single_frame_prompt(
        predictor,
        frame_path,
        frame_dir / "qwen_box",
        None,
        expand_box(expected_box, factor=1.20),
    )
    box_selected = choose_final_mask(
        box_outputs,
        expected_box,
        require_spatial_match=True,
    )
    if box_selected is None:
        return None, "qwen_verified_box_no_spatial_mask"
    return box_selected, "qwen_verified_box"


def selected_window(
    result: Mapping[str, Any], temporal_index: Mapping[str, Any]
) -> dict[str, Any] | None:
    best = result.get("qwen_temporal_selection") or result.get("best_segment")
    if best:
        return dict(best)
    candidates = result.get("sam3_rerank_candidates", [])
    if candidates:
        candidate = candidates[0]
        return {
            key: candidate.get(key)
            for key in (
                "window_id",
                "cam",
                "start_frame",
                "end_frame",
                "requested_video_ratio",
                "actual_sampled_frame_ratio",
                "actual_frame_span_ratio",
            )
        }
    return None


def window_record(
    selection: Mapping[str, Any], temporal_index: Mapping[str, Any]
) -> Mapping[str, Any]:
    windows = {
        str(item["window_id"]): item
        for item in [
            *temporal.all_windows(temporal_index),
            *temporal.dense_sliding_windows(temporal_index),
        ]
    }
    window_id = str(selection["window_id"])
    if window_id in windows:
        return windows[window_id]
    return temporal.materialize_selected_window(selection, temporal_index)


def verified_box_map(
    result: Mapping[str, Any], window_id: str
) -> dict[int, list[float]]:
    output = {}
    verification = result.get("window_verifications", {}).get(window_id, {})
    for item in verification.get("frame_results", []):
        box = temporal.normalized_box(item.get("bbox_xyxy_normalized"))
        if item.get("target_present") and box:
            output[int(item["frame_id"])] = box
    for candidate in result.get("sam3_rerank_candidates", []):
        if candidate.get("window_id") != window_id:
            continue
        for item in candidate.get("verified_seed_frames", []):
            box = temporal.normalized_box(item.get("bbox_xyxy_normalized"))
            if box:
                output[int(item["frame_id"])] = box
    return output


def nearest_box(
    boxes: Mapping[int, Sequence[float]], frame_id: int
) -> list[float] | None:
    if not boxes:
        return None
    nearest_id = min(boxes, key=lambda value: abs(int(value) - frame_id))
    return list(boxes[nearest_id])


def nearest_box_with_source(
    boxes: Mapping[int, Sequence[float]], frame_id: int
) -> tuple[list[float] | None, int | None]:
    if not boxes:
        return None, None
    nearest_id = min(boxes, key=lambda value: abs(int(value) - frame_id))
    return list(boxes[nearest_id]), int(nearest_id)


def save_mask(case_dir: Path, cam: str, frame_id: int, mask: np.ndarray) -> str:
    output_dir = case_dir / "analysis_outputs" / "final_sam3_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{cam}_frame_{frame_id:06d}.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return str(path.relative_to(case_dir))


def finalize_result(
    predictor: Any,
    case_dir: Path,
    work_case: Path,
    result: dict[str, Any],
    temporal_index: Mapping[str, Any],
    catalog: Mapping[tuple[str, int], Path],
) -> dict[str, Any]:
    selection = selected_window(result, temporal_index)
    result.pop("sam3_rerank", None)
    result.pop("sam3_frame_evidence", None)
    result.pop("sam3_rerank_candidates", None)
    if selection is None:
        result["final_segmentation"] = {
            "schema_version": FINAL_SEGMENTATION_SCHEMA_VERSION,
            "role": "visualization_only",
            "frame_results": [],
            "warning": "No Qwen-selected temporal window exists to segment.",
        }
        result["schema_version"] = RESULT_SCHEMA_VERSION
        result["pipeline_status"] = "complete"
        return result

    window = window_record(selection, temporal_index)
    cam = str(window["cam"])
    display_ids = temporal.representative_ids(window["frame_ids"], 5)
    boxes = verified_box_map(result, str(window["window_id"]))
    identity = str(
        result.get("source_identity", {}).get("object_identity")
        or result.get("target_object", "object")
    )
    mask_output_dir = case_dir / "analysis_outputs" / "final_sam3_masks"
    if mask_output_dir.exists():
        shutil.rmtree(mask_output_dir)
    frame_results = []
    for position, frame_id in enumerate(display_ids, start=1):
        print(
            f"[final mask {position}/{len(display_ids)}] {cam} frame {frame_id}",
            flush=True,
        )
        expected, expected_source_frame = nearest_box_with_source(
            boxes, int(frame_id)
        )
        selected, method = segment_final_frame(
            predictor,
            catalog[(cam, int(frame_id))],
            work_case / "sam3_final_frames" / f"{cam}_{frame_id}",
            identity,
            expected,
        )
        if selected is None:
            frame_results.append(
                {
                    "cam": cam,
                    "frame_id": int(frame_id),
                    "target_present": False,
                    "method": method,
                    "qwen_expected_box_xyxy_normalized": expected,
                    "qwen_box_source_frame": expected_source_frame,
                    "mask_path": None,
                }
            )
            continue
        mask_path = save_mask(
            case_dir, cam, int(frame_id), selected.pop("mask")
        )
        frame_results.append(
            {
                "cam": cam,
                "frame_id": int(frame_id),
                "target_present": True,
                "method": method,
                "qwen_expected_box_xyxy_normalized": expected,
                "qwen_box_source_frame": expected_source_frame,
                "mask_path": mask_path,
                **selected,
            }
        )

    result["best_segment"] = selection
    result["best_cam"] = selection["cam"]
    result["status"] = "success"
    result["uncertainty"] = ""
    result["schema_version"] = RESULT_SCHEMA_VERSION
    result["pipeline_status"] = "complete"
    result["final_segmentation"] = {
        "schema_version": FINAL_SEGMENTATION_SCHEMA_VERSION,
        "role": "visualization_only",
        "spatial_policy": "qwen_verified_box_required",
        "sam3_changes_temporal_selection": False,
        "changes_temporal_selection": False,
        "object_identity": identity,
        "selected_window_id": selection["window_id"],
        "requested_frame_count": len(display_ids),
        "segmented_frame_count": sum(
            bool(item["target_present"]) for item in frame_results
        ),
        "frame_results": frame_results,
    }
    result["sam3_frame_evidence"] = frame_results
    return result


def process_case(
    predictor: Any,
    case_dir: Path,
    work_root: Path,
) -> dict[str, Any]:
    print(f"\n===== SAM3 final masks {case_dir.name} =====", flush=True)
    work_case = work_root / case_dir.name
    result_path = case_dir / "temporal_analysis_result.json"
    result = temporal.read_json(result_path)
    temporal_index = temporal.read_json(case_dir / "temporal_window_index.json")
    catalog = temporal.make_frame_catalog(case_dir, temporal_index, work_case)
    result = finalize_result(
        predictor, case_dir, work_case, result, temporal_index, catalog
    )
    temporal.write_json(result_path, result)
    evidence = temporal.read_json(work_case / "qwen_frame_evidence.json")
    temporal.render_result(
        case_dir, work_case, result, evidence, catalog, temporal_index
    )
    best = result.get("best_segment")
    return {
        "case_id": case_dir.name,
        "status": result["status"],
        "pipeline_status": result.get("pipeline_status"),
        "window_id": best["window_id"] if best else None,
        "start_frame": best["start_frame"] if best else None,
        "end_frame": best["end_frame"] if best else None,
        "segmented_frame_count": result.get("final_segmentation", {}).get(
            "segmented_frame_count", 0
        ),
    }


def main() -> None:
    args = parse_args()
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
        apply_temporal_disambiguation=False,
        compile=False,
    )
    summaries = []
    try:
        for case_dir in cases:
            try:
                summaries.append(process_case(predictor, case_dir, work_root))
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
        else root / "batch_sam3_final_segmentation_summary.json"
    )
    temporal.write_json(summary_path, summaries)
    if args.summary_path is None:
        temporal.write_json(root / "batch_temporal_analysis_summary.json", summaries)


if __name__ == "__main__":
    main()
