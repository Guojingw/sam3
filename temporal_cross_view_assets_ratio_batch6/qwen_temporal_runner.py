#!/usr/bin/env python3
"""Run mask-anchored cross-view temporal selection with Qwen3.5-4B.

Qwen identifies the masked source object and evaluates target frames. Window
selection is deterministic and enforces the temporal constraints in code.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import textwrap
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps


CASE_GLOB = "*__*_0"
PRESENT_THRESHOLD = 0.65
WINDOW_PRESENT_FRACTION = 0.75
MAX_INTERNAL_ABSENT_RUN = 2
MAX_OCCURRENCE_GAP = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--model",
        default="/scratch/users/ntu/gwang016/qwen35-4b/model",
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def clean_target_name(value: str) -> str:
    return re.sub(r"_0$", "", value.replace("_", " ")).strip()


def parse_model_json(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No valid JSON found in model output: {text[:500]}")


def normalized_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [max(0.0, min(1.0, float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def build_source_anchor(case_dir: Path, work_case: Path) -> dict[str, Any]:
    metadata = read_json(case_dir / "metadata.json")
    source = metadata["source_best"]
    frame_path = case_dir / source["source_frame"]
    mask_path = case_dir / source["source_mask"]
    frame = Image.open(frame_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != frame.size:
        mask = mask.resize(frame.size, Image.Resampling.NEAREST)
    mask = mask.point(lambda pixel: 255 if pixel >= 128 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError(f"Empty source mask: {mask_path}")

    left, top, right, bottom = bbox
    padding = max(12, round(max(right - left, bottom - top) * 0.22))
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(frame.width, right + padding),
        min(frame.height, bottom + padding),
    )
    context = frame.crop(crop_box)
    crop_mask = mask.crop(crop_box)

    contour = context.copy()
    contour_draw = ImageDraw.Draw(contour)
    contour_bbox = crop_mask.getbbox()
    if contour_bbox:
        contour_draw.rectangle(contour_bbox, outline=(0, 255, 255), width=5)

    isolated = Image.new("RGB", context.size, (224, 224, 224))
    isolated.paste(context, mask=crop_mask)
    isolated = ImageOps.expand(isolated, border=8, fill=(0, 0, 0))

    work_case.mkdir(parents=True, exist_ok=True)
    context_path = work_case / "source_anchor_context_rgb.png"
    isolated_path = work_case / "source_anchor_isolated_rgb.png"
    contour.save(context_path)
    isolated.save(isolated_path)
    return {
        "metadata": metadata,
        "frame_path": frame_path,
        "mask_path": mask_path,
        "context_path": context_path,
        "isolated_path": isolated_path,
        "mask_bbox_xyxy": list(bbox),
        "crop_box_xyxy": list(crop_box),
    }


def all_windows(index: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for camera in index["cameras"].values():
        yield from camera["windows"]


def make_frame_catalog(
    case_dir: Path, index: Mapping[str, Any], work_case: Path
) -> dict[tuple[str, int], Path]:
    frame_dir = work_case / "target_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    catalog: dict[tuple[str, int], Path] = {}
    opened: dict[Path, Image.Image] = {}
    try:
        for window in all_windows(index):
            cam = str(window["cam"])
            sheet_path = case_dir / window["contact_sheet"]
            for cell in window["sheet_layout"]["cells"]:
                frame_id = int(cell["frame_id"])
                key = (cam, frame_id)
                if key in catalog:
                    continue
                output = frame_dir / f"{cam}_{frame_id:06d}.jpg"
                if not output.exists():
                    if sheet_path not in opened:
                        opened[sheet_path] = Image.open(sheet_path).convert("RGB")
                    crop = opened[sheet_path].crop(tuple(cell["image_xyxy"]))
                    crop.save(output, quality=96)
                catalog[key] = output
    finally:
        for image in opened.values():
            image.close()
    return catalog


class Qwen:
    def __init__(self, model_path: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        print("Loading processor...", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        print("Loading Qwen3.5-4B...", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def ask(self, images: list[Path], prompt: str) -> Any:
        content: list[dict[str, Any]] = []
        for image_path in images:
            content.append(
                {"type": "image", "image": Image.open(image_path).convert("RGB")}
            )
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated = generated[:, inputs["input_ids"].shape[1] :]
        output = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0]
        return parse_model_json(output)


def identify_source(qwen: Qwen, anchor: Mapping[str, Any]) -> dict[str, Any]:
    metadata = anchor["metadata"]
    source = metadata["source_best"]
    label = clean_target_name(metadata["target_object"])
    prompt = f"""
You are establishing the visual identity anchor for cross-view object tracking.

Image 1 is the complete first-person RGB source frame.
Image 2 is a context crop. Its cyan rectangle only marks the source-mask
bounding region; cyan is an annotation and is NOT an object color.
Image 3 isolates exactly the source-mask pixels while preserving their original
RGB colors. Gray/black pixels are artificial background.

The metadata label is "{label}". Use it only as a semantic hint. The mask pixels
in Image 3 are the authoritative identity anchor. Never infer color or material
from an overlay/annotation color. Describe structure, true color, material,
shape, parts, and cues that distinguish it from nearby similar objects.

Return JSON only:
{{
  "object_identity": "short name",
  "visual_signature": "specific appearance grounded in masked RGB pixels",
  "distinguishing_cues": ["cue 1", "cue 2"],
  "common_confusions_to_exclude": ["confusion 1"],
  "confidence": 0.0
}}
"""
    result = qwen.ask(
        [
            anchor["frame_path"],
            anchor["context_path"],
            anchor["isolated_path"],
        ],
        prompt,
    )
    if not isinstance(result, dict):
        raise ValueError("Source identity response is not an object")
    result["metadata_label"] = label
    result["source_mask"] = str(source["source_mask"])
    result["source_mask_overlay"] = str(source["source_mask_overlay"])
    result["source_frame_id"] = int(source["frame_id"])
    return result


def assess_frame(
    qwen: Qwen,
    anchor: Mapping[str, Any],
    identity: Mapping[str, Any],
    frame_path: Path,
    cam: str,
    frame_id: int,
) -> dict[str, Any]:
    prompt = f"""
Perform strict cross-view identity verification.

Image 1 contains ONLY the first-person source-mask pixels in their true RGB
colors. Gray/black is artificial background.
Image 2 is one third-person frame: camera={cam}, frame_id={frame_id}.

Target identity description:
{json.dumps(identity, ensure_ascii=False)}

Decide whether the SAME target object is genuinely visible in Image 2. Scene
activity, a person using another object, shared color, and generic object class
are not sufficient. Use structure and distinguishing cues. If ambiguous,
partially hidden beyond reliable recognition, or absent, set target_present to
false. Do not claim a box unless target_present is true.

Return JSON only:
{{
  "frame_id": {frame_id},
  "target_present": false,
  "bbox_xyxy_normalized": null,
  "visibility": 0.0,
  "occlusion": "none|low|medium|high",
  "identity_confidence": 0.0,
  "visual_evidence": "brief frame-specific evidence"
}}
"""
    raw = qwen.ask([anchor["isolated_path"], frame_path], prompt)
    if not isinstance(raw, dict):
        raise ValueError("Frame response is not an object")
    confidence = max(
        0.0, min(1.0, float(raw.get("identity_confidence", 0.0)))
    )
    present = bool(raw.get("target_present")) and confidence >= PRESENT_THRESHOLD
    box = normalized_box(raw.get("bbox_xyxy_normalized")) if present else None
    if box is None:
        present = False
    return {
        "cam": cam,
        "frame_id": frame_id,
        "target_present": present,
        "bbox_xyxy_normalized": box,
        "visibility": max(
            0.0, min(1.0, float(raw.get("visibility", 0.0)))
        ),
        "occlusion": str(raw.get("occlusion", "high")),
        "identity_confidence": confidence,
        "visual_evidence": str(raw.get("visual_evidence", "")),
    }


def absent_run_max(items: list[dict[str, Any]]) -> int:
    maximum = current = 0
    for item in items:
        if item["target_present"]:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def occurrence_runs(
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    gap = 0
    for item in evidence:
        if item["target_present"]:
            if not current:
                current = [item]
            else:
                current.append(item)
            gap = 0
        elif current:
            gap += 1
            if gap <= MAX_OCCURRENCE_GAP:
                current.append(item)
            else:
                trimmed = current[:-MAX_OCCURRENCE_GAP]
                if trimmed:
                    runs.append(trimmed)
                current = []
                gap = 0
    if current:
        while current and not current[-1]["target_present"]:
            current.pop()
        if current:
            runs.append(current)
    output = []
    for run in runs:
        positives = [item for item in run if item["target_present"]]
        if len(positives) < 2:
            continue
        output.append(
            {
                "cam": positives[0]["cam"],
                "first_confirmed_frame": positives[0]["frame_id"],
                "last_confirmed_frame": positives[-1]["frame_id"],
                "sampled_frame_count": len(run),
                "confirmed_frame_count": len(positives),
                "confirmed_fraction": len(positives) / len(run),
            }
        )
    return output


def median_box(items: list[dict[str, Any]]) -> list[float] | None:
    boxes = [
        item["bbox_xyxy_normalized"]
        for item in items
        if item["bbox_xyxy_normalized"]
    ]
    if not boxes:
        return None
    return [round(statistics.median(values), 4) for values in zip(*boxes)]


def select_representatives(
    frame_items: list[dict[str, Any]], count: int = 5
) -> list[dict[str, Any]]:
    present = [item for item in frame_items if item["target_present"]]
    if not present:
        return []
    selected = []
    for index in range(min(count, len(present))):
        position = round(index * (len(present) - 1) / max(1, count - 1))
        item = present[position]
        if item not in selected:
            selected.append(item)
    return [
        {
            "frame_id": item["frame_id"],
            "target_region_xyxy": item["bbox_xyxy_normalized"],
            "visibility": item["visibility"],
            "occlusion": item["occlusion"],
            "identity_confidence": item["identity_confidence"],
        }
        for item in selected
    ]


def score_window(
    window: Mapping[str, Any], frame_items: list[dict[str, Any]]
) -> dict[str, Any]:
    present = [item for item in frame_items if item["target_present"]]
    fraction = len(present) / max(1, len(frame_items))
    visibility = statistics.mean(
        [item["visibility"] for item in present] or [0.0]
    )
    identity = statistics.mean(
        [item["identity_confidence"] for item in present] or [0.0]
    )
    longest_absence = absent_run_max(frame_items)
    continuity = max(0.0, 1.0 - longest_absence / 4.0)
    endpoint_ok = bool(
        frame_items
        and any(item["target_present"] for item in frame_items[:2])
        and any(item["target_present"] for item in frame_items[-2:])
    )
    overall = (
        0.42 * fraction
        + 0.23 * visibility
        + 0.25 * identity
        + 0.10 * continuity
    )
    valid = (
        bool(window.get("continuity_ok"))
        and 0.20 <= float(window["requested_video_ratio"]) <= 0.30
        and fraction >= WINDOW_PRESENT_FRACTION
        and longest_absence <= MAX_INTERNAL_ABSENT_RUN
        and endpoint_ok
    )
    return {
        "valid": valid,
        "present_fraction": fraction,
        "longest_absent_run": longest_absence,
        "endpoint_evidence_ok": endpoint_ok,
        "mean_visibility": visibility,
        "mean_identity_confidence": identity,
        "overall": overall,
    }


def analyze_windows(
    metadata: Mapping[str, Any],
    temporal_index: Mapping[str, Any],
    identity: Mapping[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_map = {
        (item["cam"], item["frame_id"]): item for item in evidence
    }
    runs = occurrence_runs(evidence)
    candidates = []
    for window in all_windows(temporal_index):
        items = [
            evidence_map[(window["cam"], int(frame_id))]
            for frame_id in window["frame_ids"]
        ]
        score = score_window(window, items)
        inside_run = any(
            run["cam"] == window["cam"]
            and int(window["start_frame"]) >= run["first_confirmed_frame"]
            and int(window["end_frame"]) <= run["last_confirmed_frame"]
            for run in runs
        )
        score["inside_confirmed_occurrence"] = inside_run
        score["valid"] = score["valid"] and inside_run
        candidates.append((window, items, score))

    candidates.sort(key=lambda entry: entry[2]["overall"], reverse=True)
    valid = [entry for entry in candidates if entry[2]["valid"]]
    selected = valid[0] if valid else None

    rejected = []
    for window, _, score in candidates:
        if selected and window["window_id"] == selected[0]["window_id"]:
            continue
        reason = []
        if not score["inside_confirmed_occurrence"]:
            reason.append("window is not fully inside a confirmed occurrence")
        if score["present_fraction"] < WINDOW_PRESENT_FRACTION:
            reason.append(
                f"only {score['present_fraction']:.1%} sampled frames confirmed"
            )
        if score["longest_absent_run"] > MAX_INTERNAL_ABSENT_RUN:
            reason.append("absence gap is too long")
        if not score["endpoint_evidence_ok"]:
            reason.append("start/end evidence is insufficient")
        rejected.append(
            {
                "window_id": window["window_id"],
                "cam": window["cam"],
                "start_frame": window["start_frame"],
                "end_frame": window["end_frame"],
                "reason": "; ".join(reason) or "lower deterministic score",
                "scores": score,
            }
        )
        if len(rejected) == 3:
            break

    source = metadata["source_best"]
    result: dict[str, Any] = {
        "schema_version": 3,
        "case_id": metadata.get("case_id", ""),
        "target_object": metadata["target_object"],
        "source_best_view": source["view_name"],
        "source_best_frame": source["frame_id"],
        "source_best_mask": source["source_mask"],
        "source_best_mask_overlay": source["source_mask_overlay"],
        "source_mask_area_ratio": source["mask_area_ratio"],
        "source_identity": identity,
        "occurrence_spans": runs,
        "selection_constraints": {
            "window_ratio_min": 0.20,
            "window_ratio_max": 0.30,
            "minimum_confirmed_sample_fraction": WINDOW_PRESENT_FRACTION,
            "maximum_internal_absent_samples": MAX_INTERNAL_ABSENT_RUN,
            "window_must_be_inside_confirmed_occurrence": True,
        },
        "status": "success" if selected else "uncertain",
        "best_cam": selected[0]["cam"] if selected else None,
        "best_segment": None,
        "alternative_segments": [],
        "rejected_segments": rejected,
        "uncertainty": "",
        "confidence": 0.0,
    }
    if not selected:
        result["uncertainty"] = (
            "No generated 20%-30% continuous window is fully contained in a "
            "confirmed target occurrence while satisfying whole-window "
            "visibility and continuity requirements."
        )
        result["confidence"] = round(
            max((entry[2]["overall"] for entry in candidates), default=0.0), 4
        )
        return result

    window, items, score = selected
    stable = median_box(items)
    result["confidence"] = round(score["overall"], 4)
    result["best_segment"] = {
        "window_id": window["window_id"],
        "cam": window["cam"],
        "start_frame": window["start_frame"],
        "end_frame": window["end_frame"],
        "requested_video_ratio": window["requested_video_ratio"],
        "actual_sampled_frame_ratio": window["actual_sampled_frame_ratio"],
        "actual_frame_span_ratio": window["actual_frame_span_ratio"],
        "representative_frames": select_representatives(items),
        "target_region_summary": {
            "coordinate_system": "normalized_xyxy_per_frame",
            "stable_region_xyxy": stable,
            "region_explanation": (
                "Median of Qwen-confirmed per-frame boxes inside the selected "
                "window; boxes are localization evidence, not masks."
            ),
        },
        "scores": {
            "visibility_consistency": round(score["present_fraction"], 4),
            "identity_confidence": round(
                score["mean_identity_confidence"], 4
            ),
            "temporal_stability": round(
                1.0 - score["longest_absent_run"] / 4.0, 4
            ),
            "overall": round(score["overall"], 4),
        },
        "confidence": round(score["overall"], 4),
        "reason_selected": (
            "The complete window is inside a Qwen-confirmed occurrence span "
            "and passes ratio, endpoint, visibility, and continuity checks."
        ),
        "recommended_sam_prompt": (
            f"Segment the {identity.get('object_identity', metadata['target_object'])} "
            "matching the source masked RGB anchor."
        ),
    }
    for window_alt, _, score_alt in valid[1:3]:
        result["alternative_segments"].append(
            {
                "window_id": window_alt["window_id"],
                "cam": window_alt["cam"],
                "start_frame": window_alt["start_frame"],
                "end_frame": window_alt["end_frame"],
                "requested_video_ratio": window_alt["requested_video_ratio"],
                "actual_sampled_frame_ratio": window_alt[
                    "actual_sampled_frame_ratio"
                ],
                "actual_frame_span_ratio": window_alt[
                    "actual_frame_span_ratio"
                ],
                "reason": (
                    f"Valid but lower whole-window score "
                    f"({score_alt['overall']:.3f})."
                ),
            }
        )
    return result


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.pad(image.convert("RGB"), size, color=(240, 240, 236))


def draw_frame_panel(
    frame_path: Path,
    evidence: Mapping[str, Any] | None,
    size: tuple[int, int],
) -> Image.Image:
    panel = fit_image(Image.open(frame_path), size)
    draw = ImageDraw.Draw(panel)
    if evidence and evidence.get("bbox_xyxy_normalized"):
        x1, y1, x2, y2 = evidence["bbox_xyxy_normalized"]
        box = (
            round(x1 * size[0]),
            round(y1 * size[1]),
            round(x2 * size[0]),
            round(y2 * size[1]),
        )
        draw.rectangle(box, outline=(0, 230, 170), width=5)
    frame_id = evidence.get("frame_id") if evidence else "?"
    present = "present" if evidence and evidence["target_present"] else "absent"
    draw.rectangle((0, 0, 190, 27), fill=(15, 21, 25))
    draw.text(
        (7, 5), f"frame {frame_id} | {present}", font=font(16), fill="white"
    )
    return panel


def render_result(
    case_dir: Path,
    work_case: Path,
    result: Mapping[str, Any],
    evidence: list[dict[str, Any]],
    catalog: Mapping[tuple[str, int], Path],
    temporal_index: Mapping[str, Any],
) -> None:
    output_dir = case_dir / "analysis_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1800, 1120), (246, 243, 234))
    draw = ImageDraw.Draw(canvas)
    title_font, body_font, small_font = font(32), font(20), font(16)
    target = clean_target_name(str(result["target_object"]))
    best = result.get("best_segment")
    window_text = best["window_id"] if best else "NONE (uncertain)"
    range_text = (
        f"{best['start_frame']}-{best['end_frame']}" if best else "NONE"
    )
    lines = [
        f"Mask-anchored temporal analysis: {target}",
        f"Source mask: {result['source_best_mask']} | source frame: {result['source_best_frame']}",
        f"Selected window_id: {window_text} | frame range: {range_text}",
        f"Status: {result['status']} | object: {result['source_identity'].get('object_identity', target)}",
    ]
    for row, line in enumerate(lines):
        draw.text(
            (35, 25 + row * 39),
            line,
            font=title_font if row == 0 else body_font,
            fill=(20, 27, 31),
        )

    source_panel = fit_image(
        Image.open(work_case / "source_anchor_isolated_rgb.png"), (360, 360)
    )
    canvas.paste(source_panel, (35, 205))
    draw.text(
        (35, 575),
        "SOURCE MASK RGB ANCHOR",
        font=body_font,
        fill=(20, 27, 31),
    )
    signature = str(result["source_identity"].get("visual_signature", ""))
    wrapped = textwrap.wrap(signature, width=43)[:8]
    for row, line in enumerate(wrapped):
        draw.text(
            (35, 612 + row * 24), line, font=small_font, fill=(45, 51, 54)
        )

    evidence_map = {
        (item["cam"], item["frame_id"]): item for item in evidence
    }
    selected_items: list[dict[str, Any]] = []
    if best:
        window = next(
            item
            for item in all_windows(temporal_index)
            if item["window_id"] == best["window_id"]
        )
        ids = window["representative_frame_ids"]
        selected_items = [
            evidence_map[(window["cam"], int(frame_id))] for frame_id in ids
        ]
    else:
        present = [item for item in evidence if item["target_present"]]
        selected_items = present[:5]
        if len(selected_items) < 5:
            selected_items += [
                item
                for item in evidence
                if item not in selected_items
            ][: 5 - len(selected_items)]

    draw.text(
        (445, 180),
        "SELECTED / OCCURRENCE EVIDENCE",
        font=body_font,
        fill=(20, 27, 31),
    )
    for index, item in enumerate(selected_items[:5]):
        path = catalog[(item["cam"], item["frame_id"])]
        panel = draw_frame_panel(path, item, (255, 180))
        x = 445 + (index % 5) * 265
        canvas.paste(panel, (x, 215))

    comparison = []
    if result.get("rejected_segments"):
        rejected_id = result["rejected_segments"][0]["window_id"]
        rejected_window = next(
            item
            for item in all_windows(temporal_index)
            if item["window_id"] == rejected_id
        )
        comparison = [
            evidence_map[(rejected_window["cam"], int(frame_id))]
            for frame_id in rejected_window["representative_frame_ids"]
        ]
        comparison_label = (
            f"COMPARISON: rejected {rejected_id} "
            f"({rejected_window['start_frame']}-{rejected_window['end_frame']})"
        )
    else:
        comparison_label = "COMPARISON: no additional generated window"
    draw.text(
        (445, 435),
        comparison_label,
        font=body_font,
        fill=(20, 27, 31),
    )
    for index, item in enumerate(comparison[:5]):
        path = catalog[(item["cam"], item["frame_id"])]
        panel = draw_frame_panel(path, item, (255, 180))
        x = 445 + (index % 5) * 265
        canvas.paste(panel, (x, 470))

    draw.line((35, 845, 1765, 845), fill=(155, 151, 141), width=2)
    runs = result.get("occurrence_spans", [])
    run_text = (
        ", ".join(
            f"{run['cam']}:{run['first_confirmed_frame']}-{run['last_confirmed_frame']}"
            for run in runs
        )
        or "none confirmed"
    )
    footer = [
        f"Confirmed occurrence spans: {run_text}",
        "Decision rule: full generated window must stay inside an occurrence span; "
        "20%-30% ratio; >=75% confirmed samples; no absence gap >2 samples.",
        f"Uncertainty: {result.get('uncertainty') or 'none'}",
        "Cyan boxes are Qwen localization evidence, not masks. Overlay colors are never used as object colors.",
    ]
    y = 870
    for line in footer:
        for wrapped_line in textwrap.wrap(line, width=145):
            draw.text((35, y), wrapped_line, font=small_font, fill=(35, 41, 44))
            y += 25
        y += 7
    output = output_dir / "selected_vs_rejected_region_comparison.jpg"
    canvas.save(output, quality=96)
    write_json(
        output_dir / "render_summary.json",
        {
            "status": result["status"],
            "source_mask": result["source_best_mask"],
            "source_object": result["source_identity"].get("object_identity"),
            "selected_window_id": best["window_id"] if best else None,
            "selected_frame_range": (
                [best["start_frame"], best["end_frame"]] if best else None
            ),
            "comparison_image": output.name,
            "visual_check": "pending_human_review",
        },
    )


def case_directories(root: Path, selected: list[str]) -> list[Path]:
    directories = [
        path
        for path in sorted(root.glob(CASE_GLOB))
        if path.is_dir()
        and (path / "metadata.json").exists()
        and (path / "temporal_window_index.json").exists()
    ]
    if selected:
        wanted = set(selected)
        directories = [
            path
            for path in directories
            if path.name in wanted
            or any(name in path.name for name in wanted)
        ]
    return directories


def process_case(
    case_dir: Path,
    work_root: Path,
    qwen: Qwen | None,
    render_only: bool,
    force: bool,
) -> dict[str, Any]:
    print(f"\n===== {case_dir.name} =====", flush=True)
    work_case = work_root / case_dir.name
    anchor = build_source_anchor(case_dir, work_case)
    temporal_index = read_json(case_dir / "temporal_window_index.json")
    catalog = make_frame_catalog(case_dir, temporal_index, work_case)
    evidence_path = work_case / "qwen_frame_evidence.json"
    identity_path = work_case / "qwen_source_identity.json"
    result_path = case_dir / "temporal_analysis_result.json"

    if render_only:
        result = read_json(result_path)
        evidence = read_json(evidence_path)
    else:
        if qwen is None:
            raise RuntimeError("Qwen is required for analysis")
        if identity_path.exists() and not force:
            identity = read_json(identity_path)
        else:
            identity = identify_source(qwen, anchor)
            write_json(identity_path, identity)
        print(f"Identity: {identity.get('object_identity')}", flush=True)

        evidence = []
        if evidence_path.exists() and not force:
            evidence = read_json(evidence_path)
        completed = {
            (item["cam"], int(item["frame_id"])) for item in evidence
        }
        ordered = sorted(catalog, key=lambda item: (item[0], item[1]))
        for position, (cam, frame_id) in enumerate(ordered, start=1):
            if (cam, frame_id) in completed:
                continue
            print(
                f"[{position}/{len(ordered)}] {cam} frame {frame_id}",
                flush=True,
            )
            item = assess_frame(
                qwen,
                anchor,
                identity,
                catalog[(cam, frame_id)],
                cam,
                frame_id,
            )
            evidence.append(item)
            evidence.sort(key=lambda value: (value["cam"], value["frame_id"]))
            write_json(evidence_path, evidence)
        result = analyze_windows(
            anchor["metadata"], temporal_index, identity, evidence
        )
        write_json(result_path, result)

    render_result(
        case_dir, work_case, result, evidence, catalog, temporal_index
    )
    best = result.get("best_segment")
    return {
        "case_id": case_dir.name,
        "target_object": result["target_object"],
        "status": result["status"],
        "source_mask": result["source_best_mask"],
        "source_object": result["source_identity"].get("object_identity"),
        "window_id": best["window_id"] if best else None,
        "start_frame": best["start_frame"] if best else None,
        "end_frame": best["end_frame"] if best else None,
        "occurrence_spans": result.get("occurrence_spans", []),
        "confidence": result["confidence"],
        "rendered_box_check": "pending_human_review",
        "uncertainty": result.get("uncertainty", ""),
    }


def main() -> None:
    args = parse_args()
    root = args.assets_root.resolve()
    work_root = (
        args.work_dir.resolve()
        if args.work_dir
        else root / ".qwen_temporal_work"
    )
    cases = case_directories(root, args.case)
    if not cases:
        raise SystemExit("No case directories found")
    qwen = None if args.render_only else Qwen(args.model, args.max_new_tokens)
    summaries = []
    for case_dir in cases:
        try:
            summaries.append(
                process_case(
                    case_dir, work_root, qwen, args.render_only, args.force
                )
            )
        except Exception as error:
            print(f"ERROR: {case_dir.name}: {error}", flush=True)
            summaries.append(
                {
                    "case_id": case_dir.name,
                    "status": "failed",
                    "error": str(error),
                }
            )
    write_json(root / "batch_temporal_analysis_summary.json", summaries)
    failed = [item for item in summaries if item["status"] == "failed"]
    print(
        f"\nCompleted {len(summaries)} cases; failed={len(failed)}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
