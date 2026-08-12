#!/usr/bin/env python3
"""Run mask-anchored cross-view temporal selection with Qwen3.5-4B.

Qwen identifies the masked source object and evaluates target frames. Window
selection is deterministic and enforces the temporal constraints in code.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import statistics
import textwrap
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


CASE_GLOB = "*__*"
EVIDENCE_SCHEMA_VERSION = 4
WINDOW_VERIFICATION_SCHEMA_VERSION = 5
RESULT_SCHEMA_VERSION = 7
CONFIRMED_THRESHOLD = 0.50
POSSIBLE_THRESHOLD = 0.45
WINDOW_SUPPORT_FRACTION = 0.60
WINDOW_CONFIRMED_FRACTION = 0.30
MAX_INTERNAL_ABSENT_RUN = 4
MAX_OCCURRENCE_GAP = 4
MAX_WINDOWS_TO_VERIFY = 8
MAX_REFINEMENT_SEEDS_PER_CAMERA = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--model",
        default="/scratch/users/ntu/gwang016/qwen35-4b/model",
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--summary-path",
        type=Path,
        help=(
            "Write this invocation's summary here. Defaults to "
            "<assets-root>/batch_temporal_analysis_summary.json."
        ),
    )
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--max-frame-probes-per-camera",
        type=int,
        default=40,
        help="Maximum source-centered coarse probes per target camera.",
    )
    parser.add_argument(
        "--probe-refinement-radius",
        type=int,
        default=4,
        help="Densely inspect this many sampled positions around a positive probe.",
    )
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


def model_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


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


def make_context_strip(
    catalog: Mapping[tuple[str, int], Path],
    cam: str,
    frame_id: int,
    work_case: Path,
) -> Path:
    """Show previous/current/next frames while keeping the queried frame clear."""
    output_dir = work_case / "target_context"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{cam}_{frame_id:06d}_context.jpg"
    if output.exists():
        return output

    ids = sorted(value for camera, value in catalog if camera == cam)
    position = ids.index(frame_id)
    selected_ids = [
        ids[max(0, position - 1)],
        frame_id,
        ids[min(len(ids) - 1, position + 1)],
    ]
    width, height, header = 640, 360, 42
    canvas = Image.new("RGB", (width * 3, height + header), (238, 238, 234))
    draw = ImageDraw.Draw(canvas)
    for index, selected_id in enumerate(selected_ids):
        image = ImageOps.fit(
            Image.open(catalog[(cam, selected_id)]).convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
        )
        x = index * width
        canvas.paste(image, (x, header))
        is_query = index == 1
        color = (0, 220, 175) if is_query else (60, 66, 70)
        draw.rectangle(
            (x + 2, header + 2, x + width - 3, header + height - 3),
            outline=color,
            width=7 if is_query else 2,
        )
        label = (
            f"QUERY frame {selected_id}"
            if is_query
            else f"context frame {selected_id}"
        )
        draw.text((x + 12, 12), label, fill=(20, 25, 28), font=font(18))
    canvas.save(output, quality=96)
    return output


def make_window_summary(
    window: Mapping[str, Any],
    catalog: Mapping[tuple[str, int], Path],
    work_case: Path,
    priority_frame_ids: Sequence[int] = (),
) -> Path:
    """Create a readable 3x3 summary spanning the complete candidate window."""
    output_dir = work_case / "window_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{window['window_id']}_v4.jpg"
    if output.exists():
        return output

    frame_ids = [int(value) for value in window["frame_ids"]]
    uniform_ids = [
        frame_ids[round(index * (len(frame_ids) - 1) / 8)]
        for index in range(9)
    ]
    prioritized = [
        int(frame_id)
        for frame_id in priority_frame_ids
        if int(frame_id) in frame_ids
    ][:3]
    sampled_ids = sorted(
        dict.fromkeys([*prioritized, *uniform_ids]),
        key=frame_ids.index,
    )
    if len(sampled_ids) > 9:
        keep = set(prioritized)
        remaining = [frame_id for frame_id in sampled_ids if frame_id not in keep]
        needed = max(0, 9 - len(keep))
        if needed:
            keep.update(
                remaining[
                    round(index * (len(remaining) - 1) / max(1, needed - 1))
                ]
                for index in range(needed)
            )
        sampled_ids = [frame_id for frame_id in sampled_ids if frame_id in keep]
    cell_width, cell_height, header = 640, 360, 40
    canvas = Image.new(
        "RGB",
        (cell_width * 3, (cell_height + header) * 3),
        (238, 238, 234),
    )
    draw = ImageDraw.Draw(canvas)
    for index, selected_id in enumerate(sampled_ids):
        row, column = divmod(index, 3)
        x = column * cell_width
        y = row * (cell_height + header)
        image = ImageOps.fit(
            Image.open(catalog[(window["cam"], selected_id)]).convert("RGB"),
            (cell_width, cell_height),
            method=Image.Resampling.LANCZOS,
        )
        canvas.paste(image, (x, y + header))
        draw.text(
            (x + 10, y + 10),
            (
                f"frame {selected_id} | scout evidence"
                if selected_id in prioritized
                else f"frame {selected_id}"
            ),
            fill=(20, 25, 28),
            font=font(18),
        )
    canvas.save(output, quality=96)
    return output


class Qwen:
    def __init__(self, model_path: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        print("Loading processor...", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        print("Loading Qwen3.5-4B...", flush=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def ask(self, images: list[Path], prompt: str) -> Any:
        image_content: list[dict[str, Any]] = []
        for image_path in images:
            image_content.append(
                {"type": "image", "path": str(image_path.resolve())}
            )
        last_output = ""
        for attempt in range(2):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\nCRITICAL RETRY: Return one complete minified JSON object "
                    "only. No markdown fence. Every string value must be at "
                    "most 18 words. Do not explain outside JSON."
                )
                print("Retrying truncated or invalid model JSON...", flush=True)
            content = [*image_content, {"type": "text", "text": attempt_prompt}]
            messages = [{"role": "user", "content": content}]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            inputs = inputs.to(self.model.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=(
                        self.max_new_tokens
                        if attempt == 0
                        else max(512, self.max_new_tokens)
                    ),
                    do_sample=False,
                    use_cache=True,
                )
            generated = generated[:, inputs["input_ids"].shape[1] :]
            last_output = self.processor.batch_decode(
                generated, skip_special_tokens=True
            )[0]
            try:
                return parse_model_json(last_output)
            except ValueError:
                if attempt == 0:
                    continue
        raise ValueError(
            "No valid JSON found after one concise retry: "
            f"{last_output[:1000]}"
        )


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
    query_frame_path: Path,
    context_strip_path: Path,
    cam: str,
    frame_id: int,
) -> dict[str, Any]:
    prompt = f"""
Perform cross-view temporal object matching without inventing evidence.

Image 1 is the complete first-person source frame.
Image 2 is a source context crop; its cyan box marks the binary-mask region and
cyan is NOT an object color.
Image 3 contains only the source-mask pixels in their original RGB colors;
gray/black is artificial background.
Image 4 is ONLY the third-person QUERY frame ({cam}, frame_id={frame_id}).
Image 5 contains previous, QUERY, and next frames for motion context. Judge
presence and localization only in Image 4. The bbox coordinate system is Image
4 alone, never the three-panel Image 5.

Target identity description:
{json.dumps(identity, ensure_ascii=False)}

Match the object indicated by the first-person binary mask, not merely its name.
Use shape, parts, material, scale, how it is held/used, and temporal context.
Do not demand impossible proof that two views show the same physical instance:
"confirmed" means the visible object matches the source anchor with specific
visual evidence and a valid localization box; "possible" means plausible but
partially hidden/small and still requires a localization box; "absent" means
no matching object is visible. Use null bbox for absent. If you cannot localize
the target, do not answer confirmed. State its concrete image location and
identity-specific appearance in visual_evidence.

The bbox must tightly enclose the target object itself in Image 4, using
[left, top, right, bottom] normalized to Image 4 width and height. Do not box a
person, hand, counter edge, appliance, or nearby container. Before returning a
positive answer, verify that the pixels inside your proposed box visibly match
the source object's distinctive shape, parts, and true colors.

Return JSON only:
{{
  "frame_id": {frame_id},
  "presence": "confirmed|possible|absent",
  "bbox_xyxy_normalized": null,
  "visibility": 0.0,
  "occlusion": "none|low|medium|high",
  "identity_confidence": 0.0,
  "visual_evidence": "maximum 18 words of frame-specific evidence"
}}
"""
    raw = qwen.ask(
        [
            anchor["frame_path"],
            anchor["context_path"],
            anchor["isolated_path"],
            query_frame_path,
            context_strip_path,
        ],
        prompt,
    )
    if not isinstance(raw, dict):
        raise ValueError("Frame response is not an object")
    confidence = max(
        0.0, min(1.0, float(raw.get("identity_confidence", 0.0)))
    )
    model_presence = str(raw.get("presence", "absent")).strip().lower()
    if model_presence not in {"confirmed", "possible", "absent"}:
        model_presence = "absent"
    box = normalized_box(raw.get("bbox_xyxy_normalized"))
    visibility = max(
        0.0, min(1.0, float(raw.get("visibility", 0.0)))
    )
    raw_model_presence = model_presence
    adjustment = None
    if model_presence in {"confirmed", "possible"} and (
        box is None or visibility <= 0.0
    ):
        model_presence = "absent"
        adjustment = "rejected_presence_without_bbox_or_visibility"
    if model_presence == "confirmed":
        evidence_score = confidence
        target_present = confidence >= CONFIRMED_THRESHOLD
    elif model_presence == "possible":
        evidence_score = confidence * 0.72
        target_present = confidence >= 0.70
    else:
        evidence_score = 0.0
        target_present = False
    return {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "cam": cam,
        "frame_id": frame_id,
        "model_presence": model_presence,
        "raw_model_presence": raw_model_presence,
        "presence_adjustment": adjustment,
        "target_present": target_present,
        "evidence_score": round(evidence_score, 4),
        "bbox_xyxy_normalized": box,
        "visibility": visibility,
        "occlusion": str(raw.get("occlusion", "high")),
        "identity_confidence": confidence,
        "visual_evidence": str(raw.get("visual_evidence", "")),
        "raw_model_response": raw,
        "inference_kind": "qwen",
        "probe_config": None,
    }


def scan_keys(
    catalog: Mapping[tuple[str, int], Path],
    source_frame: int,
    max_per_camera: int,
) -> list[tuple[str, int]]:
    """Search densely near source time, then expand exponentially outward."""
    selected: set[tuple[str, int]] = set()
    cameras = sorted({cam for cam, _ in catalog})
    for cam in cameras:
        ids = sorted(frame_id for camera, frame_id in catalog if camera == cam)
        center = min(range(len(ids)), key=lambda index: abs(ids[index] - source_frame))
        positions: list[int] = []

        def add(position: int) -> None:
            if 0 <= position < len(ids) and position not in positions:
                positions.append(position)

        add(center)
        for distance in range(1, 9):
            add(center - distance)
            add(center + distance)
        distance = 12
        while len(positions) < max_per_camera and distance < len(ids) * 2:
            add(center - distance)
            add(center + distance)
            distance = max(distance + 1, round(distance * 1.55))
        add(0)
        add(len(ids) - 1)
        required = list(dict.fromkeys([center, 0, len(ids) - 1]))[
            :max_per_camera
        ]
        chosen = required + [
            position for position in positions if position not in required
        ][: max(0, max_per_camera - len(required))]
        for position in chosen:
            selected.add((cam, ids[position]))
    return sorted(
        selected,
        key=lambda item: (item[0], abs(item[1] - source_frame), item[1]),
    )


def refinement_keys(
    catalog: Mapping[tuple[str, int], Path],
    actual: Sequence[Mapping[str, Any]],
    radius: int,
    source_frame: int,
    max_seeds_per_camera: int = MAX_REFINEMENT_SEEDS_PER_CAMERA,
) -> list[tuple[str, int]]:
    """Expand both directions around the strongest non-overlapping scouts."""
    selected: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for cam in sorted({camera for camera, _ in catalog}):
        ids = sorted(frame_id for camera, frame_id in catalog if camera == cam)
        positions = {frame_id: index for index, frame_id in enumerate(ids)}
        candidates = [
            item
            for item in actual
            if item["cam"] == cam and is_supported(item)
        ]
        candidates.sort(
            key=lambda item: (
                float(item.get("evidence_score", 0.0))
                * float(item.get("visibility", 0.0))
                * float(item.get("identity_confidence", 0.0)),
                -abs(int(item["frame_id"]) - source_frame),
            ),
            reverse=True,
        )
        seeds: list[Mapping[str, Any]] = []
        for item in candidates:
            item_position = positions[int(item["frame_id"])]
            if any(
                abs(item_position - positions[int(seed["frame_id"])]) <= radius
                for seed in seeds
            ):
                continue
            seeds.append(item)
            if len(seeds) >= max_seeds_per_camera:
                break
        for item in seeds:
            center = positions[int(item["frame_id"])]
            for distance in range(radius + 1):
                # Evaluate both frontiers at every distance; source proximity
                # never prevents expansion toward the visually stronger side.
                offsets = (0,) if distance == 0 else (-distance, distance)
                for offset in offsets:
                    position = center + offset
                    key = (cam, ids[position]) if 0 <= position < len(ids) else None
                    if key and key not in seen:
                        selected.append(key)
                        seen.add(key)
    return selected


def boxes_temporally_consistent(
    first: Sequence[float], second: Sequence[float]
) -> bool:
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    center_distance = math.dist(first_center, second_center)
    first_area = max(1e-6, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1e-6, (second[2] - second[0]) * (second[3] - second[1]))
    area_ratio = max(first_area, second_area) / min(first_area, second_area)
    return center_distance <= 0.35 and area_ratio <= 5.0


def interpolate_evidence(
    actual: list[dict[str, Any]],
    catalog: Mapping[tuple[str, int], Path],
    probe_config: str,
    max_position_gap: int,
) -> list[dict[str, Any]]:
    """Fill only intervals bracketed by two genuine localized detections."""
    actual_map = {(item["cam"], int(item["frame_id"])): item for item in actual}
    output: list[dict[str, Any]] = []
    for cam in sorted({camera for camera, _ in catalog}):
        ids = sorted(frame_id for camera, frame_id in catalog if camera == cam)
        positions = {frame_id: index for index, frame_id in enumerate(ids)}
        actual_ids = sorted(
            frame_id for camera, frame_id in actual_map if camera == cam
        )
        for frame_id in ids:
            key = (cam, frame_id)
            if key in actual_map:
                item = dict(actual_map[key])
                item["probe_config"] = probe_config
                output.append(item)
                continue
            position = bisect.bisect_left(actual_ids, frame_id)
            before_id = actual_ids[position - 1] if position else None
            after_id = actual_ids[position] if position < len(actual_ids) else None
            before = actual_map.get((cam, before_id)) if before_id is not None else None
            after = actual_map.get((cam, after_id)) if after_id is not None else None
            bracketed = bool(
                before
                and after
                and is_supported(before)
                and is_supported(after)
                and positions[int(after_id)] - positions[int(before_id)]
                <= max_position_gap
                and boxes_temporally_consistent(
                    before["bbox_xyxy_normalized"],
                    after["bbox_xyxy_normalized"],
                )
            )
            if bracketed:
                span = max(1, int(after_id) - int(before_id))
                ratio = (frame_id - int(before_id)) / span
                before_box = before["bbox_xyxy_normalized"]
                after_box = after["bbox_xyxy_normalized"]
                box = [
                    round(left + ratio * (right - left), 4)
                    for left, right in zip(before_box, after_box)
                ]
                confirmed = is_confirmed(before) and is_confirmed(after)
                confidence = min(
                    float(before["identity_confidence"]),
                    float(after["identity_confidence"]),
                )
                item = {
                    "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                    "cam": cam,
                    "frame_id": frame_id,
                    "model_presence": "confirmed" if confirmed else "possible",
                    "raw_model_presence": None,
                    "presence_adjustment": None,
                    "target_present": confirmed,
                    "evidence_score": round(
                        0.9
                        * min(
                            float(before["evidence_score"]),
                            float(after["evidence_score"]),
                        ),
                        4,
                    ),
                    "bbox_xyxy_normalized": box,
                    "visibility": min(
                        float(before["visibility"]), float(after["visibility"])
                    ),
                    "occlusion": "interpolated",
                    "identity_confidence": confidence,
                    "visual_evidence": (
                        f"Interpolated between localized frames {before_id} and {after_id}."
                    ),
                    "raw_model_response": None,
                    "inference_kind": "interpolated",
                    "probe_config": probe_config,
                }
            else:
                item = {
                    "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                    "cam": cam,
                    "frame_id": frame_id,
                    "model_presence": "absent",
                    "raw_model_presence": None,
                    "presence_adjustment": "not_bracketed_by_real_detections",
                    "target_present": False,
                    "evidence_score": 0.0,
                    "bbox_xyxy_normalized": None,
                    "visibility": 0.0,
                    "occlusion": "unknown",
                    "identity_confidence": 0.0,
                    "visual_evidence": "No two-sided localized evidence.",
                    "raw_model_response": None,
                    "inference_kind": "unscanned",
                    "probe_config": probe_config,
                }
            output.append(item)
    return output


def is_supported(item: Mapping[str, Any]) -> bool:
    return bool(
        item.get("bbox_xyxy_normalized")
        and float(item.get("visibility", 0.0)) > 0.0
        and float(item.get("evidence_score", 0.0)) >= POSSIBLE_THRESHOLD
    )


def is_confirmed(item: Mapping[str, Any]) -> bool:
    return (
        item.get("model_presence") == "confirmed"
        and item.get("bbox_xyxy_normalized")
        and float(item.get("visibility", 0.0)) > 0.0
        and float(item.get("identity_confidence", 0.0)) >= CONFIRMED_THRESHOLD
    )


def absent_run_max(items: list[dict[str, Any]]) -> int:
    maximum = current = 0
    for item in items:
        if is_supported(item):
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
    current_cam: str | None = None
    for item in evidence:
        item_cam = str(item["cam"])
        if current and current_cam != item_cam:
            while current and not is_supported(current[-1]):
                current.pop()
            if current:
                runs.append(current)
            current = []
            gap = 0
        current_cam = item_cam
        if is_supported(item):
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
        while current and not is_supported(current[-1]):
            current.pop()
        if current:
            runs.append(current)
    output = []
    for run in runs:
        positives = [item for item in run if is_supported(item)]
        if len(positives) < 2:
            continue
        output.append(
            {
                "cam": positives[0]["cam"],
                "first_confirmed_frame": positives[0]["frame_id"],
                "last_confirmed_frame": positives[-1]["frame_id"],
                "sampled_frame_count": len(run),
                "supported_frame_count": len(positives),
                "supported_fraction": len(positives) / len(run),
                "strict_confirmed_frame_count": sum(
                    1 for item in run if is_confirmed(item)
                ),
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
    present = [item for item in frame_items if is_supported(item)]
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
            "model_presence": item["model_presence"],
            "evidence_score": item["evidence_score"],
        }
        for item in selected
    ]


def score_window(
    window: Mapping[str, Any], frame_items: list[dict[str, Any]]
) -> dict[str, Any]:
    supported = [item for item in frame_items if is_supported(item)]
    confirmed = [item for item in frame_items if is_confirmed(item)]
    support_fraction = len(supported) / max(1, len(frame_items))
    confirmed_fraction = len(confirmed) / max(1, len(frame_items))
    visibility = statistics.mean(
        [item["visibility"] for item in supported] or [0.0]
    )
    identity = statistics.mean(
        [item["identity_confidence"] for item in supported] or [0.0]
    )
    mean_evidence = statistics.mean(
        [float(item.get("evidence_score", 0.0)) for item in frame_items]
    )
    longest_absence = absent_run_max(frame_items)
    continuity = max(0.0, 1.0 - longest_absence / 6.0)
    endpoint_ok = bool(
        frame_items
        and any(is_supported(item) for item in frame_items[:3])
        and any(is_supported(item) for item in frame_items[-3:])
    )
    actual_probes = [
        item
        for item in frame_items
        if item.get("inference_kind", "qwen") == "qwen"
    ]
    actual_boxes = [
        item["bbox_xyxy_normalized"]
        for item in actual_probes
        if is_supported(item)
    ]
    if len(actual_boxes) >= 2:
        centers = [
            ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            for box in actual_boxes
        ]
        movement = statistics.mean(
            math.dist(first, second)
            for first, second in zip(centers, centers[1:])
        )
        bbox_stability = max(0.0, 1.0 - movement / 0.35)
    else:
        bbox_stability = 0.0
    overall = (
        0.28 * support_fraction
        + 0.16 * confirmed_fraction
        + 0.20 * mean_evidence
        + 0.14 * visibility
        + 0.12 * identity
        + 0.10 * continuity
    )
    valid = (
        bool(window.get("continuity_ok"))
        and 0.20 <= float(window["requested_video_ratio"]) <= 0.30
        and support_fraction >= WINDOW_SUPPORT_FRACTION
        and confirmed_fraction >= WINDOW_CONFIRMED_FRACTION
        and longest_absence <= MAX_INTERNAL_ABSENT_RUN
        and endpoint_ok
    )
    return {
        "valid": valid,
        "support_fraction": support_fraction,
        "confirmed_fraction": confirmed_fraction,
        "longest_absent_run": longest_absence,
        "endpoint_evidence_ok": endpoint_ok,
        "mean_visibility": visibility,
        "mean_identity_confidence": identity,
        "mean_evidence_score": mean_evidence,
        "real_localized_sample_count": len(actual_boxes),
        "real_probe_support_fraction": len(actual_boxes) / max(1, len(actual_probes)),
        "bbox_temporal_stability": bbox_stability,
        "overall": overall,
    }


def source_frame_proximity(
    window: Mapping[str, Any],
    temporal_index: Mapping[str, Any],
    source_frame: int,
) -> float:
    """Weak same-take timing prior; never substitutes for visual evidence."""
    camera = temporal_index["cameras"][str(window["cam"])]
    video_span = max(1, int(camera.get("video_frame_span", 0)))
    midpoint = (int(window["start_frame"]) + int(window["end_frame"])) / 2
    return max(0.0, 1.0 - abs(midpoint - source_frame) / video_span)


def occurrence_capture(
    window: Mapping[str, Any],
    runs: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure how much of the strongest occurrence a candidate contains."""
    best = {
        "overlaps_occurrence": False,
        "occurrence_first_frame": None,
        "occurrence_last_frame": None,
        "occurrence_supported_count": 0,
        "captured_supported_count": 0,
        "captured_confirmed_count": 0,
        "captured_real_supported_count": 0,
        "captured_occurrence_fraction": 0.0,
    }
    for run in runs:
        if run["cam"] != window["cam"]:
            continue
        run_items = [
            item
            for item in evidence
            if item["cam"] == run["cam"]
            and int(run["first_confirmed_frame"])
            <= int(item["frame_id"])
            <= int(run["last_confirmed_frame"])
            and is_supported(item)
        ]
        captured = [
            item
            for item in run_items
            if int(window["start_frame"])
            <= int(item["frame_id"])
            <= int(window["end_frame"])
        ]
        fraction = len(captured) / max(1, len(run_items))
        candidate = {
            "overlaps_occurrence": len(captured) >= 2,
            "occurrence_first_frame": run["first_confirmed_frame"],
            "occurrence_last_frame": run["last_confirmed_frame"],
            "occurrence_supported_count": len(run_items),
            "captured_supported_count": len(captured),
            "captured_confirmed_count": sum(
                1 for item in captured if is_confirmed(item)
            ),
            "captured_real_supported_count": sum(
                1
                for item in captured
                if item.get("inference_kind", "qwen") == "qwen"
            ),
            "captured_occurrence_fraction": fraction,
        }
        if (
            candidate["captured_supported_count"],
            candidate["captured_occurrence_fraction"],
        ) > (
            best["captured_supported_count"],
            best["captured_occurrence_fraction"],
        ):
            best = candidate
    return best


def verification_candidates(
    temporal_index: Mapping[str, Any],
    evidence: list[dict[str, Any]],
    source_frame: int,
    limit: int = MAX_WINDOWS_TO_VERIFY,
) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
    """Compare equal 20% windows across time, not only source-near windows."""
    evidence_map = {
        (item["cam"], int(item["frame_id"])): item for item in evidence
    }
    windows = list(all_windows(temporal_index))
    preferred = [
        window
        for window in windows
        if abs(float(window["requested_video_ratio"]) - 0.20) < 1e-6
    ]
    if preferred:
        windows = preferred
    ranked: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    for window in windows:
        items = [
            evidence_map[(window["cam"], int(frame_id))]
            for frame_id in window["frame_ids"]
        ]
        score = score_window(window, items)
        score["source_frame_temporal_prior"] = source_frame_proximity(
            window, temporal_index, source_frame
        )
        score["verification_priority"] = (
            0.70 * score["overall"]
            + 0.20 * score["real_probe_support_fraction"]
            + 0.10 * score["source_frame_temporal_prior"]
        )
        ranked.append((window, score))

    # Reserve equal temporal bins per camera so a stronger segment far from the
    # synchronized source frame can still beat the source-centered candidate.
    cameras = sorted({str(window["cam"]) for window, _ in ranked})
    slots_per_camera = max(1, limit // max(1, len(cameras)))
    selected: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    selected_ids: set[str] = set()
    for cam in cameras:
        camera_entries = sorted(
            (entry for entry in ranked if str(entry[0]["cam"]) == cam),
            key=lambda entry: int(entry[0]["start_frame"]),
        )
        for slot in range(min(slots_per_camera, len(camera_entries))):
            start = math.floor(slot * len(camera_entries) / slots_per_camera)
            end = math.floor((slot + 1) * len(camera_entries) / slots_per_camera)
            bucket = camera_entries[start : max(start + 1, end)]
            choice = max(
                bucket, key=lambda entry: entry[1]["verification_priority"]
            )
            window_id = str(choice[0]["window_id"])
            if window_id not in selected_ids:
                selected.append(choice)
                selected_ids.add(window_id)

    for entry in sorted(
        ranked, key=lambda value: value[1]["verification_priority"], reverse=True
    ):
        window_id = str(entry[0]["window_id"])
        if window_id in selected_ids:
            continue
        selected.append(entry)
        selected_ids.add(window_id)
        if len(selected) >= limit:
            break
    return selected[:limit]


def window_verification_frame_ids(
    window: Mapping[str, Any],
    evidence_map: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[int]:
    """Keep target frames large while covering the full continuous window."""
    frame_ids = [int(frame_id) for frame_id in window["frame_ids"]]
    uniform = [
        frame_ids[round(index * (len(frame_ids) - 1) / 4)]
        for index in range(5)
    ]
    localized = sorted(
        (
            evidence_map[(window["cam"], frame_id)]
            for frame_id in frame_ids
            if is_supported(evidence_map[(window["cam"], frame_id)])
            and evidence_map[(window["cam"], frame_id)].get(
                "inference_kind", "qwen"
            )
            == "qwen"
        ),
        key=lambda item: (
            float(item.get("evidence_score", 0.0))
            * float(item.get("visibility", 0.0))
        ),
        reverse=True,
    )
    priority: list[int] = []
    positions = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    for item in localized:
        frame_id = int(item["frame_id"])
        if any(abs(positions[frame_id] - positions[other]) <= 1 for other in priority):
            continue
        priority.append(frame_id)
        if len(priority) >= 3:
            break
    return sorted(dict.fromkeys([*uniform, *priority]), key=positions.get)


def verify_candidate_windows(
    qwen: Qwen,
    anchor: Mapping[str, Any],
    identity: Mapping[str, Any],
    temporal_index: Mapping[str, Any],
    evidence: list[dict[str, Any]],
    catalog: Mapping[tuple[str, int], Path],
    work_case: Path,
    force: bool,
) -> dict[str, dict[str, Any]]:
    """Ask Qwen to inspect the full temporal extent of top candidate windows."""
    cache_path = work_case / "qwen_window_verifications.json"
    cached: dict[str, dict[str, Any]] = {}
    if cache_path.exists() and not force:
        cached = read_json(cache_path)
        if cached and any(
            item.get("verification_schema_version")
            != WINDOW_VERIFICATION_SCHEMA_VERSION
            for item in cached.values()
        ):
            print(
                "Cached window verification uses an obsolete schema; "
                "recomputing this case.",
                flush=True,
            )
            cached = {}

    evidence_map = {
        (item["cam"], int(item["frame_id"])): item for item in evidence
    }
    source_frame = int(anchor["metadata"]["source_best"]["frame_id"])
    selected = verification_candidates(
        temporal_index, evidence, source_frame, MAX_WINDOWS_TO_VERIFY
    )

    for position, (window, frame_score) in enumerate(selected, start=1):
        window_id = str(window["window_id"])
        if window_id in cached:
            continue
        print(
            f"[window {position}/{len(selected)}] verify {window_id} "
            f"{window['start_frame']}-{window['end_frame']}",
            flush=True,
        )
        target_frame_ids = window_verification_frame_ids(window, evidence_map)
        image_map = [
            {"image_number": index + 4, "frame_id": frame_id}
            for index, frame_id in enumerate(target_frame_ids)
        ]
        prompt = f"""
Independently verify several full-resolution frames from one continuous
third-person candidate window against a first-person binary-mask identity.

Images 1-3 are the complete source frame, source context crop with a synthetic
cyan marker, and original-RGB masked object. Images 4 onward are separate,
standalone third-person frames from continuous window {window_id}, frames
{window['start_frame']}-{window['end_frame']}. They are not a contact sheet:
{json.dumps(image_map, ensure_ascii=False)}

Target identity:
{json.dumps(identity, ensure_ascii=False)}

The largest first-person source mask occurs at frame {source_frame} in the
same take. Treat proximity to that frame only as a weak timing prior. It is not
proof of third-person visibility and must not override the window images.

Evaluate EVERY target frame independently. Match the masked object's structure,
parts, true color, and material. Do not infer presence from scene activity or
from another frame. A positive frame requires a tight bbox around the target
itself; do not box a person, hand, table, appliance, or nearby object. Bbox
coordinates are normalized within that one standalone image. The object may be
absent in most of the legal 20% window.

Return JSON only:
{{
  "window_id": "{window_id}",
  "frames": [
    {{
      "frame_id": {target_frame_ids[0]},
      "presence": "confirmed|possible|absent",
      "bbox_xyxy_normalized": null,
      "identity_confidence": 0.0
    }}
  ]
}}
Return exactly one frames entry for every frame ID in the image map. Do not
return an overall presence decision; deterministic code will calculate it.
"""
        raw = qwen.ask(
            [
                anchor["frame_path"],
                anchor["context_path"],
                anchor["isolated_path"],
                *[
                    catalog[(window["cam"], frame_id)]
                    for frame_id in target_frame_ids
                ],
            ],
            prompt,
        )
        if not isinstance(raw, dict):
            raise ValueError(f"Window verification is not JSON: {window_id}")
        returned = {}
        for item in raw.get("frames", []):
            if not isinstance(item, dict) or item.get("frame_id") is None:
                continue
            try:
                returned[int(item["frame_id"])] = item
            except (TypeError, ValueError):
                continue
        frame_results = []
        for frame_id in target_frame_ids:
            item = returned.get(frame_id, {})
            presence = str(item.get("presence", "absent")).lower()
            confidence = max(
                0.0,
                min(1.0, float(item.get("identity_confidence", 0.0))),
            )
            box = normalized_box(item.get("bbox_xyxy_normalized"))
            accepted = bool(
                box
                and (
                    (presence == "confirmed" and confidence >= CONFIRMED_THRESHOLD)
                    or (presence == "possible" and confidence >= 0.70)
                )
            )
            frame_results.append(
                {
                    "frame_id": frame_id,
                    "presence": presence if accepted else "absent",
                    "target_present": accepted,
                    "bbox_xyxy_normalized": box if accepted else None,
                    "identity_confidence": confidence if accepted else 0.0,
                }
            )
        positives = [item for item in frame_results if item["target_present"]]
        representative = max(
            positives,
            key=lambda item: float(item["identity_confidence"]),
            default=None,
        )
        thirds = max(1, len(target_frame_ids) // 3)
        cached[window_id] = {
            "verification_schema_version": WINDOW_VERIFICATION_SCHEMA_VERSION,
            "window_id": window_id,
            "cam": str(window["cam"]),
            "verified_frame_ids": target_frame_ids,
            "frame_results": frame_results,
            "target_in_beginning": any(
                item["target_present"] for item in frame_results[:thirds]
            ),
            "target_in_middle": any(
                item["target_present"] for item in frame_results[thirds:-thirds]
            ),
            "target_in_end": any(
                item["target_present"] for item in frame_results[-thirds:]
            ),
            "contains_target_occurrence": bool(positives),
            "visible_summary_position_count": len(positives),
            "first_visible_frame_id": (
                positives[0]["frame_id"] if positives else None
            ),
            "last_visible_frame_id": (
                positives[-1]["frame_id"] if positives else None
            ),
            "representative_frame_id": (
                representative["frame_id"] if representative else None
            ),
            "representative_bbox_xyxy_normalized": (
                representative["bbox_xyxy_normalized"]
                if representative
                else None
            ),
            "identity_confidence": (
                float(representative["identity_confidence"])
                if representative
                else 0.0
            ),
            "reason": (
                f"Independent full-frame matches: {len(positives)}/{len(frame_results)}."
            ),
            "raw_model_response": raw,
        }
        write_json(cache_path, cached)
    return cached


def analyze_windows(
    metadata: Mapping[str, Any],
    temporal_index: Mapping[str, Any],
    identity: Mapping[str, Any],
    evidence: list[dict[str, Any]],
    window_verifications: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evidence_map = {
        (item["cam"], item["frame_id"]): item for item in evidence
    }
    runs = occurrence_runs(evidence)
    source_frame = int(metadata["source_best"]["frame_id"])
    candidates = []
    for window in all_windows(temporal_index):
        items = [
            evidence_map[(window["cam"], int(frame_id))]
            for frame_id in window["frame_ids"]
        ]
        score = score_window(window, items)
        score["source_frame_temporal_prior"] = source_frame_proximity(
            window, temporal_index, source_frame
        )
        capture = occurrence_capture(window, runs, evidence)
        score.update(capture)
        verification = window_verifications.get(str(window["window_id"]))
        verification_ok = bool(
            verification
            and verification.get("contains_target_occurrence")
            and int(verification.get("visible_summary_position_count", 0)) >= 1
            and verification.get("representative_frame_id")
            in window["frame_ids"]
            and verification.get("representative_bbox_xyxy_normalized")
            and float(verification.get("identity_confidence", 0.0))
            >= CONFIRMED_THRESHOLD
        )
        score["window_verification"] = verification
        score["window_verification_ok"] = verification_ok
        verification_identity = float(
            (verification or {}).get("identity_confidence", 0.0)
        )
        verification_frame_fraction = int(
            (verification or {}).get("visible_summary_position_count", 0)
        ) / max(1, len((verification or {}).get("verified_frame_ids", [])))
        score["verification_frame_fraction"] = verification_frame_fraction
        if verification:
            score["overall"] = (
                0.80 * score["overall"]
                + 0.20 * verification_identity
            )
        score["selection_score"] = (
            0.30 * verification_frame_fraction
            + 0.25 * verification_identity
            + 0.15 * score["captured_occurrence_fraction"]
            + 0.10 * score["bbox_temporal_stability"]
            + 0.10 * score["real_probe_support_fraction"]
            + 0.05 * score["mean_visibility"]
            + 0.05 * score["source_frame_temporal_prior"]
        )
        score["valid"] = bool(
            window.get("continuity_ok")
            and 0.20 <= float(window["requested_video_ratio"]) <= 0.30
            and score["overlaps_occurrence"]
            and score["captured_supported_count"] >= 2
            and score["captured_confirmed_count"] >= 1
            and verification_ok
        )
        candidates.append((window, items, score))

    candidates.sort(
        key=lambda entry: (
            -abs(float(entry[0]["requested_video_ratio"]) - 0.20),
            entry[2]["selection_score"],
            entry[2]["captured_real_supported_count"],
            entry[2]["captured_supported_count"],
            entry[2]["source_frame_temporal_prior"],
        ),
        reverse=True,
    )
    valid = [entry for entry in candidates if entry[2]["valid"]]
    selected = valid[0] if valid else None
    challenger = None
    if selected:
        selected_window = selected[0]
        selected_span = max(
            1,
            int(selected_window["end_frame"])
            - int(selected_window["start_frame"]),
        )
        for entry in valid[1:]:
            window = entry[0]
            if window["cam"] != selected_window["cam"]:
                challenger = entry
                break
            intersection = max(
                0,
                min(int(window["end_frame"]), int(selected_window["end_frame"]))
                - max(
                    int(window["start_frame"]),
                    int(selected_window["start_frame"]),
                ),
            )
            if intersection / selected_span <= 0.35:
                challenger = entry
                break

    rejected = []
    for window, _, score in candidates:
        if selected and window["window_id"] == selected[0]["window_id"]:
            continue
        if selected:
            selected_window = selected[0]
            intersection = max(
                0,
                min(int(window["end_frame"]), int(selected_window["end_frame"]))
                - max(
                    int(window["start_frame"]),
                    int(selected_window["start_frame"]),
                ),
            )
            shorter_span = min(
                int(window["end_frame"]) - int(window["start_frame"]),
                int(selected_window["end_frame"])
                - int(selected_window["start_frame"]),
            )
            if shorter_span and intersection / shorter_span > 0.35:
                continue
        elif any(
            abs(int(window["start_frame"]) - int(item["start_frame"])) < 420
            for item in rejected
        ):
            continue
        reason = []
        if not score["overlaps_occurrence"]:
            reason.append("window does not capture two supported occurrence frames")
        if score["captured_confirmed_count"] < 1:
            reason.append("window has no confirmed localized occurrence frame")
        if not score["window_verification_ok"]:
            reason.append("occurrence verification did not pass")
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
        "schema_version": RESULT_SCHEMA_VERSION,
        "case_id": metadata.get("case_id", ""),
        "target_object": metadata["target_object"],
        "source_best_view": source["view_name"],
        "source_best_frame": source["frame_id"],
        "source_best_mask": source["source_mask"],
        "source_best_mask_overlay": source["source_mask_overlay"],
        "source_mask_area_ratio": source["mask_area_ratio"],
        "source_frame_temporal_prior": {
            "frame_id": source["frame_id"],
            "role": (
                "source-centered search origin in the synchronized take; "
                "visual evidence remains required"
            ),
        },
        "source_identity": identity,
        "occurrence_spans": runs,
        "selection_constraints": {
            "window_ratio_min": 0.20,
            "window_ratio_max": 0.30,
            "preferred_window_ratio": 0.20,
            "minimum_captured_supported_frames": 2,
            "minimum_captured_confirmed_frames": 1,
            "window_must_overlap_confirmed_occurrence": True,
            "target_may_be_absent_in_window_context": True,
            "occurrence_qwen_verification_required": True,
        },
        "selection_algorithm": {
            "name": "source_centered_bidirectional_search_with_global_challenger",
            "source_frame_is_weak_prior": True,
            "preferred_ratio": 0.20,
            "candidate_comparison": "equal-duration temporal bins per camera",
            "ranking_weights": {
                "independently_verified_frame_fraction": 0.30,
                "window_identity_verification": 0.25,
                "captured_occurrence_fraction": 0.15,
                "bbox_temporal_stability": 0.10,
                "real_probe_support_fraction": 0.10,
                "mean_visibility": 0.05,
                "source_frame_temporal_prior": 0.05,
            },
        },
        "window_verifications": window_verifications,
        "status": "success" if selected else "uncertain",
        "best_cam": selected[0]["cam"] if selected else None,
        "best_segment": None,
        "alternative_segments": [],
        "rejected_segments": rejected,
        "uncertainty": "",
        "confidence": 0.0,
        "global_challenger_comparison": None,
    }
    if not selected:
        result["uncertainty"] = (
            "No generated 20%-30% continuous window contains at least two "
            "localized scout samples and one independently verified full-frame match."
        )
        result["confidence"] = round(
            max((entry[2]["overall"] for entry in candidates), default=0.0), 4
        )
        return result

    window, items, score = selected
    verified_items = [
        item
        for item in (score["window_verification"] or {}).get(
            "frame_results", []
        )
        if item.get("target_present")
    ]
    verified_boxes = [
        item["bbox_xyxy_normalized"] for item in verified_items
    ]
    stable = (
        [round(statistics.median(values), 4) for values in zip(*verified_boxes)]
        if verified_boxes
        else None
    )
    verified_representatives = [
        {
            "frame_id": item["frame_id"],
            "target_region_xyxy": item["bbox_xyxy_normalized"],
            "visibility": 1.0,
            "occlusion": "independently_verified",
            "identity_confidence": item["identity_confidence"],
            "model_presence": item["presence"],
            "evidence_score": item["identity_confidence"],
        }
        for item in verified_items
    ]
    challenger_score = (
        float(challenger[2]["selection_score"]) if challenger else None
    )
    result["global_challenger_comparison"] = {
        "passed": challenger_score is None
        or float(score["selection_score"]) >= challenger_score,
        "selected_score": round(float(score["selection_score"]), 4),
        "challenger_window_id": challenger[0]["window_id"] if challenger else None,
        "challenger_score": (
            round(challenger_score, 4) if challenger_score is not None else None
        ),
        "margin": (
            round(float(score["selection_score"]) - challenger_score, 4)
            if challenger_score is not None
            else None
        ),
    }
    result["confidence"] = round(score["overall"], 4)
    result["best_segment"] = {
        "window_id": window["window_id"],
        "cam": window["cam"],
        "start_frame": window["start_frame"],
        "end_frame": window["end_frame"],
        "requested_video_ratio": window["requested_video_ratio"],
        "actual_sampled_frame_ratio": window["actual_sampled_frame_ratio"],
        "actual_frame_span_ratio": window["actual_frame_span_ratio"],
        "captured_occurrence": {
            key: score[key]
            for key in (
                "occurrence_first_frame",
                "occurrence_last_frame",
                "occurrence_supported_count",
                "captured_supported_count",
                "captured_confirmed_count",
                "captured_real_supported_count",
                "captured_occurrence_fraction",
            )
        },
        "independently_verified_occurrence": {
            "first_visible_frame": verified_items[0]["frame_id"],
            "last_visible_frame": verified_items[-1]["frame_id"],
            "verified_frame_count": len(verified_items),
            "tested_frame_count": len(
                (score["window_verification"] or {}).get(
                    "verified_frame_ids", []
                )
            ),
        },
        "representative_frames": verified_representatives,
        "target_region_summary": {
            "coordinate_system": "normalized_xyxy_per_frame",
            "stable_region_xyxy": stable,
            "region_explanation": (
                "Median of independently rechecked standalone-frame boxes; "
                "boxes are localization evidence, not masks."
            ),
        },
        "scores": {
            "support_fraction": round(score["support_fraction"], 4),
            "confirmed_fraction": round(score["confirmed_fraction"], 4),
            "identity_confidence": round(
                score["mean_identity_confidence"], 4
            ),
            "temporal_stability": round(
                max(0.0, 1.0 - score["longest_absent_run"] / 6.0), 4
            ),
            "bbox_temporal_stability": round(
                score["bbox_temporal_stability"], 4
            ),
            "occurrence_verification": score["window_verification"],
            "selection_score": round(score["selection_score"], 4),
            "overall": round(score["overall"], 4),
        },
        "confidence": round(score["overall"], 4),
        "reason_selected": (
            "The continuous 20% window has the strongest localized occurrence "
            "score and beats the same-duration global challenger."
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
                    f"Valid but lower occurrence-capture score "
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
    source = Image.open(frame_path).convert("RGB")
    fitted = ImageOps.contain(source, size, method=Image.Resampling.LANCZOS)
    image_x = (size[0] - fitted.width) // 2
    image_y = (size[1] - fitted.height) // 2
    panel = Image.new("RGB", size, (240, 240, 236))
    panel.paste(fitted, (image_x, image_y))
    draw = ImageDraw.Draw(panel)
    if evidence and evidence.get("bbox_xyxy_normalized"):
        x1, y1, x2, y2 = evidence["bbox_xyxy_normalized"]
        box = (
            round(image_x + x1 * fitted.width),
            round(image_y + y1 * fitted.height),
            round(image_x + x2 * fitted.width),
            round(image_y + y2 * fitted.height),
        )
        draw.rectangle(box, outline=(0, 230, 170), width=5)
    frame_id = evidence.get("frame_id") if evidence else "?"
    presence = evidence.get("model_presence", "absent") if evidence else "absent"
    inference_kind = evidence.get("inference_kind", "qwen") if evidence else "none"
    draw.rectangle((0, 0, 245, 27), fill=(15, 21, 25))
    draw.text(
        (7, 5),
        f"frame {frame_id} | {presence} | {inference_kind}",
        font=font(14),
        fill="white",
    )
    return panel


def render_selected_contact_sheet(
    case_dir: Path,
    output_dir: Path,
    result: Mapping[str, Any],
    evidence_map: Mapping[tuple[str, int], Mapping[str, Any]],
    temporal_index: Mapping[str, Any],
    work_case: Path,
) -> Path:
    output = output_dir / "selected_best_segment_contact_sheet.jpg"
    best = result.get("best_segment")
    if not best:
        canvas = Image.new("RGB", (1400, 620), (246, 243, 234))
        source = fit_image(
            Image.open(work_case / "source_anchor_isolated_rgb.png"),
            (430, 430),
        )
        canvas.paste(source, (45, 135))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (45, 35),
            "UNCERTAIN: NO LEGAL TEMPORAL WINDOW",
            font=font(32),
            fill=(145, 38, 32),
        )
        draw.text(
            (520, 150),
            f"Object: {result['source_identity'].get('object_identity')}",
            font=font(23),
            fill=(20, 27, 31),
        )
        draw.text(
            (520, 195),
            f"Source mask: {result['source_best_mask']}",
            font=font(20),
            fill=(20, 27, 31),
        )
        y = 255
        for line in textwrap.wrap(str(result.get("uncertainty", "")), width=68):
            draw.text((520, y), line, font=font(19), fill=(45, 51, 54))
            y += 30
        draw.text(
            (520, 470),
            "Selected window_id: NONE | frame range: NONE",
            font=font(21),
            fill=(145, 38, 32),
        )
        canvas.save(output, quality=96)
        return output

    window = next(
        item
        for item in all_windows(temporal_index)
        if item["window_id"] == best["window_id"]
    )
    sheet = Image.open(case_dir / window["contact_sheet"]).convert("RGB")
    banner_height = 105
    canvas = Image.new(
        "RGB", (sheet.width, sheet.height + banner_height), (246, 243, 234)
    )
    canvas.paste(sheet, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (15, 12),
        (
            f"SELECTED {best['window_id']} | frames "
            f"{best['start_frame']}-{best['end_frame']} | "
            f"object: {result['source_identity'].get('object_identity')}"
        ),
        font=font(22),
        fill=(20, 27, 31),
    )
    draw.text(
        (15, 55),
        (
            f"Source mask: {result['source_best_mask']} "
            f"(first-person frame {result['source_best_frame']})"
        ),
        font=font(19),
        fill=(20, 27, 31),
    )
    for cell in window["sheet_layout"]["cells"]:
        frame_id = int(cell["frame_id"])
        item = evidence_map[(window["cam"], frame_id)]
        box = item.get("bbox_xyxy_normalized") if is_supported(item) else None
        if not box:
            continue
        left, top, right, bottom = cell["image_xyxy"]
        x1, y1, x2, y2 = box
        mapped = (
            round(left + x1 * (right - left)),
            round(banner_height + top + y1 * (bottom - top)),
            round(left + x2 * (right - left)),
            round(banner_height + top + y2 * (bottom - top)),
        )
        draw.rectangle(mapped, outline=(0, 230, 170), width=4)
    canvas.save(output, quality=96)
    return output


def timeline_bin_samples(
    evidence: list[dict[str, Any]], count: int, highest: bool
) -> list[dict[str, Any]]:
    """Choose time-distributed evidence instead of falling back to first frames."""
    ordered = sorted(evidence, key=lambda item: (item["cam"], item["frame_id"]))
    if not ordered:
        return []
    output = []
    for index in range(count):
        start = math.floor(index * len(ordered) / count)
        end = math.floor((index + 1) * len(ordered) / count)
        bucket = ordered[start : max(start + 1, end)]
        output.append(
            max(bucket, key=lambda item: float(item.get("evidence_score", 0.0)))
            if highest
            else min(
                bucket, key=lambda item: float(item.get("evidence_score", 0.0))
            )
        )
    return output


def verified_render_evidence(
    evidence: list[dict[str, Any]],
    window_verifications: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Render only boxes independently rechecked as standalone full frames."""
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for item in evidence:
        cleared = dict(item)
        cleared.update(
            {
                "model_presence": "unverified",
                "target_present": False,
                "bbox_xyxy_normalized": None,
                "visibility": 0.0,
                "identity_confidence": 0.0,
                "evidence_score": 0.0,
                "inference_kind": "scout-only",
            }
        )
        output[(str(item["cam"]), int(item["frame_id"]))] = cleared
    for verification in window_verifications.values():
        cam = str(verification.get("cam", ""))
        for item in verification.get("frame_results", []):
            key = (cam, int(item["frame_id"]))
            if key not in output or not item.get("target_present"):
                continue
            confidence = float(item.get("identity_confidence", 0.0))
            previous = output[key]
            if (
                previous.get("inference_kind") == "window-verified"
                and float(previous.get("identity_confidence", 0.0)) >= confidence
            ):
                continue
            previous.update(
                {
                    "model_presence": "confirmed",
                    "target_present": True,
                    "bbox_xyxy_normalized": item["bbox_xyxy_normalized"],
                    "visibility": 1.0,
                    "identity_confidence": confidence,
                    "evidence_score": confidence,
                    "inference_kind": "window-verified",
                }
            )
    return output


def window_display_items(
    window: Mapping[str, Any],
    evidence_map: Mapping[tuple[str, int], Mapping[str, Any]],
    count: int = 5,
) -> list[Mapping[str, Any]]:
    """Show continuous-window endpoints plus localized occurrence evidence."""
    frame_ids = [int(frame_id) for frame_id in window["frame_ids"]]
    items = [evidence_map[(window["cam"], frame_id)] for frame_id in frame_ids]
    supported = [item for item in items if is_supported(item)]
    chosen_ids: list[int] = []

    def add(frame_id: int) -> None:
        if frame_id not in chosen_ids and len(chosen_ids) < count:
            chosen_ids.append(frame_id)

    add(frame_ids[0])
    if supported:
        for index in range(min(3, len(supported))):
            position = round(
                index
                * (len(supported) - 1)
                / max(1, min(3, len(supported)) - 1)
            )
            add(int(supported[position]["frame_id"]))
    add(frame_ids[-1])
    for index in range(count):
        add(frame_ids[round(index * (len(frame_ids) - 1) / max(1, count - 1))])
    return [
        evidence_map[(window["cam"], frame_id)]
        for frame_id in sorted(chosen_ids)
    ]


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

    evidence_map = verified_render_evidence(
        evidence, result.get("window_verifications", {})
    )
    selected_sheet = render_selected_contact_sheet(
        case_dir,
        output_dir,
        result,
        evidence_map,
        temporal_index,
        work_case,
    )
    selected_items: list[dict[str, Any]] = []
    if best:
        window = next(
            item
            for item in all_windows(temporal_index)
            if item["window_id"] == best["window_id"]
        )
        selected_items = window_display_items(window, evidence_map)
    else:
        selected_items = timeline_bin_samples(evidence, 5, highest=True)

    evidence_heading = (
        (
            f"CONTINUOUS SELECTED WINDOW {best['start_frame']}-{best['end_frame']} "
            "(ENDPOINTS + OBJECT EVIDENCE)"
        )
        if best
        else "TIMELINE EVIDENCE (NO WINDOW SELECTED)"
    )
    draw.text(
        (445, 180),
        evidence_heading,
        font=body_font,
        fill=(20, 27, 31),
    )
    for index, item in enumerate(selected_items[:5]):
        path = catalog[(item["cam"], item["frame_id"])]
        panel = draw_frame_panel(path, item, (255, 180))
        x = 445 + (index % 5) * 265
        canvas.paste(panel, (x, 215))

    comparison = []
    meaningful_rejected = [
        segment
        for segment in result.get("rejected_segments", [])
        if float(segment.get("scores", {}).get("overall", 0.0)) > 0.0
    ]
    if meaningful_rejected:
        if best:
            best_midpoint = (best["start_frame"] + best["end_frame"]) / 2
            rejected_segment = max(
                meaningful_rejected,
                key=lambda segment: abs(
                    (segment["start_frame"] + segment["end_frame"]) / 2
                    - best_midpoint
                ),
            )
        else:
            rejected_segment = meaningful_rejected[0]
        rejected_id = rejected_segment["window_id"]
        rejected_window = next(
            item
            for item in all_windows(temporal_index)
            if item["window_id"] == rejected_id
        )
        comparison = window_display_items(rejected_window, evidence_map)
        verification = rejected_segment.get("scores", {}).get(
            "window_verification"
        ) or {}
        candidate_kind = (
            "ALTERNATIVE OCCURRENCE WINDOW"
            if verification.get("contains_target_occurrence")
            else "UNSELECTED COMPARISON WINDOW"
        )
        comparison_label = (
            f"{candidate_kind}: {rejected_id} "
            f"({rejected_window['start_frame']}-{rejected_window['end_frame']})"
        )
    else:
        comparison = timeline_bin_samples(evidence, 5, highest=False)
        comparison_label = "COMPARISON: lowest-evidence samples across full timeline"
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
        "Decision rule: prefer a continuous 20% window that captures the most "
        "localized occurrence evidence; target may be absent in surrounding "
        "context; at least two supported and one confirmed sample required.",
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
            "selected_contact_sheet": selected_sheet.name,
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
    max_frame_probes_per_camera: int,
    probe_refinement_radius: int,
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
        evidence_cache_invalidated = False
        probe_config = (
            f"source_centered_v1:max={max_frame_probes_per_camera}:"
            f"radius={probe_refinement_radius}"
        )
        if evidence_path.exists() and not force:
            evidence = read_json(evidence_path)
            if evidence and any(
                item.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION
                or item.get("probe_config") != probe_config
                for item in evidence
            ):
                print(
                    "Cached frame evidence uses an obsolete schema; "
                    "recomputing this case.",
                    flush=True,
                )
                evidence = []
                evidence_cache_invalidated = True
        actual_evidence = [
            item
            for item in evidence
            if item.get("inference_kind", "qwen") == "qwen"
        ]
        completed = {
            (item["cam"], int(item["frame_id"])) for item in actual_evidence
        }

        def assess_pending(keys: Sequence[tuple[str, int]], stage: str) -> None:
            for position, (cam, frame_id) in enumerate(keys, start=1):
                if (cam, frame_id) in completed:
                    continue
                print(
                    f"[{stage} {position}/{len(keys)}] {cam} frame {frame_id}",
                    flush=True,
                )
                context_strip = make_context_strip(
                    catalog, cam, frame_id, work_case
                )
                item = assess_frame(
                    qwen,
                    anchor,
                    identity,
                    catalog[(cam, frame_id)],
                    context_strip,
                    cam,
                    frame_id,
                )
                item["probe_config"] = probe_config
                actual_evidence.append(item)
                actual_evidence.sort(
                    key=lambda value: (value["cam"], value["frame_id"])
                )
                completed.add((cam, frame_id))
                write_json(evidence_path, actual_evidence)

        coarse = scan_keys(
            catalog,
            int(anchor["metadata"]["source_best"]["frame_id"]),
            max_frame_probes_per_camera,
        )
        assess_pending(coarse, "coarse")
        refinement = refinement_keys(
            catalog,
            actual_evidence,
            probe_refinement_radius,
            int(anchor["metadata"]["source_best"]["frame_id"]),
        )
        assess_pending(refinement, "refine")
        evidence = interpolate_evidence(
            actual_evidence,
            catalog,
            probe_config,
            max_position_gap=probe_refinement_radius * 2 + 2,
        )
        write_json(evidence_path, evidence)
        window_verifications = verify_candidate_windows(
            qwen,
            anchor,
            identity,
            temporal_index,
            evidence,
            catalog,
            work_case,
            force or evidence_cache_invalidated,
        )
        windows_by_id = {
            str(window["window_id"]): window for window in all_windows(temporal_index)
        }
        rescue_keys = []
        for window_id, verification in window_verifications.items():
            window = windows_by_id.get(window_id)
            if not window or not verification.get("contains_target_occurrence"):
                continue
            for field in (
                "first_visible_frame_id",
                "representative_frame_id",
                "last_visible_frame_id",
            ):
                frame_id = verification.get(field)
                if frame_id in window["frame_ids"]:
                    rescue_keys.append((str(window["cam"]), int(frame_id)))
        assess_pending(rescue_keys, "window-rescue")
        rescue_refinement = refinement_keys(
            catalog,
            actual_evidence,
            probe_refinement_radius,
            int(anchor["metadata"]["source_best"]["frame_id"]),
        )
        assess_pending(rescue_refinement, "rescue-refine")
        evidence = interpolate_evidence(
            actual_evidence,
            catalog,
            probe_config,
            max_position_gap=probe_refinement_radius * 2 + 2,
        )
        write_json(evidence_path, evidence)
        result = analyze_windows(
            anchor["metadata"],
            temporal_index,
            identity,
            evidence,
            window_verifications,
        )
        per_camera_probe_count = {
            cam: sum(1 for item in actual_evidence if item["cam"] == cam)
            for cam in sorted({camera for camera, _ in catalog})
        }
        result["probe_statistics"] = {
            "probe_config": probe_config,
            "full_sampled_frame_count": len(catalog),
            "qwen_frame_probe_count": len(actual_evidence),
            "per_camera_qwen_frame_probe_count": per_camera_probe_count,
            "window_verification_count": len(window_verifications),
        }
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
    if args.max_frame_probes_per_camera <= 0:
        raise SystemExit("--max-frame-probes-per-camera must be positive")
    if args.probe_refinement_radius <= 0:
        raise SystemExit("--probe-refinement-radius must be positive")
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
                    case_dir,
                    work_root,
                    qwen,
                    args.render_only,
                    args.force,
                    args.max_frame_probes_per_camera,
                    args.probe_refinement_radius,
                )
            )
        except Exception as error:
            print(f"ERROR: {case_dir.name}: {error}", flush=True)
            failure = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "case_id": case_dir.name,
                "status": "failed",
                "best_segment": None,
                "error": str(error),
            }
            write_json(case_dir / "temporal_analysis_result.json", failure)
            summaries.append(failure)
    summary_path = (
        args.summary_path.resolve()
        if args.summary_path
        else root / "batch_temporal_analysis_summary.json"
    )
    write_json(summary_path, summaries)
    failed = [item for item in summaries if item["status"] == "failed"]
    print(
        f"\nCompleted {len(summaries)} cases; failed={len(failed)}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
