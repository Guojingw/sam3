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


CASE_GLOB = "*__*"
EVIDENCE_SCHEMA_VERSION = 3
WINDOW_VERIFICATION_SCHEMA_VERSION = 2
CONFIRMED_THRESHOLD = 0.50
POSSIBLE_THRESHOLD = 0.45
WINDOW_SUPPORT_FRACTION = 0.60
WINDOW_CONFIRMED_FRACTION = 0.30
MAX_INTERNAL_ABSENT_RUN = 4
MAX_OCCURRENCE_GAP = 4
MAX_WINDOWS_TO_VERIFY = 8


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
) -> Path:
    """Create a readable 3x3 summary spanning the complete candidate window."""
    output_dir = work_case / "window_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{window['window_id']}.jpg"
    if output.exists():
        return output

    frame_ids = [int(value) for value in window["frame_ids"]]
    sampled_ids = [
        frame_ids[round(index * (len(frame_ids) - 1) / 8)]
        for index in range(9)
    ]
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
            f"frame {selected_id}",
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
Image 4 contains previous, QUERY, and next third-person frames. Judge only the
center QUERY frame ({cam}, frame_id={frame_id}); adjacent frames provide motion
and handling context.

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
    }


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
    for item in evidence:
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
    ranked = []
    for window in all_windows(temporal_index):
        items = [
            evidence_map[(window["cam"], int(frame_id))]
            for frame_id in window["frame_ids"]
        ]
        score = score_window(window, items)
        score["source_frame_temporal_prior"] = source_frame_proximity(
            window, temporal_index, source_frame
        )
        score["verification_priority"] = (
            score["overall"] + 0.12 * score["source_frame_temporal_prior"]
        )
        ranked.append((window, score))
    ranked.sort(
        key=lambda entry: entry[1]["verification_priority"], reverse=True
    )

    # Avoid spending all verification calls on nearly identical overlapping windows.
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for window, score in ranked:
        if any(
            window["cam"] == other["cam"]
            and abs(int(window["start_frame"]) - int(other["start_frame"])) < 180
            for other, _ in selected
        ):
            continue
        selected.append((window, score))
        if len(selected) >= MAX_WINDOWS_TO_VERIFY:
            break

    for position, (window, frame_score) in enumerate(selected, start=1):
        window_id = str(window["window_id"])
        if window_id in cached:
            continue
        print(
            f"[window {position}/{len(selected)}] verify {window_id} "
            f"{window['start_frame']}-{window['end_frame']}",
            flush=True,
        )
        summary_path = make_window_summary(window, catalog, work_case)
        prompt = f"""
Evaluate one complete third-person candidate window against a first-person
binary-mask identity anchor.

Images 1-3 are the complete source frame, source context crop with a synthetic
cyan marker, and original-RGB masked object. Image 4 is a 3x3 chronological
summary spanning the ENTIRE candidate window {window_id}, frames
{window['start_frame']}-{window['end_frame']}. The nine frame labels are real.

Target identity:
{json.dumps(identity, ensure_ascii=False)}

The largest first-person source mask occurs at frame {source_frame} in the
same take. Treat proximity to that frame only as a weak timing prior. It is not
proof of third-person visibility and must not override the window images.

The target does NOT need to remain visible throughout the complete window.
Judge whether this window contains a genuine, identity-matching occurrence in
at least two nearby summary positions. Generic scene activity and one isolated
ambiguous glimpse do not count. Report where in the window it appears. The
frame-level pre-score is only a hint:
{json.dumps(frame_score, ensure_ascii=False)}

Return JSON only:
{{
  "window_id": "{window_id}",
  "target_in_beginning": false,
  "target_in_middle": false,
  "target_in_end": false,
  "contains_target_occurrence": false,
  "visible_summary_position_count": 0,
  "identity_confidence": 0.0,
  "reason": "maximum 18 words about the localized occurrence"
}}
"""
        raw = qwen.ask(
            [
                anchor["frame_path"],
                anchor["context_path"],
                anchor["isolated_path"],
                summary_path,
            ],
            prompt,
        )
        if not isinstance(raw, dict):
            raise ValueError(f"Window verification is not JSON: {window_id}")
        confidence = max(
            0.0, min(1.0, float(raw.get("identity_confidence", 0.0)))
        )
        cached[window_id] = {
            "verification_schema_version": WINDOW_VERIFICATION_SCHEMA_VERSION,
            "window_id": window_id,
            "target_in_beginning": model_bool(raw.get("target_in_beginning")),
            "target_in_middle": model_bool(raw.get("target_in_middle")),
            "target_in_end": model_bool(raw.get("target_in_end")),
            "contains_target_occurrence": model_bool(
                raw.get("contains_target_occurrence")
            ),
            "visible_summary_position_count": max(
                0, min(9, int(raw.get("visible_summary_position_count", 0)))
            ),
            "identity_confidence": confidence,
            "reason": str(raw.get("reason", "")),
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
            and int(verification.get("visible_summary_position_count", 0)) >= 2
            and float(verification.get("identity_confidence", 0.0))
            >= CONFIRMED_THRESHOLD
        )
        score["window_verification"] = verification
        score["window_verification_ok"] = verification_ok
        if verification:
            score["overall"] = (
                0.80 * score["overall"]
                + 0.20
                * float(verification.get("identity_confidence", 0.0))
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
            entry[2]["captured_supported_count"],
            entry[2]["captured_occurrence_fraction"],
            float(
                (entry[2].get("window_verification") or {}).get(
                    "identity_confidence", 0.0
                )
            ),
            entry[2]["overall"],
            entry[2]["source_frame_temporal_prior"],
        ),
        reverse=True,
    )
    valid = [entry for entry in candidates if entry[2]["valid"]]
    selected = valid[0] if valid else None

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
        "schema_version": 5,
        "case_id": metadata.get("case_id", ""),
        "target_object": metadata["target_object"],
        "source_best_view": source["view_name"],
        "source_best_frame": source["frame_id"],
        "source_best_mask": source["source_mask"],
        "source_best_mask_overlay": source["source_mask_overlay"],
        "source_mask_area_ratio": source["mask_area_ratio"],
        "source_frame_temporal_prior": {
            "frame_id": source["frame_id"],
            "role": "weak same-take timing prior; visual evidence remains required",
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
        "window_verifications": window_verifications,
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
            "No generated 20%-30% continuous window contains at least two "
            "localized occurrence samples and passes occurrence verification."
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
        "captured_occurrence": {
            key: score[key]
            for key in (
                "occurrence_first_frame",
                "occurrence_last_frame",
                "occurrence_supported_count",
                "captured_supported_count",
                "captured_confirmed_count",
                "captured_occurrence_fraction",
            )
        },
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
            "support_fraction": round(score["support_fraction"], 4),
            "confirmed_fraction": round(score["confirmed_fraction"], 4),
            "identity_confidence": round(
                score["mean_identity_confidence"], 4
            ),
            "temporal_stability": round(
                max(0.0, 1.0 - score["longest_absent_run"] / 6.0), 4
            ),
            "occurrence_verification": score["window_verification"],
            "overall": round(score["overall"], 4),
        },
        "confidence": round(score["overall"], 4),
        "reason_selected": (
            "Preferred 20% continuous window captures the strongest localized "
            "occurrence and passes identity verification."
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
    presence = evidence.get("model_presence", "absent") if evidence else "absent"
    draw.rectangle((0, 0, 190, 27), fill=(15, 21, 25))
    draw.text(
        (7, 5), f"frame {frame_id} | {presence}", font=font(16), fill="white"
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
        ids = window["representative_frame_ids"]
        selected_items = [
            evidence_map[(window["cam"], int(frame_id))] for frame_id in ids
        ]
    else:
        selected_items = timeline_bin_samples(evidence, 5, highest=True)

    evidence_heading = (
        "SELECTED WINDOW EVIDENCE"
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
        comparison = [
            evidence_map[(rejected_window["cam"], int(frame_id))]
            for frame_id in rejected_window["representative_frame_ids"]
        ]
        comparison_label = (
            f"REJECTED CANDIDATE: {rejected_id} "
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
        if evidence_path.exists() and not force:
            evidence = read_json(evidence_path)
            if evidence and any(
                item.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION
                for item in evidence
            ):
                print(
                    "Cached frame evidence uses an obsolete schema; "
                    "recomputing this case.",
                    flush=True,
                )
                evidence = []
                evidence_cache_invalidated = True
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
            context_strip = make_context_strip(
                catalog, cam, frame_id, work_case
            )
            item = assess_frame(
                qwen,
                anchor,
                identity,
                context_strip,
                cam,
                frame_id,
            )
            evidence.append(item)
            evidence.sort(key=lambda value: (value["cam"], value["frame_id"]))
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
        result = analyze_windows(
            anchor["metadata"],
            temporal_index,
            identity,
            evidence,
            window_verifications,
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
            failure = {
                "schema_version": 4,
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
