#!/usr/bin/env python3
"""Prepare and finalize a GPU-free manual temporal audit."""

from __future__ import annotations

import argparse
import bisect
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Validate filled ground truth and compute expected 20% windows.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def cases(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.glob("*__*"))
        if path.is_dir()
        and (path / "metadata.json").exists()
        and (path / "temporal_window_index.json").exists()
    ]


def uniform_samples(values: Sequence[int], count: int) -> list[int]:
    if len(values) <= count:
        return list(values)
    return sorted(
        {
            values[round(index * (len(values) - 1) / (count - 1))]
            for index in range(count)
        }
    )


def nearby_samples(values: Sequence[int], center: int, count: int) -> list[int]:
    if len(values) <= count:
        return list(values)
    position = bisect.bisect_left(values, center)
    start = max(0, min(len(values) - count, position - count // 2))
    return list(values[start : start + count])


def frame_sources(index: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for cam, camera in index["cameras"].items():
        for window in camera["windows"]:
            for cell in window["sheet_layout"]["cells"]:
                key = (cam, int(cell["frame_id"]))
                output.setdefault(
                    key,
                    {
                        "contact_sheet": window["contact_sheet"],
                        "image_xyxy": cell["image_xyxy"],
                    },
                )
    return output


def extract_frame(
    case_dir: Path,
    source: Mapping[str, Any],
    sheets: dict[Path, Image.Image],
) -> Image.Image:
    sheet_path = case_dir / str(source["contact_sheet"])
    if sheet_path not in sheets:
        sheets[sheet_path] = Image.open(sheet_path).convert("RGB")
    left, top, right, bottom = (int(value) for value in source["image_xyxy"])
    return sheets[sheet_path].crop((left, top, right, bottom))


def panel(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    header = 28
    canvas = Image.new("RGB", size, (244, 243, 237))
    fitted = ImageOps.contain(
        image.convert("RGB"),
        (size[0], size[1] - header),
        method=Image.Resampling.LANCZOS,
    )
    canvas.paste(
        fitted,
        ((size[0] - fitted.width) // 2, header + (size[1] - header - fitted.height) // 2),
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(60, 64, 65))
    draw.rectangle((0, 0, size[0] - 1, header), fill=(17, 24, 28))
    draw.text((7, 5), label, font=font(14), fill="white")
    return canvas


def render_case_audit(
    case_dir: Path,
    output: Path,
    metadata: Mapping[str, Any],
    index: Mapping[str, Any],
) -> dict[str, Any]:
    source_frame = int(metadata["source_best"]["frame_id"])
    sources = frame_sources(index)
    rows: list[tuple[str, list[tuple[str, int]]]] = []
    camera_summary: dict[str, Any] = {}
    for cam, camera in index["cameras"].items():
        frame_ids = sorted(
            frame_id for source_cam, frame_id in sources if source_cam == cam
        )
        timeline = uniform_samples(frame_ids, 12)
        nearby = nearby_samples(frame_ids, source_frame, 9)
        rows.append((f"{cam}: full timeline", [(cam, value) for value in timeline]))
        rows.append((f"{cam}: near source frame {source_frame}", [(cam, value) for value in nearby]))
        camera_summary[cam] = {
            "video_start_frame": camera.get("video_start_frame"),
            "video_end_frame": camera.get("video_end_frame"),
            "sampled_frame_count": camera.get("sampled_frame_count"),
            "timeline_audit_frames": timeline,
            "near_source_audit_frames": nearby,
        }

    cell_size = (300, 190)
    columns = 6
    row_height = 40 + math.ceil(12 / columns) * cell_size[1]
    width = columns * cell_size[0]
    source_height = 480
    height = source_height + len(rows) * row_height
    canvas = Image.new("RGB", (width, height), (247, 245, 238))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 18),
        f"Manual temporal audit: {metadata['target_object']}",
        font=font(28),
        fill=(20, 27, 31),
    )
    draw.text(
        (24, 58),
        f"case={case_dir.name} | largest source mask frame={source_frame}",
        font=font(18),
        fill=(35, 42, 45),
    )
    source_image = Image.open(case_dir / metadata["source_best"]["source_mask_overlay"])
    source_panel = panel(source_image, "AUTHORITATIVE SOURCE MASK OVERLAY", (420, 350))
    canvas.paste(source_panel, (24, 100))
    instructions = [
        "1. Confirm the mask covers the intended object.",
        "2. Ignore overlay color when identifying true object color.",
        "3. Find the same object in each third-person timeline.",
        "4. Record first/last visible frames in the JSON template.",
        "5. Use uncertain if identity cannot be verified.",
    ]
    for index_line, line in enumerate(instructions):
        draw.text(
            (475, 125 + index_line * 42),
            line,
            font=font(18),
            fill=(35, 42, 45),
        )

    sheets: dict[Path, Image.Image] = {}
    y = source_height
    for row_label, keys in rows:
        draw.text((20, y + 8), row_label, font=font(20), fill=(20, 27, 31))
        for item_index, key in enumerate(keys):
            image = extract_frame(case_dir, sources[key], sheets)
            frame_panel = panel(image, f"{key[0]} frame {key[1]}", cell_size)
            row, column = divmod(item_index, columns)
            canvas.paste(frame_panel, (column * cell_size[0], y + 40 + row * cell_size[1]))
        y += row_height
    for image in sheets.values():
        image.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)
    return camera_summary


def ground_truth_template(
    case_dir: Path,
    metadata: Mapping[str, Any],
    camera_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case_id": case_dir.name,
        "review_status": "pending",
        "source_mask_correct": None,
        "source_object_identity": "",
        "source_best_frame": int(metadata["source_best"]["frame_id"]),
        "target_presence": "pending|present|absent|uncertain",
        "target_cam": "",
        "first_visible_frame": None,
        "last_visible_frame": None,
        "identity_confidence": "low|medium|high",
        "expected_status": "pending",
        "expected_window": None,
        "notes": "",
        "available_cameras": camera_summary,
    }


def prepare(root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for position, case_dir in enumerate(cases(root), start=1):
        print(f"[{position}] preparing {case_dir.name}", flush=True)
        metadata = read_json(case_dir / "metadata.json")
        index = read_json(case_dir / "temporal_window_index.json")
        case_out = output_dir / case_dir.name
        case_out.mkdir(parents=True, exist_ok=True)
        audit_image = case_out / "manual_audit.jpg"
        camera_summary = render_case_audit(
            case_dir, audit_image, metadata, index
        )
        ground_truth_path = case_out / "manual_temporal_ground_truth.json"
        if not ground_truth_path.exists():
            write_json(
                ground_truth_path,
                ground_truth_template(case_dir, metadata, camera_summary),
            )
        entries.append(
            {
                "case_id": case_dir.name,
                "target_object": metadata["target_object"],
                "source_best_frame": metadata["source_best"]["frame_id"],
                "audit_image": str(audit_image.relative_to(output_dir)),
                "ground_truth": str(ground_truth_path.relative_to(output_dir)),
            }
        )
    write_json(output_dir / "manual_audit_index.json", entries)
    html_rows = [
        "<!doctype html><meta charset='utf-8'><title>Manual temporal audit</title>",
        "<style>body{font-family:Arial,sans-serif;background:#f5f1e7;margin:30px}"
        "section{margin:0 0 48px}img{max-width:100%;border:1px solid #555}"
        "code{background:#e5e0d5;padding:3px 6px}</style>",
        "<h1>Manual temporal audit</h1>",
    ]
    for entry in entries:
        html_rows.extend(
            [
                "<section>",
                f"<h2>{html.escape(entry['target_object'])}</h2>",
                f"<p><code>{html.escape(entry['case_id'])}</code></p>",
                f"<p>Fill: <code>{html.escape(entry['ground_truth'])}</code></p>",
                f"<img src='{html.escape(entry['audit_image'])}'>",
                "</section>",
            ]
        )
    (output_dir / "index.html").write_text("\n".join(html_rows), encoding="utf-8")
    print(f"Prepared {len(entries)} cases in {output_dir}", flush=True)


def choose_expected_window(
    index: Mapping[str, Any], cam: str, first: int, last: int
) -> Mapping[str, Any]:
    windows = [
        window
        for window in index["cameras"][cam]["windows"]
        if abs(float(window["requested_video_ratio"]) - 0.20) < 1e-6
    ]
    if not windows:
        raise ValueError(f"No 20% windows for {cam}")
    occurrence_midpoint = (first + last) / 2

    def rank(window: Mapping[str, Any]) -> tuple[float, float, float]:
        overlap = max(
            0,
            min(last, int(window["end_frame"]))
            - max(first, int(window["start_frame"])),
        )
        midpoint = (int(window["start_frame"]) + int(window["end_frame"])) / 2
        return overlap, -abs(midpoint - occurrence_midpoint), -int(window["start_frame"])

    return max(windows, key=rank)


def finalize(root: Path, output_dir: Path) -> None:
    results = []
    errors = []
    for case_dir in cases(root):
        truth_path = output_dir / case_dir.name / "manual_temporal_ground_truth.json"
        if not truth_path.exists():
            errors.append(f"{case_dir.name}: missing ground truth template")
            continue
        truth = read_json(truth_path)
        source_mask_correct = truth.get("source_mask_correct")
        if source_mask_correct is False:
            truth["review_status"] = "complete"
            truth["expected_status"] = "data_invalid"
            truth["expected_window"] = None
        elif source_mask_correct is not True:
            errors.append(f"{case_dir.name}: source_mask_correct must be true or false")
            continue
        presence = truth.get("target_presence")
        if source_mask_correct is False:
            pass
        elif presence in {"absent", "uncertain"}:
            truth["review_status"] = "complete"
            truth["expected_status"] = "uncertain"
            truth["expected_window"] = None
        elif presence == "present":
            cam = str(truth.get("target_cam", ""))
            first = truth.get("first_visible_frame")
            last = truth.get("last_visible_frame")
            index = read_json(case_dir / "temporal_window_index.json")
            if cam not in index["cameras"] or not isinstance(first, int) or not isinstance(last, int) or first > last:
                errors.append(f"{case_dir.name}: invalid cam/first/last visible frame")
                continue
            window = choose_expected_window(index, cam, first, last)
            truth["review_status"] = "complete"
            truth["expected_status"] = "success"
            truth["expected_window"] = {
                "window_id": window["window_id"],
                "cam": window["cam"],
                "start_frame": window["start_frame"],
                "end_frame": window["end_frame"],
                "requested_video_ratio": window["requested_video_ratio"],
                "contact_sheet": window["contact_sheet"],
            }
        else:
            errors.append(
                f"{case_dir.name}: target_presence must be present, absent, or uncertain"
            )
            continue
        write_json(truth_path, truth)
        results.append(
            {
                "case_id": case_dir.name,
                "expected_status": truth["expected_status"],
                "expected_window": truth["expected_window"],
            }
        )
    write_json(
        output_dir / "manual_ground_truth_summary.json",
        {"completed_count": len(results), "error_count": len(errors), "cases": results, "errors": errors},
    )
    for error in errors:
        print(f"ERROR: {error}", flush=True)
    print(f"Finalized {len(results)} cases; errors={len(errors)}", flush=True)
    if errors:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    root = args.assets_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.finalize:
        finalize(root, output_dir)
    else:
        prepare(root, output_dir)


if __name__ == "__main__":
    main()
