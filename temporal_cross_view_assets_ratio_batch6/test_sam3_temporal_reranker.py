#!/usr/bin/env python3
"""CPU-only tests for final visualization segmentation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import qwen_temporal_runner as temporal
import sam3_temporal_reranker as segmenter


class FakePredictor:
    def __init__(self, with_mask: bool = True) -> None:
        self.with_mask = with_mask
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": f"session-{len(self.requests)}"}
        if request["type"] == "close_session":
            return {"is_success": True}
        if not self.with_mask:
            return {
                "outputs": {
                    "out_obj_ids": np.array([]),
                    "out_binary_masks": np.array([]),
                    "out_probs": np.array([]),
                }
            }
        mask = np.zeros((60, 80), dtype=bool)
        mask[20:45, 25:60] = True
        return {
            "outputs": {
                "out_obj_ids": np.array([7]),
                "out_binary_masks": np.stack([mask]),
                "out_probs": np.array([0.92]),
            }
        }


def temporal_fixture(root: Path):
    case = root / "take__CPR_dummy"
    work = root / "work" / case.name
    case.mkdir(parents=True)
    work.mkdir(parents=True)
    frame_ids = [0, 30, 60, 90, 120]
    catalog = {}
    evidence = []
    for frame_id in frame_ids:
        path = root / f"frame_{frame_id:06d}.jpg"
        Image.new("RGB", (80, 60), (95, 105, 115)).save(path)
        catalog[("cam02", frame_id)] = path
        evidence.append(
            {
                "cam": "cam02",
                "frame_id": frame_id,
                "model_presence": "unverified",
                "target_present": False,
                "bbox_xyxy_normalized": None,
                "visibility": 0.0,
                "identity_confidence": 0.0,
                "evidence_score": 0.0,
            }
        )
    Image.new("RGB", (120, 120), "gray").save(
        work / "source_anchor_isolated_rgb.png"
    )
    window = {
        "window_id": "cam02_window_0000",
        "cam": "cam02",
        "frame_ids": frame_ids,
        "start_frame": 0,
        "end_frame": 120,
        "requested_video_ratio": 0.20,
        "actual_sampled_frame_ratio": 0.20,
        "actual_frame_span_ratio": 0.20,
        "continuity_ok": True,
    }
    index = {
        "cameras": {
            "cam02": {
                "sampled_frames": [
                    {"frame_id": frame_id} for frame_id in frame_ids
                ],
                "windows": [window],
                "continuity_split_threshold": 60,
            }
        }
    }
    selection = {key: window[key] for key in window if key != "frame_ids"}
    result = {
        "schema_version": 11,
        "status": "success",
        "pipeline_status": "awaiting_final_sam3_segmentation",
        "source_best_mask": "source_best_mask.png",
        "source_best_frame": 0,
        "source_identity": {"object_identity": "CPR dummy"},
        "target_object": "CPR dummy",
        "best_segment": dict(selection),
        "qwen_temporal_selection": dict(selection),
        "window_verifications": {
            window["window_id"]: {
                "frame_results": [
                    {
                        "frame_id": 60,
                        "target_present": True,
                        "bbox_xyxy_normalized": [0.3, 0.3, 0.8, 0.8],
                    }
                ]
            }
        },
        "global_challenger_comparison": None,
        "rejected_segments": [],
        "uncertainty": "",
    }
    return case, work, index, catalog, evidence, result


class FinalSegmentationTests(unittest.TestCase):
    def test_mask_bbox_uses_mask_pixels(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 5:15] = True
        self.assertEqual(segmenter.mask_bbox(mask), [0.25, 0.2, 0.75, 0.8])

    def test_qwen_box_rejects_higher_probability_wrong_instance(self) -> None:
        wrong_instance = np.zeros((20, 20), dtype=bool)
        wrong_instance[2:9, 2:9] = True
        qwen_instance = np.zeros((20, 20), dtype=bool)
        qwen_instance[12:18, 12:18] = True
        selected = segmenter.choose_final_mask(
            {
                "out_obj_ids": np.array([1, 2]),
                "out_binary_masks": np.stack([wrong_instance, qwen_instance]),
                "out_probs": np.array([0.90, 0.60]),
            },
            [0.6, 0.6, 0.9, 0.9],
            require_spatial_match=True,
        )
        self.assertEqual(selected["object_id"], 2)
        self.assertTrue(selected["qwen_spatial_match"])

    def test_qwen_box_is_primary_prompt_and_text_is_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "frame.jpg"
            Image.new("RGB", (80, 60), "gray").save(frame)
            predictor = FakePredictor()
            selected, method = segmenter.segment_final_frame(
                predictor,
                frame,
                Path(tmp) / "session",
                "CPR dummy",
                [0.3, 0.3, 0.8, 0.8],
            )
            self.assertIsNotNone(selected)
            self.assertEqual(method, "qwen_verified_box")
            add_prompts = [
                item for item in predictor.requests if item["type"] == "add_prompt"
            ]
            self.assertEqual(len(add_prompts), 1)
            self.assertIn("bounding_boxes", add_prompts[0])
            self.assertNotIn("text", add_prompts[0])

    def test_mask_without_qwen_overlap_is_rejected(self) -> None:
        wrong = np.zeros((20, 20), dtype=bool)
        wrong[1:5, 1:5] = True
        selected = segmenter.choose_final_mask(
            {
                "out_obj_ids": np.array([1]),
                "out_binary_masks": np.stack([wrong]),
                "out_probs": np.array([0.99]),
            },
            [0.6, 0.6, 0.9, 0.9],
            require_spatial_match=True,
        )
        self.assertIsNone(selected)

    def test_final_segmentation_never_changes_temporal_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case, work, index, catalog, _, result = temporal_fixture(Path(tmp))
            expected = dict(result["best_segment"])
            updated = segmenter.finalize_result(
                FakePredictor(with_mask=False),
                case,
                work,
                result,
                index,
                catalog,
            )
            self.assertEqual(updated["schema_version"], 14)
            self.assertEqual(updated["status"], "success")
            self.assertEqual(updated["best_segment"], expected)
            self.assertEqual(
                updated["final_segmentation"]["segmented_frame_count"], 0
            )
            self.assertFalse(
                updated["final_segmentation"]["changes_temporal_selection"]
            )

    def test_final_segmentation_writes_five_independent_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case, work, index, catalog, _, result = temporal_fixture(Path(tmp))
            stale = case / "analysis_outputs" / "final_sam3_masks" / "stale.png"
            stale.parent.mkdir(parents=True)
            Image.new("L", (5, 5), 255).save(stale)
            predictor = FakePredictor()
            updated = segmenter.finalize_result(
                predictor, case, work, result, index, catalog
            )
            frames = updated["final_segmentation"]["frame_results"]
            self.assertEqual(len(frames), 5)
            self.assertEqual(
                updated["final_segmentation"]["segmented_frame_count"], 5
            )
            self.assertTrue(
                all((case / item["mask_path"]).is_file() for item in frames)
            )
            self.assertFalse(stale.exists())

    def test_mask_overlay_changes_only_mask_pixels_and_hides_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (100, 100), (100, 100, 100)).save(frame)
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[50:80, 50:80] = 255
            Image.fromarray(mask).save(mask_path)
            panel = temporal.draw_frame_panel(
                frame,
                {
                    "frame_id": 42,
                    "inference_kind": "sam3-mask",
                    "mask_path": str(mask_path),
                    "bbox_xyxy_normalized": [0.1, 0.4, 0.9, 0.9],
                },
                (100, 100),
            )
            self.assertEqual(panel.getpixel((10, 40)), (100, 100, 100))
            self.assertNotEqual(panel.getpixel((60, 60)), (100, 100, 100))

    def test_schema14_renderer_uses_only_selected_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case, work, index, catalog, evidence, result = temporal_fixture(Path(tmp))
            result = segmenter.finalize_result(
                FakePredictor(), case, work, result, index, catalog
            )
            temporal.render_result(case, work, result, evidence, catalog, index)
            comparison = (
                case
                / "analysis_outputs"
                / "selected_vs_rejected_region_comparison.jpg"
            )
            with Image.open(comparison) as rendered:
                self.assertEqual(rendered.size, (3200, 1180))


if __name__ == "__main__":
    unittest.main()
