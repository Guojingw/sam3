#!/usr/bin/env python3
"""CPU-only tests for SAM3 temporal mask reranking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import sam3_temporal_reranker as reranker
import qwen_temporal_runner as temporal


def track(window_id: str, evidence: float, coverage: float) -> dict:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:12, 6:14] = True
    return {
        "window_id": window_id,
        "cam": "cam01",
        "start_frame": 0,
        "end_frame": 600,
        "requested_video_ratio": 0.20,
        "actual_sampled_frame_ratio": 0.20,
        "actual_frame_span_ratio": 0.20,
        "frame_results": [
            {
                "frame_id": frame_id,
                "target_present": True,
                "mask_area_ratio": 0.14,
                "mask_bbox_xyxy_normalized": [0.3, 0.25, 0.7, 0.6],
            }
            for frame_id in (0, 300, 600)
        ],
        "mask_track_metrics": {
            "present_frame_count": 3,
            "tested_frame_count": 3,
            "coverage_fraction": coverage,
            "longest_continuous_fraction": coverage,
            "mean_mask_area_ratio": 0.14,
            "total_mask_area_evidence": evidence,
            "mask_area_stability": 0.95,
            "verified_anchor_iou": 0.8,
            "verified_anchor_count": 2,
            "verified_anchor_match_count": 2,
            "prompt_mask_iou": 0.8,
            "seed_anchor_coverage": 0.9,
            "seed_mask_box_area_ratio": 1.1,
            "bbox_motion_stability": 0.95,
        },
        "_masks_by_frame": {0: mask, 300: mask, 600: mask},
    }


class Sam3RerankerTests(unittest.TestCase):
    def test_repository_root_contains_sam3_package(self) -> None:
        self.assertTrue(
            (reranker.REPOSITORY_ROOT / "sam3" / "model_builder.py").is_file()
        )

    def test_visual_box_tracking_disables_detector_hotstart(self) -> None:
        self.assertFalse(reranker.APPLY_TEMPORAL_DISAMBIGUATION)

    def test_mask_bbox_uses_mask_pixels(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 5:15] = True
        self.assertEqual(reranker.mask_bbox(mask), [0.25, 0.2, 0.75, 0.8])

    def test_expand_box_is_centered_and_clamped(self) -> None:
        for actual, expected in zip(
            reranker.expand_box([0.2, 0.3, 0.4, 0.5], 2.0),
            [0.1, 0.2, 0.5, 0.6],
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            reranker.expand_box([0.0, 0.0, 0.2, 0.2], 2.0),
            [0.0, 0.0, 0.3, 0.3],
        ):
            self.assertAlmostEqual(actual, expected)

    def test_reference_coverage_rewards_full_anchor_coverage(self) -> None:
        reference = [0.4, 0.4, 0.6, 0.6]
        self.assertEqual(
            reranker.box_reference_coverage([0.2, 0.2, 0.8, 0.8], reference),
            1.0,
        )
        self.assertAlmostEqual(
            reranker.box_reference_coverage([0.4, 0.4, 0.5, 0.5], reference),
            0.25,
        )

    def test_prompt_object_selection_does_not_take_first_object_id(self) -> None:
        wrong = np.zeros((20, 20), dtype=bool)
        wrong[1:3, 1:3] = True
        target = np.zeros((20, 20), dtype=bool)
        target[8:16, 9:17] = True
        object_id, mask, overlap = reranker.prompt_aligned_output_mask(
            {
                "out_obj_ids": np.array([10, 20]),
                "out_binary_masks": np.stack([wrong, target]),
            },
            [0.4, 0.35, 0.9, 0.85],
        )
        self.assertEqual(object_id, 20)
        self.assertTrue(np.array_equal(mask, target))
        self.assertGreater(overlap, 0.5)

    def test_prompt_object_selection_prefers_complete_anchor_coverage(self) -> None:
        partial = np.zeros((20, 20), dtype=bool)
        partial[8:12, 8:12] = True
        complete = np.zeros((20, 20), dtype=bool)
        complete[5:17, 5:17] = True
        object_id, mask, _ = reranker.prompt_aligned_output_mask(
            {
                "out_obj_ids": np.array([10, 20]),
                "out_binary_masks": np.stack([partial, complete]),
            },
            [0.35, 0.35, 0.65, 0.65],
        )
        self.assertEqual(object_id, 20)
        self.assertTrue(np.array_equal(mask, complete))

    def test_partial_seed_mask_retries_expanded_semantic_prompt(self) -> None:
        class FakePredictor:
            def __init__(self) -> None:
                self.add_requests = []
                self.current_mask = None

            def handle_request(self, request):
                if request["type"] == "start_session":
                    return {"session_id": "session"}
                if request["type"] == "close_session":
                    return {"is_success": True}
                self.add_requests.append(request)
                width = request["bounding_boxes"][0][2]
                mask = np.zeros((20, 20), dtype=bool)
                if width < 0.5:
                    mask[8:12, 8:12] = True
                else:
                    mask[5:15, 5:15] = True
                self.current_mask = mask
                return {
                    "outputs": {
                        "out_obj_ids": np.array([7]),
                        "out_binary_masks": np.stack([mask]),
                    }
                }

            def handle_stream_request(self, request):
                for frame_index in (0, 1):
                    yield {
                        "frame_index": frame_index,
                        "outputs": {
                            "out_obj_ids": np.array([7]),
                            "out_binary_masks": np.stack([self.current_mask]),
                        },
                    }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = {}
            for frame_id in (0, 1):
                path = root / f"frame_{frame_id}.jpg"
                Image.new("RGB", (40, 40), "white").save(path)
                catalog[("cam01", frame_id)] = path
            box = [0.3, 0.3, 0.7, 0.7]
            candidate = {
                "window_id": "window",
                "cam": "cam01",
                "start_frame": 0,
                "end_frame": 1,
                "requested_video_ratio": 0.2,
                "actual_sampled_frame_ratio": 0.2,
                "actual_frame_span_ratio": 0.2,
                "frame_ids": [0, 1],
                "seed_frame_id": 0,
                "seed_bbox_xyxy_normalized": box,
                "verified_seed_frames": [
                    {"frame_id": 0, "bbox_xyxy_normalized": box},
                    {"frame_id": 1, "bbox_xyxy_normalized": box},
                ],
                "sam_prompt_text": "CPR dummy",
            }
            predictor = FakePredictor()
            tracked = reranker.track_candidate(
                predictor, candidate, catalog, root / "tracks"
            )
            self.assertEqual(tracked["sam_prompt_scale"], 1.5)
            self.assertEqual(len(predictor.add_requests), 2)
            self.assertTrue(
                all(
                    request["text"] == "CPR dummy"
                    for request in predictor.add_requests
                )
            )
            self.assertEqual(
                tracked["mask_track_metrics"]["seed_anchor_coverage"], 1.0
            )

    def test_mask_score_prefers_continuous_captured_evidence(self) -> None:
        strong = track("strong", 2.0, 1.0)
        weak = track("weak", 0.5, 0.4)
        reranker.score_tracks([weak, strong])
        self.assertGreater(
            strong["mask_track_metrics"]["sam3_selection_score"],
            weak["mask_track_metrics"]["sam3_selection_score"],
        )

    def test_update_result_selects_distinct_anchor_consistent_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = {"status": "success", "best_segment": {"window_id": "old"}}
            updated = reranker.update_result(
                Path(tmp),
                result,
                [track("strong", 2.0, 1.0), track("weak", 0.3, 0.3)],
                minimum_score_margin=0.03,
            )
            self.assertEqual(updated["schema_version"], 12)
            self.assertEqual(updated["status"], "success")
            self.assertEqual(updated["best_segment"]["window_id"], "strong")
            self.assertEqual(updated["pipeline_status"], "complete")
            self.assertTrue(
                (Path(tmp) / "analysis_outputs" / "sam3_masks").is_dir()
            )

    def test_update_result_rejects_tied_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            updated = reranker.update_result(
                Path(tmp),
                {"status": "success", "best_segment": {"window_id": "old"}},
                [track("left", 1.0, 0.8), track("right", 1.0, 0.8)],
                minimum_score_margin=0.03,
            )
            self.assertEqual(updated["status"], "uncertain")
            self.assertIsNone(updated["best_segment"])
            self.assertTrue(
                (
                    Path(tmp)
                    / "analysis_outputs"
                    / "sam3_candidate_masks"
                    / "left"
                ).is_dir()
            )

    def test_materialize_png_as_ordered_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "frame.png"
            Image.new("RGB", (32, 24), "red").save(source)
            output = root / "frames"
            reranker.materialize_window_frames(
                {"cam": "cam01", "frame_ids": [42]},
                {("cam01", 42): source},
                output,
            )
            with Image.open(output / "000000.jpg") as image:
                self.assertEqual(image.size, (32, 24))

    def test_schema12_renderer_uses_dense_window_and_mask_box(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "take__object"
            work = root / "work" / case.name
            case.mkdir(parents=True)
            work.mkdir(parents=True)
            frame_ids = list(range(0, 300, 30))
            catalog = {}
            evidence = []
            for frame_id in frame_ids:
                path = root / f"frame_{frame_id}.jpg"
                Image.new("RGB", (320, 180), (80, 100, 120)).save(path)
                catalog[("cam01", frame_id)] = path
                evidence.append(
                    {
                        "cam": "cam01",
                        "frame_id": frame_id,
                        "model_presence": "absent",
                        "target_present": False,
                        "bbox_xyxy_normalized": None,
                        "visibility": 0.0,
                        "identity_confidence": 0.0,
                        "evidence_score": 0.0,
                    }
                )
            Image.new("RGB", (240, 240), "gray").save(
                work / "source_anchor_isolated_rgb.png"
            )
            mask_dir = case / "analysis_outputs" / "sam3_masks"
            mask_dir.mkdir(parents=True)
            mask_path = mask_dir / "frame_000060.png"
            Image.new("L", (320, 180), 0).save(mask_path)
            index = {
                "cameras": {
                    "cam01": {
                        "continuity_split_threshold": 60,
                        "windows": [
                            {
                                "window_id": "generated",
                                "cam": "cam01",
                                "frame_ids": frame_ids,
                                "start_frame": 0,
                                "end_frame": 270,
                            }
                        ],
                    }
                }
            }
            result = {
                "status": "success",
                "source_best_mask": "source_best_mask.png",
                "source_best_frame": 0,
                "source_identity": {"object_identity": "object"},
                "window_verifications": {},
                "best_segment": {
                    "window_id": "cam01_dense_r20_s00002",
                    "cam": "cam01",
                    "start_frame": 60,
                    "end_frame": 90,
                },
                "global_challenger_comparison": {
                    "challenger_window_id": "cam01_dense_r20_s00007"
                },
                "sam3_rerank": {"candidates": []},
                "sam3_frame_evidence": [
                    {
                        "frame_id": 60,
                        "target_present": True,
                        "mask_bbox_xyxy_normalized": [0.2, 0.2, 0.6, 0.7],
                        "mask_path": str(mask_path.relative_to(case)),
                    }
                ],
                "rejected_segments": [],
            }
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
