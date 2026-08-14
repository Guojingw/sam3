#!/usr/bin/env python3
"""Deterministic tests for temporal selection; no model weights are required."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from PIL import Image

import qwen_temporal_runner as runner


ROOT = Path(__file__).resolve().parent
SUGAR = ROOT / "0099226c-9bec-44aa-ba43-2b90eb7b8379__sugar_container_0"


def synthetic_evidence(first: int, last: int) -> list[dict]:
    evidence = []
    for frame_id in range(0, 4351, 30):
        present = first <= frame_id <= last
        evidence.append(
            {
                "evidence_schema_version": runner.EVIDENCE_SCHEMA_VERSION,
                "cam": "cam01",
                "frame_id": frame_id,
                "model_presence": "confirmed" if present else "absent",
                "target_present": present,
                "evidence_score": 0.9 if present else 0.0,
                "bbox_xyxy_normalized": (
                    [0.25, 0.25, 0.45, 0.55] if present else None
                ),
                "visibility": 0.9 if present else 0.0,
                "occlusion": "low" if present else "none",
                "identity_confidence": 0.9 if present else 0.1,
                "visual_evidence": "synthetic",
            }
        )
    return evidence


class TemporalSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = runner.read_json(SUGAR / "metadata.json")
        cls.metadata["case_id"] = SUGAR.name
        cls.index = runner.read_json(SUGAR / "temporal_window_index.json")
        cls.identity = {"object_identity": "synthetic target"}

    def verifications(self, evidence: list[dict]) -> dict[str, dict]:
        evidence_map = {
            (item["cam"], item["frame_id"]): item for item in evidence
        }
        output = {}
        for window in runner.dense_sliding_windows(self.index):
            items = [
                evidence_map[(window["cam"], frame_id)]
                for frame_id in window["frame_ids"]
            ]
            supported = sum(runner.is_supported(item) for item in items)
            if supported >= 2:
                supported_items = [
                    item for item in items if runner.is_supported(item)
                ]
                representative = supported_items[0]
                output[window["window_id"]] = {
                    "verification_schema_version": (
                        runner.WINDOW_VERIFICATION_SCHEMA_VERSION
                    ),
                    "contains_target_occurrence": True,
                    "visible_summary_position_count": min(9, supported),
                    "verified_frame_ids": [
                        item["frame_id"] for item in items[:9]
                    ],
                    "frame_results": [
                        {
                            "frame_id": representative["frame_id"],
                            "presence": "confirmed",
                            "target_present": True,
                            "bbox_xyxy_normalized": [0.25, 0.25, 0.45, 0.55],
                            "identity_confidence": 0.9,
                            "matched_visible_cues": ["shape", "material"],
                            "conflicting_visible_cues": [],
                        }
                    ],
                    "representative_frame_id": representative["frame_id"],
                    "representative_bbox_xyxy_normalized": [
                        0.25,
                        0.25,
                        0.45,
                        0.55,
                    ],
                    "identity_confidence": 0.9,
                    "target_in_beginning": True,
                    "target_in_middle": True,
                    "target_in_end": True,
                    "reason": "synthetic",
                }
        return output

    def test_long_occurrence_is_fully_captured_with_legal_padding(self) -> None:
        evidence = synthetic_evidence(1200, 2400)
        result = runner.analyze_windows(
            self.metadata,
            self.index,
            self.identity,
            evidence,
            self.verifications(evidence),
        )
        self.assertEqual(result["status"], "success")
        best = result["best_segment"]
        self.assertEqual(result["qwen_temporal_selection"], best)
        self.assertEqual(
            result["pipeline_status"], "awaiting_final_sam3_segmentation"
        )
        self.assertLessEqual(best["start_frame"], 1200)
        self.assertGreaterEqual(best["end_frame"], 2400)
        self.assertEqual(
            best["captured_occurrence"]["captured_occurrence_fraction"], 1.0
        )

    def test_short_occurrence_selects_nearest_twenty_percent_window(self) -> None:
        evidence = synthetic_evidence(1200, 1800)
        result = runner.analyze_windows(
            self.metadata,
            self.index,
            self.identity,
            evidence,
            self.verifications(evidence),
        )
        self.assertEqual(result["status"], "success")
        best = result["best_segment"]
        self.assertEqual(best["requested_video_ratio"], 0.20)
        self.assertLessEqual(best["start_frame"], 1200)
        self.assertGreaterEqual(best["end_frame"], 1800)
        self.assertEqual(
            best["captured_occurrence"]["captured_occurrence_fraction"], 1.0
        )

    def test_single_unverified_glimpse_remains_uncertain(self) -> None:
        evidence = synthetic_evidence(1200, 1200)
        result = runner.analyze_windows(
            self.metadata,
            self.index,
            self.identity,
            evidence,
            self.verifications(evidence),
        )
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["schema_version"], 13)
        self.assertEqual(result["pipeline_status"], "complete")
        self.assertEqual(
            result["final_segmentation"]["role"], "visualization_only"
        )
        self.assertIsNone(result["best_segment"])

    def test_single_strong_crop_match_selects_padded_window(self) -> None:
        evidence = synthetic_evidence(1200, 1200)
        containing = next(
            window
            for window in runner.dense_sliding_windows(self.index)
            if 1200 in window["frame_ids"]
            and window["requested_video_ratio"] == 0.20
        )
        verification = {
            "verification_schema_version": runner.WINDOW_VERIFICATION_SCHEMA_VERSION,
            "verified_frame_ids": runner.representative_ids(
                containing["frame_ids"], 5
            ),
            "frame_results": [
                {
                    "frame_id": 1200,
                    "presence": "confirmed",
                    "target_present": True,
                    "localization_check_passed": True,
                    "bbox_xyxy_normalized": [0.25, 0.25, 0.45, 0.55],
                    "identity_confidence": 0.95,
                    "matched_visible_cues": ["distinct shape", "matching material"],
                    "conflicting_visible_cues": [],
                    "presentation_quality": 0.8,
                    "target_scale_score": 0.6,
                    "estimated_target_frame_fraction": 0.05,
                    "bbox_tightness": 0.8,
                    "object_completeness": 0.9,
                }
            ],
            "representative_frame_id": 1200,
            "representative_bbox_xyxy_normalized": [0.25, 0.25, 0.45, 0.55],
            "identity_confidence": 0.95,
        }
        result = runner.analyze_windows(
            self.metadata,
            self.index,
            self.identity,
            evidence,
            {containing["window_id"]: verification},
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["best_segment"]["requested_video_ratio"], 0.20)
        self.assertLessEqual(result["best_segment"]["start_frame"], 1200)
        self.assertGreaterEqual(result["best_segment"]["end_frame"], 1200)

    def test_presence_requires_bbox(self) -> None:
        item = synthetic_evidence(1200, 1200)[40]
        item["bbox_xyxy_normalized"] = None
        self.assertFalse(runner.is_supported(item))
        self.assertFalse(runner.is_confirmed(item))

    def test_uncertain_samples_span_timeline(self) -> None:
        evidence = synthetic_evidence(2100, 2100)
        frame_ids = [
            item["frame_id"]
            for item in runner.timeline_bin_samples(evidence, 5, highest=True)
        ]
        self.assertNotEqual(frame_ids, [0, 30, 60, 90, 120])
        self.assertGreater(frame_ids[-1] - frame_ids[0], 3000)

    def test_case_without_numeric_suffix_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "take__CPR_dummy"
            case.mkdir()
            (case / "metadata.json").write_text("{}", encoding="utf-8")
            (case / "temporal_window_index.json").write_text(
                "{}", encoding="utf-8"
            )
            discovered = runner.case_directories(Path(tmp), [])
        self.assertEqual([path.name for path in discovered], ["take__CPR_dummy"])

    def test_source_frame_is_used_only_as_window_tiebreaker(self) -> None:
        early, late = list(runner.all_windows(self.index))[0:2]
        source_frame = int(self.metadata["source_best"]["frame_id"])
        early_prior = runner.source_frame_proximity(
            early, self.index, source_frame
        )
        late_prior = runner.source_frame_proximity(late, self.index, source_frame)
        self.assertNotEqual(early_prior, late_prior)

    def test_source_centered_probe_schedule_is_bounded_and_covers_edges(self) -> None:
        catalog = {
            ("cam01", frame_id): Path(f"{frame_id}.jpg")
            for frame_id in range(0, 61951, 30)
        }
        keys = runner.scan_keys(catalog, source_frame=0, max_per_camera=40)
        self.assertLessEqual(len(keys), 40)
        self.assertIn(("cam01", 0), keys)
        self.assertIn(("cam01", 61950), keys)
        self.assertTrue(all(("cam01", value) in keys for value in range(0, 270, 30)))

    def test_inconsistent_boxes_are_not_interpolated(self) -> None:
        self.assertFalse(
            runner.boxes_temporally_consistent(
                [0.05, 0.05, 0.10, 0.10],
                [0.75, 0.75, 0.95, 0.95],
            )
        )
        self.assertTrue(
            runner.boxes_temporally_consistent(
                [0.20, 0.20, 0.35, 0.40],
                [0.22, 0.21, 0.37, 0.41],
            )
        )

    def test_refinement_is_bounded_when_every_scout_is_positive(self) -> None:
        catalog = {
            ("cam01", frame_id): Path(f"{frame_id}.jpg")
            for frame_id in range(0, 3000, 30)
        }
        evidence = synthetic_evidence(0, 4350)[:100]
        keys = runner.refinement_keys(
            catalog,
            evidence,
            radius=2,
            source_frame=1500,
            max_seeds_per_camera=3,
        )
        self.assertLessEqual(len(keys), 15)

    def test_verification_candidates_are_dense_scored_and_bounded(self) -> None:
        evidence = synthetic_evidence(2100, 2400)
        selected = runner.verification_candidates(
            self.index,
            evidence,
            source_frame=int(self.metadata["source_best"]["frame_id"]),
            limit=8,
        )
        starts = sorted(int(window["start_frame"]) for window, _ in selected)
        self.assertTrue(
            all(
                0.20 <= float(window["requested_video_ratio"]) <= 0.30
                for window, _ in selected
            )
        )
        self.assertLessEqual(len(selected), 8)
        self.assertTrue(any(start <= 2100 <= start + 1290 for start in starts))
        generated_starts = {
            int(window["start_frame"])
            for window in runner.all_windows(self.index)
        }
        self.assertTrue(any(start not in generated_starts for start in starts))

    def test_dense_windows_include_non_grid_cross_boundary_start(self) -> None:
        dense = runner.dense_sliding_windows(self.index, ratios=(0.20,))
        starts = [int(window["start_frame"]) for window in dense]
        generated_starts = {
            int(window["start_frame"])
            for window in runner.all_windows(self.index)
            if float(window["requested_video_ratio"]) == 0.20
        }
        self.assertTrue(any(start not in generated_starts for start in starts))
        self.assertIn(180, starts)

    def test_occurrence_runs_do_not_join_different_cameras(self) -> None:
        evidence = synthetic_evidence(0, 30)[:2]
        second_camera = [dict(item, cam="cam02") for item in evidence]
        runs = runner.occurrence_runs([*evidence, *second_camera])
        self.assertEqual([run["cam"] for run in runs], ["cam01", "cam02"])

    def test_window_display_includes_endpoints_and_short_occurrence(self) -> None:
        evidence = synthetic_evidence(390, 450)
        evidence_map = {
            (item["cam"], item["frame_id"]): item for item in evidence
        }
        window = next(runner.all_windows(self.index))
        displayed = runner.window_display_items(window, evidence_map)
        displayed_ids = [item["frame_id"] for item in displayed]
        self.assertIn(window["start_frame"], displayed_ids)
        self.assertIn(window["end_frame"], displayed_ids)
        self.assertTrue(any(runner.is_supported(item) for item in displayed))

    def test_window_verification_uses_full_timeline_and_localized_scouts(self) -> None:
        evidence = synthetic_evidence(390, 450)
        evidence_map = {
            (item["cam"], item["frame_id"]): item for item in evidence
        }
        window = next(runner.all_windows(self.index))
        frame_ids = runner.window_verification_frame_ids(window, evidence_map)
        self.assertIn(window["start_frame"], frame_ids)
        self.assertIn(window["end_frame"], frame_ids)
        self.assertTrue(any(390 <= frame_id <= 450 for frame_id in frame_ids))
        self.assertLessEqual(len(frame_ids), 8)

    def test_renderer_ignores_scout_boxes_until_independent_verification(self) -> None:
        evidence = synthetic_evidence(0, 60)[:3]
        verified = {
            "window": {
                "cam": "cam01",
                "frame_results": [
                    {
                        "frame_id": 30,
                        "target_present": True,
                        "bbox_xyxy_normalized": [0.1, 0.2, 0.3, 0.4],
                        "identity_confidence": 0.9,
                        "matched_visible_cues": ["shape", "material"],
                        "conflicting_visible_cues": [],
                    }
                ],
            }
        }
        render_evidence = runner.verified_render_evidence(evidence, verified)
        self.assertIsNone(render_evidence[("cam01", 0)]["bbox_xyxy_normalized"])
        self.assertEqual(
            render_evidence[("cam01", 30)]["inference_kind"],
            "crop-verified",
        )

    def test_candidate_identity_requires_two_cues_and_no_conflict(self) -> None:
        accepted = runner.candidate_identity_check(
            {
                "same_object_type": True,
                "candidate_crop_is_visually_verifiable": True,
                "matched_visible_cues": ["green screw cap", "red bottle body"],
                "conflicting_visible_cues": [],
                "identity_confidence": 0.9,
            }
        )
        color_only = runner.candidate_identity_check(
            {
                "same_object_type": True,
                "candidate_crop_is_visually_verifiable": True,
                "matched_visible_cues": ["red color"],
                "conflicting_visible_cues": [],
                "identity_confidence": 0.95,
            }
        )
        conflicted = runner.candidate_identity_check(
            {
                "same_object_type": True,
                "candidate_crop_is_visually_verifiable": True,
                "matched_visible_cues": ["green cap", "red body"],
                "conflicting_visible_cues": ["flat appliance control"],
                "identity_confidence": 0.95,
            }
        )
        self.assertTrue(accepted["identity_check_passed"])
        self.assertFalse(color_only["identity_check_passed"])
        self.assertFalse(conflicted["identity_check_passed"])

    def test_presentation_metrics_reject_loose_box(self) -> None:
        loose = runner.presentation_metrics(
            {
                "object_completeness": 1.0,
                "bbox_tightness": 0.2,
                "object_fill_fraction_in_bbox": 0.1,
            },
            [0.25, 0.25, 0.75, 0.75],
        )
        tight = runner.presentation_metrics(
            {
                "object_completeness": 0.9,
                "bbox_tightness": 0.9,
                "object_fill_fraction_in_bbox": 0.8,
            },
            [0.40, 0.40, 0.65, 0.70],
        )
        self.assertFalse(loose["localization_check_passed"])
        self.assertTrue(tight["localization_check_passed"])
        self.assertGreater(
            tight["presentation_quality"], loose["presentation_quality"]
        )

    def test_presentation_scores_reward_large_tight_target(self) -> None:
        scores = runner.verification_presentation_scores(
            {
                "frame_results": [
                    {
                        "target_present": True,
                        "localization_check_passed": True,
                        "presentation_quality": 0.9,
                        "target_scale_score": 0.8,
                        "estimated_target_frame_fraction": 0.08,
                        "bbox_tightness": 0.9,
                        "object_completeness": 0.95,
                    }
                ]
            }
        )
        self.assertEqual(scores["mean_presentation_quality"], 0.9)
        self.assertEqual(scores["mean_target_scale_score"], 0.8)

    def test_presentation_claim_cannot_override_verified_coverage(self) -> None:
        evidence = []
        for frame_id in range(40):
            evidence.append(
                {
                    "cam": "cam01",
                    "frame_id": frame_id,
                    "model_presence": "confirmed",
                    "target_present": True,
                    "evidence_score": 0.9,
                    "bbox_xyxy_normalized": [0.2, 0.2, 0.5, 0.5],
                    "visibility": 0.9,
                    "identity_confidence": 0.9,
                    "inference_kind": "qwen",
                }
            )
        source_windows = []
        for index, start in enumerate((0, 20)):
            source_windows.append(
                {
                    "window_id": f"cam01_window_{index:04d}",
                    "cam": "cam01",
                    "start_frame": start,
                    "end_frame": start + 19,
                    "frame_ids": list(range(start, start + 20)),
                    "requested_video_ratio": 0.20,
                    "actual_sampled_frame_ratio": 0.20,
                    "actual_frame_span_ratio": 0.20,
                    "continuity_ok": True,
                }
            )
        temporal_index = {
            "cameras": {
                "cam01": {
                    "video_frame_span": 39,
                    "windows": source_windows,
                }
            }
        }

        dense = runner.dense_sliding_windows(temporal_index, ratios=(0.20,))
        first = next(window for window in dense if window["start_frame"] == 0)
        second = next(window for window in dense if window["start_frame"] == 20)

        def verification(window: dict, count: int, scale: float) -> dict:
            start = window["start_frame"]
            frame_results = []
            for offset in range(count):
                frame_results.append(
                    {
                        "frame_id": start + offset,
                        "presence": "confirmed",
                        "target_present": True,
                        "localization_check_passed": True,
                        "bbox_xyxy_normalized": [0.2, 0.2, 0.5, 0.5],
                        "identity_confidence": 0.9,
                        "matched_visible_cues": ["shape", "material"],
                        "conflicting_visible_cues": [],
                        "presentation_quality": 0.9,
                        "target_scale_score": scale,
                        "estimated_target_frame_fraction": 0.08 * scale,
                        "bbox_tightness": 0.9,
                        "object_completeness": 0.9,
                    }
                )
            return {
                "contains_target_occurrence": True,
                "visible_summary_position_count": count,
                "verified_frame_ids": [start + offset * 4 for offset in range(5)],
                "representative_frame_id": start,
                "representative_bbox_xyxy_normalized": [0.2, 0.2, 0.5, 0.5],
                "identity_confidence": 0.9,
                "frame_results": frame_results,
            }

        result = runner.analyze_windows(
            {
                "case_id": "case",
                "target_object": "object",
                "source_best": {
                    "view_name": "aria01",
                    "frame_id": 0,
                    "source_mask": "source_best_mask.png",
                    "source_mask_overlay": "source_best_mask_overlay.png",
                    "mask_area_ratio": 0.1,
                },
            },
            temporal_index,
            {"object_identity": "object"},
            evidence,
            {
                first["window_id"]: verification(first, 2, 0.95),
                second["window_id"]: verification(second, 5, 0.25),
            },
        )
        self.assertEqual(result["best_segment"]["window_id"], second["window_id"])

    def test_candidate_panel_repeats_source_context_and_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame.jpg"
            source = root / "source.png"
            Image.new("RGB", (960, 540), (120, 80, 40)).save(frame)
            Image.new("RGB", (300, 300), (180, 20, 20)).save(source)
            panel_path = runner.make_candidate_identity_panel(
                frame,
                source,
                [0.45, 0.40, 0.55, 0.60],
                root,
                "window",
                30,
            )
            with Image.open(panel_path) as panel:
                self.assertEqual(panel.size, (1760, 500))

    def test_frame_catalog_prefers_original_target_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "case"
            work = root / "work"
            case.mkdir()
            original = root / "cam01_000030.jpg"
            Image.new("RGB", (960, 540), (10, 20, 30)).save(original)
            sheet = case / "sheet.jpg"
            Image.new("RGB", (480, 314), (40, 50, 60)).save(sheet)
            (case / "metadata.json").write_text(
                '{"take_dir": ""}', encoding="utf-8"
            )
            index = {
                "cameras": {
                    "cam01": {
                        "windows": [
                            {
                                "cam": "cam01",
                                "contact_sheet": "sheet.jpg",
                                "sheet_layout": {
                                    "cells": [
                                        {
                                            "frame_id": 30,
                                            "image_xyxy": [0, 44, 480, 314],
                                            "original_frame_path": str(original),
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                }
            }
            catalog = runner.make_frame_catalog(case, index, work)
            self.assertEqual(catalog[("cam01", 30)], original)
            stats = runner.frame_catalog_statistics(catalog, work)
            self.assertEqual(
                stats["cameras"]["cam01"]["source"], "native_target_frame"
            )
            self.assertEqual(
                stats["cameras"]["cam01"]["sample_image_size"], [960, 540]
            )

    def test_tile_box_maps_back_to_full_frame(self) -> None:
        mapped = runner.map_tile_box_to_frame(
            [0.25, 0.5, 0.75, 1.0], [0.2, 0.1, 0.6, 0.5]
        )
        self.assertEqual(mapped, [0.35, 0.55, 0.55, 0.75])
        boxes = runner.spatial_tile_boxes()
        self.assertEqual(len(boxes), 9)
        self.assertEqual(boxes[0][1], [0.0, 0.0, 0.5, 0.5])
        self.assertEqual(boxes[-1][1], [0.5, 0.5, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
