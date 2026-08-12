#!/usr/bin/env python3
"""Deterministic tests for temporal selection; no model weights are required."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

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
        for window in runner.all_windows(self.index):
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

    def test_selects_only_inside_long_occurrence(self) -> None:
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
        self.assertGreaterEqual(best["start_frame"], 1200)
        self.assertLessEqual(best["end_frame"], 2400)

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

    def test_single_frame_glimpse_remains_uncertain(self) -> None:
        evidence = synthetic_evidence(1200, 1200)
        result = runner.analyze_windows(
            self.metadata,
            self.index,
            self.identity,
            evidence,
            self.verifications(evidence),
        )
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["best_segment"])

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

    def test_verification_candidates_cover_the_full_timeline(self) -> None:
        evidence = synthetic_evidence(2100, 2400)
        selected = runner.verification_candidates(
            self.index,
            evidence,
            source_frame=int(self.metadata["source_best"]["frame_id"]),
            limit=8,
        )
        starts = sorted(int(window["start_frame"]) for window, _ in selected)
        self.assertTrue(
            all(float(window["requested_video_ratio"]) == 0.20 for window, _ in selected)
        )
        self.assertLess(starts[0], 1000)
        self.assertGreater(starts[-1], 3000)

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
                    }
                ],
            }
        }
        render_evidence = runner.verified_render_evidence(evidence, verified)
        self.assertIsNone(render_evidence[("cam01", 0)]["bbox_xyxy_normalized"])
        self.assertEqual(
            render_evidence[("cam01", 30)]["inference_kind"],
            "window-verified",
        )


if __name__ == "__main__":
    unittest.main()
