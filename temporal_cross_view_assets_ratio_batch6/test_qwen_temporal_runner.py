#!/usr/bin/env python3
"""Deterministic tests for temporal selection; no model weights are required."""

from __future__ import annotations

import unittest
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
                "evidence_schema_version": 2,
                "cam": "cam01",
                "frame_id": frame_id,
                "model_presence": "confirmed" if present else "absent",
                "target_present": present,
                "evidence_score": 0.9 if present else 0.0,
                # Presence must not depend on whether localization succeeded.
                "bbox_xyxy_normalized": None,
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
            if runner.score_window(window, items)["support_fraction"] == 1.0:
                output[window["window_id"]] = {
                    "whole_window_suitable": True,
                    "estimated_coverage_fraction": 1.0,
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

    def test_short_occurrence_remains_uncertain(self) -> None:
        evidence = synthetic_evidence(1200, 1800)
        result = runner.analyze_windows(
            self.metadata,
            self.index,
            self.identity,
            evidence,
            self.verifications(evidence),
        )
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["best_segment"])

    def test_presence_does_not_require_bbox(self) -> None:
        item = synthetic_evidence(1200, 1200)[40]
        self.assertTrue(runner.is_supported(item))
        self.assertIsNone(item["bbox_xyxy_normalized"])

    def test_uncertain_samples_span_timeline(self) -> None:
        evidence = synthetic_evidence(2100, 2100)
        frame_ids = [
            item["frame_id"]
            for item in runner.timeline_bin_samples(evidence, 5, highest=True)
        ]
        self.assertNotEqual(frame_ids, [0, 30, 60, 90, 120])
        self.assertGreater(frame_ids[-1] - frame_ids[0], 3000)


if __name__ == "__main__":
    unittest.main()
