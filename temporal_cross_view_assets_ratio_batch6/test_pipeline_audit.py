#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline_audit


class PipelineAuditTests(unittest.TestCase):
    def write_result(self, root: Path, payload: dict) -> Path:
        case = root / "take__object"
        case.mkdir()
        (case / "metadata.json").write_text("{}", encoding="utf-8")
        (case / "temporal_analysis_result.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return case

    def test_legacy_result_returns_to_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = self.write_result(
                Path(tmp), {"schema_version": 2, "status": "success"}
            )
            self.assertEqual(pipeline_audit.classify(case)[0], "needs_qwen")

    def test_schema11_selected_result_needs_sam3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = self.write_result(
                Path(tmp),
                {
                    "schema_version": 11,
                    "status": "success",
                    "pipeline_status": "awaiting_final_sam3_segmentation",
                    "best_segment": {"window_id": "cam01_dense_1"},
                },
            )
            self.assertEqual(pipeline_audit.classify(case)[0], "needs_sam3")

    def test_uncertain_schema13_is_complete_without_sam3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = self.write_result(
                Path(tmp),
                {
                    "schema_version": 13,
                    "status": "uncertain",
                    "pipeline_status": "complete",
                    "final_segmentation": {"frame_results": []},
                },
            )
            self.assertEqual(
                pipeline_audit.classify(case)[0], "complete_uncertain"
            )


if __name__ == "__main__":
    unittest.main()
