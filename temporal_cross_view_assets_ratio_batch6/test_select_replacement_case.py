#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import select_replacement_case as selector


class ReplacementSelectionTests(unittest.TestCase):
    def test_largest_contiguous_run_splits_large_frame_gaps(self) -> None:
        self.assertEqual(
            selector.largest_contiguous_run([0, 30, 60, 90, 900, 930]), 4
        )

    def test_largest_contiguous_run_handles_empty_input(self) -> None:
        self.assertEqual(selector.largest_contiguous_run([]), 0)

    def test_existing_take_ids_are_read_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "take-a__object"
            case.mkdir()
            (case / "metadata.json").write_text(
                json.dumps({"take_id": "take-a"}), encoding="utf-8"
            )
            self.assertEqual(selector.existing_take_ids(root), {"take-a"})

    def test_default_recommendations_use_distinct_takes(self) -> None:
        rows = [
            {"case_id": "a__one", "take_id": "a", "object_name": "one", "quality_score": 0.9},
            {"case_id": "a__two", "take_id": "a", "object_name": "two", "quality_score": 0.8},
            {"case_id": "b__three", "take_id": "b", "object_name": "three", "quality_score": 0.7},
            {"case_id": "c__four", "take_id": "c", "object_name": "four", "quality_score": 0.6},
        ]
        selected = selector.diverse_recommendations(rows, 3)
        self.assertEqual(
            [item["case_id"] for item in selected],
            ["a__one", "b__three", "c__four"],
        )

    def test_default_recommendations_use_distinct_objects(self) -> None:
        rows = [
            {"case_id": "a__cup", "take_id": "a", "object_name": "cup"},
            {"case_id": "b__cup", "take_id": "b", "object_name": "cup"},
            {"case_id": "c__plate", "take_id": "c", "object_name": "plate"},
        ]
        selected = selector.diverse_recommendations(rows, 3)
        self.assertEqual(
            [item["case_id"] for item in selected], ["a__cup", "c__plate"]
        )

    def test_multiple_objects_per_take_requires_opt_in(self) -> None:
        rows = [
            {"case_id": "a__one", "take_id": "a", "object_name": "one"},
            {"case_id": "a__two", "take_id": "a", "object_name": "two"},
        ]
        selected = selector.diverse_recommendations(
            rows, 2, one_per_take=False
        )
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
