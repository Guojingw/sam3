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
            {"case_id": "a__one", "take_id": "a", "quality_score": 0.9},
            {"case_id": "a__two", "take_id": "a", "quality_score": 0.8},
            {"case_id": "b__one", "take_id": "b", "quality_score": 0.7},
            {"case_id": "c__one", "take_id": "c", "quality_score": 0.6},
        ]
        selected = selector.diverse_recommendations(rows, 3)
        self.assertEqual(
            [item["case_id"] for item in selected],
            ["a__one", "b__one", "c__one"],
        )

    def test_multiple_objects_per_take_requires_opt_in(self) -> None:
        rows = [
            {"case_id": "a__one", "take_id": "a"},
            {"case_id": "a__two", "take_id": "a"},
        ]
        selected = selector.diverse_recommendations(
            rows, 2, one_per_take=False
        )
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
