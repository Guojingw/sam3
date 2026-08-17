#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

from PIL import Image


if importlib.util.find_spec("pycocotools") is None:
    pycocotools = types.ModuleType("pycocotools")
    pycocotools.mask = types.SimpleNamespace()
    sys.modules["pycocotools"] = pycocotools
    sys.modules["pycocotools.mask"] = pycocotools.mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_temporal_cross_view_assets as generator


class ShortWindowGenerationTests(unittest.TestCase):
    def test_fragmented_runs_generate_best_effort_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            take = root / "take"
            camera = take / "cam01"
            output = root / "output"
            camera.mkdir(parents=True)
            output.mkdir()
            for run_index in range(6):
                for offset in range(3):
                    frame_id = run_index * 1000 + offset * 30
                    Image.new("RGB", (16, 12), "gray").save(
                        camera / f"frame_{frame_id:06d}.jpg"
                    )
            case = generator.CaseSpec(
                take,
                take / "annotation.json",
                "take",
                "object",
                "take__object",
            )
            index, _ = generator.generate_target_windows(
                case=case,
                case_dir=output,
                target_prefix="cam",
                window_sizes=[16],
                window_stride=8,
                window_ratios=[0.20, 0.25, 0.30],
                window_stride_ratio=0.05,
                target_sample_every=1,
                max_gap_factor=2.5,
                max_windows_per_cam=0,
                sheet_columns=2,
                cell_width=32,
                cell_height=24,
                header_height=4,
                jpeg_quality=80,
            )
            camera_record = index["cameras"]["cam01"]
            self.assertEqual(camera_record["contiguous_run_lengths"], [3] * 6)
            self.assertEqual(camera_record["window_count"], 6)
            self.assertTrue(
                all(
                    window["duration_fallback_allowed"]
                    for window in camera_record["windows"]
                )
            )


if __name__ == "__main__":
    unittest.main()
