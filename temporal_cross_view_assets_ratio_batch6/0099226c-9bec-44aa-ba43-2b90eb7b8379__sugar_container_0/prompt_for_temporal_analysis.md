# Whole-camera-relative temporal cross-view analysis

Analyze this case locally. The goal is to select the relatively best continuous
portion covering approximately 20%–30% of a complete third-person camera
sequence. Do not choose a shorter clip simply because one frame looks clear.

## Case
- case_id: `0099226c-9bec-44aa-ba43-2b90eb7b8379__sugar_container_0`
- target object: `sugar container_0`
- source best view/frame: `aria01_214-1` / `1650`
- source mask area ratio: `0.03499786`
- target window counts: `{"cam01": 51}`

## Required files
1. `source_best_mask_overlay.png` — authoritative source identity.
2. `source_best_frame.jpg`
3. `source_best_mask.png`
4. `metadata.json`
5. `temporal_window_index.json`
6. Every referenced temporal contact sheet.

## Selection rules
1. Only compare windows produced by the ratio pipeline. Their length is based on
   the complete sampled frame count of the corresponding camera.
2. Prefer windows whose `actual_sampled_frame_ratio` is between 0.20 and 0.30.
3. Evaluate the complete window rather than isolated attractive frames.
4. Prefer high minimum visibility, identity consistency, manageable occlusion,
   stable apparent size, stable spatial location, and suitability for SAM3
   initialization and propagation.
5. Short accidental clear moments must not outweigh poor surrounding frames.
6. Copy `requested_video_ratio`, `actual_sampled_frame_ratio`, and
   `actual_frame_span_ratio` exactly from the selected window index entry.
7. Rank meaningfully different alternatives; do not fill the ranking with
   heavily overlapping windows from the same event.
8. Use normalized per-frame `[x_min, y_min, x_max, y_max]` boxes only as
   AI-estimated localization evidence. They are not masks.
9. Return `uncertain` or `failed` when no 20%–30% window is reliably better than
   the alternatives.

## Output
Write strict JSON to `temporal_analysis_result.json`:

```json
{
  "schema_version": 2,
  "case_id": "0099226c-9bec-44aa-ba43-2b90eb7b8379__sugar_container_0",
  "target_object": "sugar container_0",
  "source_best_view": "aria01_214-1",
  "source_best_frame": 1650,
  "source_mask_area_ratio": 0.034997861247417356,
  "status": "success | uncertain | failed",
  "best_cam": "camXX or null",
  "best_segment": {
    "window_id": "camXX_window_0000",
    "cam": "camXX",
    "start_frame": 0,
    "end_frame": 0,
    "requested_video_ratio": 0.25,
    "actual_sampled_frame_ratio": 0.25,
    "actual_frame_span_ratio": 0.25,
    "representative_frames": [
      {
        "frame_id": 0,
        "target_region_xyxy": [
          0.0,
          0.0,
          1.0,
          1.0
        ],
        "visibility": 0.0,
        "occlusion": "none | low | medium | high"
      }
    ],
    "target_region_summary": {
      "coordinate_system": "normalized_xyxy_per_frame",
      "stable_region_xyxy": [
        0.0,
        0.0,
        1.0,
        1.0
      ],
      "position_drift": "low | medium | high",
      "scale_change": "low | medium | high",
      "region_explanation": ""
    },
    "scores": {
      "visibility_consistency": 0.0,
      "average_apparent_size": 0.0,
      "minimum_apparent_size": 0.0,
      "occlusion_consistency": 0.0,
      "identity_confidence": 0.0,
      "segmentation_suitability": 0.0,
      "temporal_stability": 0.0,
      "overall": 0.0
    },
    "confidence": 0.0,
    "reason_selected": "",
    "recommended_sam_prompt": ""
  },
  "alternative_segments": [],
  "rejected_segments": [],
  "uncertainty": ""
}
```

After writing it, run the renderer with this case directory. The renderer can
use the same ratio-window contact sheets without modification.
