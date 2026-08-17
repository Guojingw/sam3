# Method feasibility report

- Case: `0099226c-9bec-44aa-ba43-2b90eb7b8379__stainless_milk_container_0`
- Target object: `stainless milk container_0`
- Analysis status: **success**
- Source reference: `aria01_214-1` frame `960`
- Source mask area ratio: `0.02920607889979339`
- Target cameras/windows inspected: `1` cameras / `51` windows

## Selected temporal segment

- Camera: `cam01`
- Frame range: `1260-2550`
- Window ID: `cam01_window_0020`
- Confidence: `0.72`
- Selection reason: The container remains visually consistent and localized on the stove area for most of the window, despite intermittent hand interaction.
- Recommended SAM3 prompt: `Initialize on the steel container body and lid; avoid the hand and the adjacent stove controls.`

### Temporal scores

- visibility_consistency: 0.840
- average_apparent_size: 0.790
- minimum_apparent_size: 0.740
- occlusion_consistency: 0.750
- identity_confidence: 0.720
- segmentation_suitability: 0.800
- temporal_stability: 0.820
- overall: 0.720

### Regional evidence

- Stable normalized region: `[0.51, 0.56, 0.68, 0.82]`
- Position drift: `low`
- Scale change: `low`
- Explanation: Estimated box around the stainless milk container based on the authoritative source mask overlay and the recurring lower-half working-area placement across the selected segment.

## Alternatives and rejected ranges

- Alternative 1: `cam01` frames `630-1680` — Earlier but more transitional and less stable overall.
- Alternative 2: `cam01` frames `2730-3780` — Later segment is usable but has more occlusion and drift.
- Rejected 1: `cam01` frames `0-840` —  Regional difference: 
- Rejected 2: `cam01` frames `3510-4350` —  Regional difference: 

## Uncertainty and feasibility conclusion

No uncertainty statement supplied.

The source-frame choice is grounded in decoded annotation mask area. The target temporal decision is an AI-assisted visual assessment over contact sheets, not a target mask annotation. It is feasible for candidate selection only when identity, continuity, and regional stability are sufficiently clear. Cases with small, transparent, heavily occluded, or confusable targets should remain uncertain/failed and should not be forced into SAM3.

## Generated artifacts

```json
{
  "selected": {
    "source_sheet": "target_temporal_windows/cam01/window_0020_ratio_0.30_frames_1260_2550.jpg",
    "output": "selected_best_segment_contact_sheet.jpg",
    "drawn_region_frames": [
      1260,
      1590,
      1920,
      2250,
      2550
    ]
  },
  "alternatives": [
    {
      "source_sheet": "target_temporal_windows/cam01/window_0010_ratio_0.25_frames_630_1680.jpg",
      "output": "alternative_segment_01.jpg",
      "drawn_region_frames": []
    },
    {
      "source_sheet": "target_temporal_windows/cam01/window_0040_ratio_0.25_frames_2730_3780.jpg",
      "output": "alternative_segment_02.jpg",
      "drawn_region_frames": []
    }
  ],
  "rejected": [
    {
      "source_sheet": "target_temporal_windows/cam01/window_0000_ratio_0.20_frames_0_840.jpg",
      "output": "rejected_segment_01.jpg",
      "drawn_region_frames": []
    },
    {
      "source_sheet": "target_temporal_windows/cam01/window_0050_ratio_0.20_frames_3510_4350.jpg",
      "output": "rejected_segment_02.jpg",
      "drawn_region_frames": []
    }
  ],
  "warnings": []
}
```
