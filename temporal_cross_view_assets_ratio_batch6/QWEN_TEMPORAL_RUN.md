# First-person-mask anchored Qwen temporal selection

The required pipeline ends with Qwen temporal selection. SAM3 is optional and
only adds display masks.

## Decision ownership

1. Qwen reads `metadata.json`, the first-person source mask, the source overlay,
   and the isolated original-RGB masked object.
2. Qwen searches the synchronized third-person timeline from the source frame in
   both directions, refines positive areas, and verifies identity on standalone
   crops.
3. Code first derives the target's verified occurrence span in each third-person
   camera, then constructs an occurrence-centered window:
   - shorter than 20%: pad to at least 20% using the nearest frames on both sides;
   - between 20% and 30%: retain the exact occurrence span;
   - longer than 30%: keep the strongest continuous 30% slice.
   Boundary overflow is moved to the available side. If fragmented extraction
   leaves fewer contiguous frames than the preferred 20%, retain the best short
   run and record its ratio shortfall rather than returning no window.
4. Dense 20%, 25%, and 30% sliding windows remain comparison baselines. Qwen
   independently verifies the strongest occurrence-centered and dense windows.
   If an initial bbox fails crop identity, every verified candidate window is
   eligible for bounded nine-tile full-frame spatial rescue rather than only
   the first three windows.
5. Deterministic scoring selects the final continuous window. The result is
   saved in both `best_segment` and `qwen_temporal_selection`.
6. A brief occurrence may be accepted from one independently crop-verified
   frame when confidence is at least 0.90, at least two physical identity cues
   match, and no cue conflicts.
7. If Qwen has no valid identity-supported window, the result stays `uncertain`.

SAM3 does not search time, rerank windows, reject a Qwen window, or change
`success` to `uncertain`.

Successful Qwen output is schema 14 with `pipeline_status=complete`. Important
window fields include `duration_adjustment`, `padding_sampled_frames_before`,
`padding_sampled_frames_after`, and `captured_occurrence`.

## Optional SAM3 visualization

After a Qwen window is selected, SAM3 independently segments five representative
frames from that one window. Qwen's independently verified target bbox is the
required positive spatial prompt. SAM3 does not use a free text-only mask when
multiple same-class instances may exist. A mask with insufficient overlap with
the Qwen bbox is rejected instead of displaying the wrong instance. The nearest
verified Qwen bbox is used for display frames without an exact verified bbox,
and its source frame is recorded for audit.

Without SAM3, `final_segmentation.status` is `not_requested`. This is a complete
result, not a missing pipeline stage. If SAM3 is run later, it may populate
display masks but must not change `qwen_temporal_selection`.

Final segmentation schema 2 records
`spatial_policy=qwen_verified_box_required`, each frame's
`qwen_expected_box_xyxy_normalized`, and `qwen_box_source_frame`. Existing
schema-1 masks are automatically eligible for regeneration by the parallel
SAM3 submission script.

- `qwen_temporal_selection`: the time decision.
- `final_segmentation.role`: always `visualization_only`.
- `final_segmentation.frame_results`: five independent SAM3 display masks.
- `analysis_outputs/selected_vs_rejected_region_comparison.png`: source anchor,
  selected frames with mask overlays, and an unannotated alternative row.
- `analysis_outputs/final_sam3_masks/`: binary masks used by the renderer.

Segmentation failure is recorded by a missing mask for that display frame. It
does not alter the selected temporal window.

## NSCC run

On the login node:

```bash
cd "$HOME/worldmodel/sam3"
git fetch origin agent/add-temporal-batch6
git checkout agent/add-temporal-batch6
git pull --ff-only origin agent/add-temporal-batch6

EASY6="$HOME/scratch/temporal_cross_view_assets_ratio_easy6"
WORK="$HOME/scratch/qwen35-4b/temporal_easy6_native_tiles_v3"
cd "$HOME/worldmodel/sam3/temporal_cross_view_assets_ratio_batch6"
```

Audit every case before submitting jobs. Legacy results return to Qwen:

```bash
"$HOME/.conda/envs/sam3/bin/python" pipeline_audit.py --assets-root "$EASY6"
```

Run Qwen selection first. The existing parallel Qwen submission script can be
used for all cases:

```bash
bash submit_qwen_temporal_parallel.sh "$EASY6" "$WORK"
qstat -u "$USER"
```

Inspect the selected ranges and padding decisions:

```bash
for f in "$EASY6"/*/temporal_analysis_result.json; do
  jq -r '[.case_id, .status, .pipeline_status,
          (.best_segment.window_id // "NONE"),
          (.best_segment.start_frame // "NONE"),
          (.best_segment.end_frame // "NONE"),
          (.best_segment.duration_adjustment // "NONE"),
          (.best_segment.padding_sampled_frames_before // 0),
          (.best_segment.padding_sampled_frames_after // 0)] | @tsv' "$f"
done
```

This update changes the independently verified candidate windows. Run Qwen once
normally (not `--rescore-only`). Existing frame evidence is reused, while the
old verification cache is automatically replaced. For one case:

```bash
"$HOME/.conda/envs/sam3/bin/python" qwen_temporal_runner.py \
  --assets-root "$EASY6" \
  --work-dir "$WORK" \
  --case '<case_id>'
```

After schema-9 window verifications exist, later scoring-only code changes can
be applied with `--rescore-only`. SAM3 submission is optional:

```bash
bash submit_sam3_temporal_parallel.sh "$EASY6" "$WORK"
qstat -u "$USER"
```

Run the audit again after Qwen. `needs_qwen` must reach either `complete` or
`complete_uncertain`.
`complete_uncertain` is a completed pipeline with no identity-verified window,
not an execution failure, and must not be relabeled as a semantic success.

## Selecting a replacement dataset case

Use the dataset-wide quality scanner to replace an ambiguous item such as
`white_mug_0`. It excludes every take already represented in the assets
directory, returns at most one object per remaining take and one case per object
label, and by default avoids names containing `mug` or `stainless`. This
prevents a batch from silently becoming several objects from one scene or the
same object repeated across scenes. Preferred-duration and short-fallback target
cameras are reported separately; fragmented extraction remains eligible but is
audited explicitly:

```bash
DATA="$HOME/scratch/datasets/Ego-Exo4D-Relation-Test/extracted/work/yuqian_fu/Ego/data_segswap_test"
"$HOME/.conda/envs/sam3/bin/python" select_replacement_case.py \
  --data-root "$DATA" \
  --assets-root "$EASY6" \
  --exclude-object white_mug_0 \
  --exclude-take 0bacb5bb-591d-4756-a2cf-ed90793e65bb \
  --top-k 6 \
  --output "$WORK/replacement_candidates.json"
```

The six recommendations have six distinct `take_id` and `object_name` values.
Inspect their source overlays, then generate each exact case with
`generate_temporal_cross_view_assets.py --case-id <case_id>` rather than
selecting by object name across every take. Pass `--allow-existing-takes`,
`--allow-multiple-per-take`, or `--allow-duplicate-objects` only when that reuse
is intentional. If the viable candidate pool is too small, pass
`--no-default-avoid-terms` to admit mug/stainless labels from new takes while
keeping explicit object/take exclusions and the target-window viability gate.
For dataset-wide scans, run Python unbuffered (`python -u`) and capture both
stdout and stderr with `tee`; the selector reports progress every 25 cases.

Rebuild rendered outputs and the batch summary without SAM3:

```bash
PYTHON="$HOME/.conda/envs/sam3/bin/python"
"$PYTHON" qwen_temporal_runner.py \
  --assets-root "$EASY6" \
  --work-dir "$WORK" \
  --render-only
```

Verify final state:

```bash
for f in "$EASY6"/*/temporal_analysis_result.json; do
  jq -r '[.case_id, .schema_version, .status, .pipeline_status,
          (.best_segment.window_id // "NONE"),
          (.best_segment.duration_adjustment // "NONE"),
          (.best_segment.actual_sampled_frame_ratio // 0)] | @tsv' "$f"
done
```

Open the latest comparison image for a case at:

```text
$EASY6/<case_id>/analysis_outputs/selected_vs_rejected_region_comparison.png
```

## Reusing existing Qwen results

Schema-14 results are final temporal decisions. Schema-11 and older results
should be rerun through Qwen because they do not contain occurrence-centered
padding and still treat SAM3 as a required completion stage.
