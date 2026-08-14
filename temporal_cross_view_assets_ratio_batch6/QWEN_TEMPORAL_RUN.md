# Qwen temporal selection and final SAM3 visualization

The pipeline has two strictly separated stages.

## Decision ownership

1. Qwen reads `metadata.json`, the first-person source mask, the source overlay,
   and the isolated original-RGB masked object.
2. Qwen searches the synchronized third-person timeline from the source frame in
   both directions, refines positive areas, and verifies identity on standalone
   crops.
3. Code scores dense sliding windows at every sampled-frame start for 20%, 25%,
   and 30% video lengths. This includes windows that cross old fixed boundaries.
4. Qwen and deterministic scoring select the final continuous window. The result
   is saved in both `best_segment` and immutable `qwen_temporal_selection`.
5. A brief occurrence may be accepted from one independently crop-verified
   frame when confidence is at least 0.90, at least two physical identity cues
   match, and no cue conflicts. Code pads the chosen interval to a legal 20%
   window.
6. If Qwen has no valid identity-supported window, the result stays `uncertain`.

SAM3 does not search time, rerank windows, reject a Qwen window, or change
`success` to `uncertain`.

## Final visualization

After a Qwen window is selected, SAM3 independently segments five representative
frames from that one window. It first uses the source-derived object identity as
a text prompt. A Qwen bbox is only a fallback when text produces no mask and is
never used to rank time.

Final output is schema 13:

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

Run Qwen selection first. The existing parallel Qwen submission script can be
used for all cases:

```bash
bash submit_qwen_temporal_parallel.sh "$EASY6" "$WORK"
qstat -u "$USER"
```

Check which cases are waiting only for final masks:

```bash
for f in "$EASY6"/*/temporal_analysis_result.json; do
  jq -r '[.case_id, .status, .pipeline_status,
          (.best_segment.window_id // "NONE")] | @tsv' "$f"
done
```

When Qwen evidence and crop-verification caches already exist, apply updated
window rules without loading Qwen or requesting a GPU:

```bash
"$HOME/.conda/envs/sam3/bin/python" qwen_temporal_runner.py \
  --assets-root "$EASY6" \
  --work-dir "$WORK" \
  --rescore-only
```

Submit SAM3 only after Qwen jobs finish:

```bash
bash submit_sam3_temporal_parallel.sh "$EASY6" "$WORK"
qstat -u "$USER"
```

Each selected case now runs only five independent SAM3 frame segmentations, not
multi-window video tracking. The job should therefore be much shorter than the
Qwen stage.

After the SAM3 jobs finish, rebuild all rendered outputs and the batch summary:

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
          (.final_segmentation.segmented_frame_count // 0)] | @tsv' "$f"
done
```

Open the latest comparison image for a case at:

```text
$EASY6/<case_id>/analysis_outputs/selected_vs_rejected_region_comparison.png
```

## Reusing existing Qwen results

Schema-11 results produced by the new Qwen code can go directly to the final
SAM3 stage. Old schema-12 results from the former SAM3 reranker should be rerun
through Qwen because their `best_segment` may have been changed or removed by
the old acceptance gate.
