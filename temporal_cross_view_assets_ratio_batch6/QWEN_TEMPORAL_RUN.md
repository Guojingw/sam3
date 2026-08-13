# Qwen3.5-4B temporal batch runner

This runner uses `Qwen/Qwen3.5-4B`. The directory name `qwen35-4b` refers to
Qwen3.5 with 4B parameters; it does not mean a 3.5B model.

## Decision pipeline

1. Read each case's `metadata.json`, `source_best_frame.jpg`, and
   `source_best_mask.png`.
2. Create an isolated source RGB anchor from the binary mask. Overlay colors
   are explicitly excluded from identity inference.
3. Resolve each unique third-person sample back to its native image under the
   take directory. Use a contact-sheet crop only when the source dataset is not
   available, and record the input source and resolution in the result JSON.
4. Use the largest first-person mask frame as the synchronized search origin.
   Probe densely around that frame, then expand exponentially toward the video
   boundaries rather than uniformly evaluating every frame.
5. Give Qwen the query frame as a standalone image and the three-frame strip
   only as context. Bounding boxes are normalized to the standalone query.
6. Rank genuine positive scouts by identity, visibility, localization quality,
   and source-time proximity. Expand both directions around at most six
   non-overlapping seeds per camera. Intermediate frames receive support only
   when bracketed by two real, spatially consistent localized detections.
7. Build independent occurrence spans for each case.
8. Divide every target-camera timeline into equal temporal bins and verify one
   20% challenger from every bin. This prevents a source-near local optimum
   from hiding a visually stronger segment elsewhere in the video.
9. Recheck each challenger using five time-distributed standalone full frames
   plus up to three strong scout frames. Do not use a compressed 3x3 contact
   sheet as a hard decision input. Qwen returns one presence and bbox result per
   standalone frame.
10. Enlarge every proposed bbox into a candidate panel containing both local
   context and the tight crop. Recompare it with the source-mask RGB anchor.
   Require two visible identity-specific physical cues, no conflicting cue, and
   reject crops too small or blurred to verify. Deterministic code derives the
   window-level decision only after this second pass. Score object completeness,
   bbox tightness, and object fill inside the box; loose background boxes do not
   count as verified localization.
11. If strict full-frame crop verification finds fewer than two matches, run a
   bounded 3x3 overlapping-tile search on at most two frames from each of the
   three highest-ranked windows. Run a separate tight-box localization call on
   the native tile, map that box back to full-frame coordinates, then require a
   separate strict crop-verification call.
12. Prefer a generated 20% continuous window that captures the most evidence
   from a localized occurrence. The target may be absent in surrounding window
   context when its true occurrence is shorter than 20%; this is the equivalent
   of padding the shorter side until the legal duration is reached.
13. Treat two independently verified frames as the acceptance gate, then rank
   legal windows by verification coverage (10%), identity (20%), occurrence
   capture (15%), presentation quality (20%), visible target scale (20%), box
   stability (5%), real-probe support (5%), and source-time proximity (5%).
14. Deterministically reject a window unless it captures at least two supported
   localized scout samples and at least two enlarged candidate crops pass strict
   cross-view identity verification. Color similarity and a positive claim
   without localization are insufficient.
15. Compare the winner with the best non-overlapping, equal-duration global
   challenger. Select the strongest verified continuous window or emit
   `uncertain`.
16. Rewrite `temporal_analysis_result.json`, render the comparison image, and
   update `batch_temporal_analysis_summary.json`.

The synchronized source frame is a search origin, not a hard target frame. If
the visually strongest third-person occurrence is earlier or later, it wins.

## NSCC commands

From the login node:

```bash
cd "$HOME/worldmodel/sam3"
git fetch origin agent/add-temporal-batch6
git checkout agent/add-temporal-batch6
git pull --ff-only origin agent/add-temporal-batch6

cd temporal_cross_view_assets_ratio_batch6
qsub run_qwen_temporal_batch.pbs
```

Monitor:

```bash
qstat -u gwang016
tail -f qwen4b_temporal_batch6.o*
```

## Parallel one-GPU-per-case run

Do not request multiple GPUs for one runner process. Submit one independent
one-GPU job per case instead:

```bash
EASY6="/scratch/users/ntu/$USER/temporal_cross_view_assets_ratio_easy6"
WORK="/scratch/users/ntu/$USER/qwen35-4b/temporal_easy6_native_tiles_v3"

cd "$HOME/worldmodel/sam3/temporal_cross_view_assets_ratio_batch6"
bash submit_qwen_temporal_parallel.sh "$EASY6" "$WORK"
qstat -u "$USER"
```

Each job writes only its own case outputs, cache directory, and summary under
`$WORK/job_summaries`. Do not run a serial job against the same work directory
at the same time.

After all jobs disappear from `qstat`, rebuild the complete six-case summary
and render outputs without loading Qwen or requesting a GPU:

```bash
PYTHON="$HOME/.conda/envs/sam3/bin/python"

"$PYTHON" qwen_temporal_runner.py \
  --assets-root "$EASY6" \
  --work-dir "$WORK" \
  --render-only
```

Parallel execution reduces elapsed wall-clock time but consumes approximately
the same total GPU-hours and is subject to project and queue concurrency limits.
Use the new work directory shown above. Earlier frame caches used a different
bbox coordinate system and window-verification schema and must not be mixed
with this run.

Result schema 10 keeps schema-9 native-frame scouts but invalidates old window
verifications. Reusing `temporal_easy6_native_tiles_v3` without `--force`
recomputes only candidate-window localization/verification and final ranking.
Result schema 9 deliberately invalidated old low-resolution frame evidence.
Use the new work directory above instead of mixing schema 8 contact-sheet crops
with native-frame evidence. Tile rescue is bounded and runs only after ordinary
full-frame localization and strict crop verification fail.

The frame evidence for the command above is saved under:

```text
/scratch/users/ntu/gwang016/qwen35-4b/temporal_easy6_native_tiles_v3/
```

Rerunning the PBS script resumes from cached completed frames. To discard the
cache for one case, pass `--force --case sugar_container` in an interactive
run. Do not use `--force` for ordinary resume.

Render again without loading Qwen or requesting a GPU:

```bash
cd "$HOME/worldmodel/sam3/temporal_cross_view_assets_ratio_batch6"
"$HOME/.conda/envs/sam3/bin/python" qwen_temporal_runner.py \
  --assets-root "$EASY6" \
  --work-dir "$WORK" \
  --render-only
```

## Generate an easier comparison batch

The temporal asset generator can create camera-relative windows whose lengths
are 20%, 25%, and 30% of each complete target camera. For example:

```bash
cd "$HOME/worldmodel/sam3"
DATA="/scratch/users/ntu/$USER/datasets/Ego-Exo4D-Relation-Test/extracted/work/yuqian_fu/Ego/data_segswap_test"
EASY6="/scratch/users/ntu/$USER/temporal_cross_view_assets_ratio_easy6"

"$HOME/.conda/envs/sam3/bin/python" generate_temporal_cross_view_assets.py \
  --data-root "$DATA" \
  --output-root "$EASY6" \
  --case-id 0a22a1c1-844c-4f62-8eeb-f16eee62357f__CPR_dummy \
  --case-id 0a3868ef-fdba-4aba-bc02-5028d1ed26f4__bicycle_inner_tube_0 \
  --case-id 0b82763e-b9ee-40e5-8dd5-b8da7e862662__ketchup_bottle_0 \
  --case-id 0bacb5bb-591d-4756-a2cf-ed90793e65bb__white_mug_0 \
  --case-id 0099226c-9bec-44aa-ba43-2b90eb7b8379__sugar_container_0 \
  --case-id 0099226c-9bec-44aa-ba43-2b90eb7b8379__yellow_tea_strainer_0 \
  --window-ratios 0.20 0.25 0.30 \
  --window-stride-ratio 0.05 \
  --sheet-columns 6 \
  --cell-width 480 \
  --cell-height 330 \
  --overwrite
```

Verify that exactly six cases and ratio windows were generated before using a
GPU:

```bash
find "$EASY6" -mindepth 1 -maxdepth 1 -type d | sort
for f in "$EASY6"/*/temporal_window_index.json; do
  jq -e '.windowing.mode == "camera_relative_ratio" and
         ([.windowing.window_ratios[]] == [0.2, 0.25, 0.3])' "$f"
done
```

Submit the same runner with isolated assets and cache paths:

```bash
cd "$HOME/worldmodel/sam3/temporal_cross_view_assets_ratio_batch6"
qsub -v ASSETS="$EASY6",WORK_DIR="/scratch/users/ntu/$USER/qwen35-4b/temporal_easy6_native_tiles_v3" \
  run_qwen_temporal_batch.pbs
```

## Safe scratch cleanup

First confirm that no job is running, then inspect sizes. Do not remove the
dataset, active model, Conda environment, repository, current assets, or the
new schema-9 work directory.

```bash
qstat -u "$USER"
du -sh /scratch/users/ntu/$USER/qwen35-4b/* \
  /scratch/users/ntu/$USER/temporal_* \
  /scratch/users/ntu/$USER/*prompt_assets* 2>/dev/null | sort -h
```

After schema 9 has completed and its rendered outputs have been checked, these
names are historical candidates for quarantine rather than immediate deletion:

```text
qwen35-4b/sugar_frame_crops
qwen35-4b/temporal_batch6_work
qwen35-4b/temporal_batch6_work_fresh
qwen35-4b/temporal_batch6_work_v2
qwen35-4b/temporal_easy6_work
qwen35-4b/temporal_easy6_source_centered_v2
temporal_cross_view_assets
temporal_cross_view_assets_ratio
temporal_cross_view_assets_ratio_batch6
batch_cross_view_prompt_assets
cross_view_prompt_assets_00a6dd13
sam3_test_same_object
```

Keep `datasets`, `qwen35-4b/model`, `$HOME/.conda/envs/sam3`,
`$HOME/worldmodel/sam3`, `temporal_cross_view_assets_ratio_easy6`, and
`qwen35-4b/temporal_easy6_native_tiles_v3`. Treat `hf_cache` and
`qwen35-4b/hf-cache` as undecided until model symlinks and their sizes have been
checked; deleting a cache still referenced by the model can break loading.

The main comparison is rendered at 3200x1180 with minimal labels. Each frame
shows only its frame number and `verified`/`unverified`. Both a maximum-quality
JPEG and a lossless PNG are written under each case's `analysis_outputs`.
