# Qwen3.5-4B temporal batch runner

This runner uses `Qwen/Qwen3.5-4B`. The directory name `qwen35-4b` refers to
Qwen3.5 with 4B parameters; it does not mean a 3.5B model.

## Decision pipeline

This is a two-stage inference pipeline. Qwen proposes identity-verified dense
windows; SAM3 makes the final mask-based temporal decision.

1. Read each case's `metadata.json`, first-person largest-mask frame, binary
   mask, and overlay. Build an isolated RGB anchor so the overlay color cannot
   be mistaken for the object's color.
2. Use the synchronized source frame only as a weak search prior. Probe native
   third-person frames near it first, then expand toward both timeline edges.
3. Require Qwen detections to contain a normalized bbox and at least two
   identity-specific physical cues. Color-only matches and scene activity do
   not count.
4. Build per-camera occurrence evidence. A short true occurrence is allowed:
   the legal window may contain surrounding context as long as the occurrence
   itself is captured.
5. Enumerate every possible sampled-frame start for 20%, 25%, and 30% window
   lengths. These are dense overlapping windows, not three fixed bins, so a
   globally best window may cross any old segment boundary.
6. Score all dense windows cheaply in deterministic CPU code, apply temporal
   NMS, and send at most eight diverse Top-K windows back to Qwen for strict
   full-frame and enlarged-crop identity verification.
7. Qwen presentation, completeness, and scale values remain diagnostics only.
   They cannot choose the final window. A separate shortlist score uses verified
   identity, occurrence capture, real probe support, and the weak source prior.
8. Pass up to five identity-verified windows to SAM3. Seed each track with a
   verified Qwen bbox, propagate a binary mask across the complete candidate
   window, and derive every displayed bbox directly from that mask. When SAM3
   returns multiple object IDs, select the mask with maximum IoU to the seed
   box rather than the first ID. If that track fails, retry up to two additional
   independently verified Qwen anchors from the same window. Explicit box-seed
   tracking disables SAM3's 15-frame semantic-detector hot-start because a
   legal 20% candidate can contain fewer than 15 sampled frames.
9. Rank SAM3 tracks by captured mask evidence, full-window coverage, longest
   continuous run, target scale, area stability, bbox motion stability, and IoU
   against at least two independently verified Qwen anchors.
10. Select the highest-scoring legal continuous window only when it has two
    anchor matches and beats the runner-up by the configured margin. Otherwise
    emit `uncertain` instead of forcing a choice.
11. Write schema-12 `temporal_analysis_result.json`, rerender the enlarged
    source/selected/alternative comparison, save representative masks for both
    accepted and uncertain candidates, and update the batch summary.

The synchronized source frame is a search origin, not a hard target frame. The
final segment is always continuous and occupies 20%-30% of its camera timeline.

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

After all Qwen jobs disappear from `qstat`, verify that each successful case is
schema 11 and is waiting for SAM3:

```bash
for f in "$EASY6"/*/temporal_analysis_result.json; do
  jq -r '[.case_id, .schema_version, .status, .pipeline_status,
          (.sam3_rerank_candidates | length)] | @tsv' "$f"
done
```

Then submit the SAM3 mask reranking stage. It can also run one case per PBS job:

```bash
bash submit_sam3_temporal_parallel.sh "$EASY6" "$WORK"
qstat -u "$USER"
```

Cases with no valid Qwen candidate are already finalized as schema-12
`uncertain`. The submitter skips them, so they do not consume another GPU
allocation.

The `g1` queue may limit how many jobs one user can run concurrently. Jobs in
state `Q` remain queued and start automatically as running jobs finish.

After all SAM3 jobs disappear, rebuild the final six-case summary and rerender
without loading Qwen or requesting a GPU:

```bash
PYTHON="$HOME/.conda/envs/sam3/bin/python"

"$PYTHON" qwen_temporal_runner.py \
  --assets-root "$EASY6" \
  --work-dir "$WORK" \
  --render-only
```

The final result must be schema 12. A selected result uses
`pipeline_status=complete`; a case with no valid Qwen candidate uses
`pipeline_status=complete_no_sam3_candidates`. Schema 11 is only the Qwen
candidate stage and must not be treated as the final answer:

```bash
for f in "$EASY6"/*/temporal_analysis_result.json; do
  jq -r '[.case_id, .schema_version, .status, .pipeline_status,
          (.best_segment.window_id // "NONE")] | @tsv' "$f"
done
```

Parallel execution reduces elapsed wall-clock time but consumes approximately
the same total GPU-hours and remains subject to the queue concurrency limit.
Use one work directory for one run. Earlier low-resolution caches must not be
mixed with `temporal_easy6_native_tiles_v3`.

The frame evidence for the command above is saved under:

```text
/scratch/users/ntu/gwang016/qwen35-4b/temporal_easy6_native_tiles_v3/
```

Rerunning the Qwen PBS script resumes from cached completed frames. Dense window
enumeration itself runs on CPU; only newly selected strict-verification windows
cause additional Qwen calls. Do not use `--force` for an ordinary resume.

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

After schema 12 has completed and its rendered outputs have been checked, these
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
