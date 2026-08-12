# Qwen3.5-4B temporal batch runner

This runner uses `Qwen/Qwen3.5-4B`. The directory name `qwen35-4b` refers to
Qwen3.5 with 4B parameters; it does not mean a 3.5B model.

## Decision pipeline

1. Read each case's `metadata.json`, `source_best_frame.jpg`, and
   `source_best_mask.png`.
2. Create an isolated source RGB anchor from the binary mask. Overlay colors
   are explicitly excluded from identity inference.
3. Extract each unique third-person sampled frame from the generated temporal
   contact sheets and build previous/query/next context strips.
4. Ask Qwen for `confirmed`, `possible`, or `absent` evidence on every query
   frame. Presence and bounding-box validity are evaluated independently.
5. Build independent occurrence spans for each case.
6. Use the largest source-mask frame as a weak same-take timing prior, then
   recheck the strongest candidate windows with a chronological 3x3 summary.
7. Prefer a generated 20% continuous window that captures the most evidence
   from a localized occurrence. The target may be absent in surrounding window
   context when its true occurrence is shorter than 20% of the video.
8. Deterministically reject a window unless it captures at least two supported
   localized samples, including one confirmed sample, and Qwen verifies that
   the summary contains the identity-matching occurrence. A one-frame ambiguous
   glimpse is insufficient.
9. Select the strongest verified occurrence-containing window or emit
   `uncertain`.
10. Rewrite `temporal_analysis_result.json`, render the comparison image, and
   update `batch_temporal_analysis_summary.json`.

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
WORK="/scratch/users/ntu/$USER/qwen35-4b/temporal_easy6_parallel_v1"

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

The frame evidence is saved under:

```text
/scratch/users/ntu/gwang016/qwen35-4b/temporal_batch6_work_v2/
```

Rerunning the PBS script resumes from cached completed frames. To discard the
cache for one case, pass `--force --case sugar_container` in an interactive
run. Do not use `--force` for ordinary resume.

Render again without loading Qwen:

```bash
source "$HOME/scratch/qwen35-4b/env/bin/activate"
cd "$HOME/worldmodel/sam3/temporal_cross_view_assets_ratio_batch6"
python qwen_temporal_runner.py \
  --assets-root "$PWD" \
  --work-dir "/scratch/users/ntu/gwang016/qwen35-4b/temporal_batch6_work_v2" \
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
qsub -v ASSETS="$EASY6",WORK_DIR="/scratch/users/ntu/$USER/qwen35-4b/temporal_easy6_work" \
  run_qwen_temporal_batch.pbs
```
