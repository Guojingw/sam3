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
6. Recheck the strongest candidate windows with a chronological 3x3 summary
   spanning each complete window.
7. Deterministically reject any candidate window that:
   - is not fully inside a confirmed occurrence span;
   - is not a generated 20%-30% window;
   - supports the target in fewer than 60% of sampled frames;
   - confirms the target in fewer than 30% of sampled frames;
   - has more than four consecutive absent samples;
   - lacks evidence near either endpoint; or
   - fails Qwen's beginning/middle/end whole-window verification.
8. Select the highest whole-window score or emit `uncertain`.
9. Rewrite `temporal_analysis_result.json`, render the comparison image, and
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
