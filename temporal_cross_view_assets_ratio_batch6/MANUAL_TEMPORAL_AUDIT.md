# Manual temporal audit before GPU inference

Use this workflow before submitting Qwen. It validates the source mask, checks
whether the same object is visible in the third-person timeline, and creates an
expected 20% window without loading model weights or requesting a GPU.

## 1. Prepare audit material on the login node

```bash
cd "$HOME/worldmodel/sam3/temporal_cross_view_assets_ratio_batch6"

PYTHON="$HOME/.conda/envs/sam3/bin/python"
EASY6="/scratch/users/ntu/$USER/temporal_cross_view_assets_ratio_easy6"
AUDIT="/scratch/users/ntu/$USER/manual_temporal_audit_easy6"

"$PYTHON" prepare_manual_temporal_audit.py \
  --assets-root "$EASY6" \
  --output-dir "$AUDIT"
```

This is CPU-only. Each case receives:

- `manual_audit.jpg`: source mask, 12 full-timeline samples per camera, and 9
  samples near the largest source-mask frame;
- `manual_temporal_ground_truth.json`: editable annotation template.

The audit root also contains `index.html` and `manual_audit_index.json`.

## 2. Inspect the audit images

Open the files through an NSCC remote file browser, or download them to the Mac:

```bash
rsync -avP \
  gwang016@asp2a-login-ntu02.nscc.sg:/scratch/users/ntu/gwang016/manual_temporal_audit_easy6/ \
  "$HOME/Desktop/manual_temporal_audit_easy6/"
```

For each case, answer in this order:

1. Does the source mask cover the intended object rather than a neighboring
   object or background?
2. What is the object's identity based on masked RGB pixels? Ignore overlay
   color.
3. Is the identity-matching object present in any third-person camera?
4. If present, what are the approximate first and last visible sampled frame
   IDs? Start with the full timeline row, then refine using the near-source row
   or existing contact sheets.
5. If the object cannot be distinguished from a confuser, mark it `uncertain`.

## 3. Fill each JSON template

Example for a short occurrence near the end:

```json
{
  "schema_version": 1,
  "case_id": "example__yellow_object_0",
  "review_status": "complete",
  "source_mask_correct": true,
  "source_object_identity": "yellow tea strainer",
  "source_best_frame": 3870,
  "target_presence": "present",
  "target_cam": "cam01",
  "first_visible_frame": 4020,
  "last_visible_frame": 4350,
  "identity_confidence": "high",
  "expected_status": "pending",
  "expected_window": null,
  "notes": "Yellow handled strainer visible near the pot at the end.",
  "available_cameras": {}
}
```

Allowed judgments:

- `source_mask_correct=false`: invalid input case; do not run inference;
- `target_presence=present`: supply camera and first/last frame;
- `target_presence=absent`: expected result is `uncertain`;
- `target_presence=uncertain`: identity cannot be verified; expected result is
  `uncertain`.

Keep the generated `available_cameras` object. It documents which frames were
shown during review.

## 4. Compute expected windows

After all six templates are filled:

```bash
"$PYTHON" prepare_manual_temporal_audit.py \
  --assets-root "$EASY6" \
  --output-dir "$AUDIT" \
  --finalize
```

The script validates the annotations and writes:

```text
$AUDIT/manual_ground_truth_summary.json
```

For a present object, it chooses the generated 20% window with the largest
overlap with the manually recorded occurrence. It does not call Qwen.

```bash
jq -r '
  .cases[] |
  [
    .case_id,
    .expected_status,
    (.expected_window.window_id // "NONE"),
    (.expected_window.start_frame // "NONE"),
    (.expected_window.end_frame // "NONE")
  ] | @tsv
' "$AUDIT/manual_ground_truth_summary.json"
```

Do not submit a GPU job unless `error_count` is zero:

```bash
jq '{completed_count, error_count, errors}' \
  "$AUDIT/manual_ground_truth_summary.json"
```

## 5. Run one easy case before the batch

Start with CPR dummy. Use a fresh work directory:

```bash
WORK="/scratch/users/ntu/$USER/qwen35-4b/temporal_easy6_smoke"

qsub -I \
  -P personal-gwang016 \
  -q normal \
  -l select=1:ngpus=1 \
  -l walltime=01:00:00

cd "$HOME/worldmodel/sam3/temporal_cross_view_assets_ratio_batch6"

"$PYTHON" qwen_temporal_runner.py \
  --assets-root "$EASY6" \
  --work-dir "$WORK" \
  --model "/scratch/users/ntu/$USER/qwen35-4b/model" \
  --case CPR_dummy
```

Compare its selected window with the manual JSON before submitting all six.
If the easy case fails identity or occurrence localization, stop and inspect
the evidence instead of spending another full batch job.
