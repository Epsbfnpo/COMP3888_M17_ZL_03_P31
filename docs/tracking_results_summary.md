#Summarise Tracking Results per Patient


This iteration adds a patient-level summary layer on top of the tracking pipeline that already exists in the repository.

It does **not** rerun registration, feature extraction, pair-cost generation, min-cost-flow matching, or ground-truth evaluation.

it combines:

```text
lesion_matches.csv
        +
optional P31-18 lesion_tracking_summary.csv
        ↓
per-patient tracking summary
        ↓
Dashboard summary cards
        +
optional tested-subset summary
        +
tracking_patient_summary.csv
```


## Files changed

### Added

```text
src/tracking_summary.py
tools/summarise_tracking_results.py
tests/test_tracking_summary.py
docs/tracking_results_summary.md
```

### Updated

```text
tools/align_longitudinal_patient_v4_mapped.py
```


# 1. Summary definitions

## BL lesions

Number of unique non-empty `bl_lesion_id` values for the patient.

## FU lesions

Number of unique non-empty `fu_lesion_id` values for the patient.

## Matched lesions

`matched_lesions` means:

> the number of unique BL lesions that have a FU correspondence.

This includes ordinary `MATCHED` rows and BL lesions participating in `MERGING`.

For example:

```text
BL_1 -> FU_1   MERGING
BL_2 -> FU_1   MERGING
```

is counted as:

```text
matched_lesions    = 2
matched_fu_lesions = 1
matched_links      = 2
```

This prevents a many-to-one merge from being incorrectly described as two different FU lesions.

## Appearing lesions

Unique FU lesions with:

```text
match_type = NEW
```

These have no BL correspondence.

## Disappearing lesions

Unique BL lesions with:

```text
match_type = DISAPPEARING
```

These have no FU correspondence.

## Unmatched events

The dashboard uses:

```text
unmatched_lesions = appearing_lesions + disappearing_lesions
```

This count spans two timepoints, so the UI labels it **Unmatched events** to avoid implying that BL and FU unmatched lesions are the same biological object.

# 2. Ground-truth metrics


When available, it will loads the P31-18 output:

```text
lesion_tracking_summary.csv
```

and uses its existing fields, including:

```text
correct_matches
incorrect_matches
missed_matches
tracking_accuracy
precision
recall
f1
topology_accuracy
```

The current definition of `tracking_accuracy` is preserved:

```text
correct expert-confirmed events / ground-truth events
```


If the evaluation file is missing, or the selected patient's evaluation failed, the structural lesion summary still works and ground-truth metrics are shown as `N/A`.

# 3. Dashboard

Launch the existing dashboard:

```powershell
python -m streamlit run tools/align_longitudinal_patient_v4_mapped.py
```

The sidebar now contains an additional optional input:

```text
Ground-truth evaluation summary CSV
```

This should point to the P31-18 output `lesion_tracking_summary.csv` from the same tracking/evaluation run as the currently selected `lesion_matches.csv`.

The **Lesion correspondence** section now starts with a **Tracking summary**.

For the current patient it shows:

```text
BL lesions
FU lesions
Matched lesions
Unmatched events
Appearing (NEW)
Disappearing
Correct matches       (when GT is available)
Incorrect matches     (when GT is available)
Tracking accuracy     (when GT is available)
```

The detailed match table, pair-cost information, ground-truth event inspection, and image-linked lesion correspondence remain below the summary.

---

# 4. Tested-subset / Cohort A overall summary

The dashboard also includes an expander:

```text
Tested Cohort A subset summary
```

It builds one row per patient from the combined `lesion_matches.csv` and appends an `ALL` row.

Structural counts in the `ALL` row are sums across patients.

When the P31-18 evaluation summary contains its own `ALL` row, P31-22 uses that row for the overall accuracy rather than averaging the per-patient percentages.

This is important because:

```text
average(patient accuracy)
```

is not generally equal to:

```text
total correct events / total ground-truth events
```

If an `ALL` row is not present, the summary can fall back to aggregating successful per-patient evaluation rows.

---

# 5. Export a summary CSV

P31-22 also provides a standalone CLI so summary results can be used outside Streamlit.

Example using the current tracking pipeline output:

```powershell
python tools/summarise_tracking_results.py `
  --matches "outputs\tracking_pipeline_11\lesion_matches.csv" `
  --evaluation-summary "outputs\tracking_pipeline_11\lesion_tracking_summary.csv" `
  --out "outputs\tracking_pipeline_11\tracking_patient_summary.csv"
```

If ground truth is not available yet:

```powershell
python tools/summarise_tracking_results.py `
  --matches "outputs\tracking_pipeline_11\lesion_matches.csv" `
  --out "outputs\tracking_pipeline_11\tracking_patient_summary.csv"
```

In this case the structural counts are still written, while the accuracy fields are blank / unavailable.

To omit the `ALL` row:

```powershell
python tools/summarise_tracking_results.py `
  --matches "outputs\tracking_pipeline_11\lesion_matches.csv" `
  --no-overall `
  --out "outputs\tracking_pipeline_11\tracking_patient_summary.csv"
```


# 6. Output columns

The exported CSV contains:

```text
patient_id
bl_lesions
fu_lesions
matched_lesions
matched_fu_lesions
matched_links
unmatched_lesions
appearing_lesions
disappearing_lesions
merging_links
merged_fu_lesions
ground_truth_available
ground_truth_links
correct_matches
incorrect_matches
missed_matches
tracking_accuracy
precision
recall
f1
topology_accuracy
evaluation_status
evaluation_error
```

The extra merge and evaluation fields are included so the summary remains auditable instead of only showing the minimum Jira metrics.



# 7. Example

Suppose a patient has:

```text
BL_1 -> FU_1   MERGING
BL_2 -> FU_1   MERGING
BL_3 -> —      DISAPPEARING
—    -> FU_2   NEW
```

The structural summary is:

```text
BL lesions          3
FU lesions          2
Matched lesions     2
Matched FU lesions  1
Matched links       2
Appearing           1
Disappearing        1
Unmatched events    2
Merging links       2
Merged FU lesions   1
```

If P31-18 reports:

```text
correct_matches     3
incorrect_matches   1
tracking_accuracy   0.60
```

then the dashboard shows:

```text
Correct matches     3
Incorrect matches   1
Tracking accuracy   60.0%
```



# 8. Error handling

The summary layer explicitly reports:

- missing `lesion_matches.csv`;
- missing required matcher columns;
- blank patient IDs or match types;
- malformed evaluation summary files;
- duplicate patient rows in the evaluation summary;
- patient IDs with no matching result rows;
- failed ground-truth evaluations.

Ground-truth files are optional. Their absence does not prevent structural BL/FU tracking statistics from being displayed.


# 9. Tests

Run:

```powershell
python -m pytest -q tests/test_tracking_summary.py
```

The tests cover:

- BL/FU lesion counts;
- ordinary and many-to-one matched lesion counts;
- appearing and disappearing counts;
- unmatched-event count;
- merge handling;
- correct / incorrect / missed GT metrics;
- tracking accuracy;
- `ALL` tested-subset aggregation;
- fallback aggregation when an evaluation `ALL` row is absent;
- failed evaluation handling;
- preservation of patient IDs with leading zeroes;
- clear errors for missing patients.

