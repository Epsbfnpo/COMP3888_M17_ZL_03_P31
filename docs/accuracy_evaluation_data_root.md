# Lesion Tracking Accuracy Evaluation

## Prepare the 11-Patient Manifest and Pair Files

The current baseline uses 11 Cohort A patients. Start directly from the Cohort A data root.

Replace `<DATA_ROOT>` with the local Cohort A dataset path:

```text
<DATA_ROOT>
```

### 1. Generate the Cohort A manifest and pair files from the data root

```powershell
python tools/prepare_cohort_a_subset.py `
  --root "<DATA_ROOT>" `
  --out-dir outputs/cohort_a_15 `
  --max-patients 15 `
  --path-mode relative-to-root
```

This generates the original 15-patient subset files, including:

```text
outputs/cohort_a_15/cohort_a_subset_manifest.csv
outputs/cohort_a_15/cohort_a_subset_pairs.csv
```

### 2. Filter the current 11 evaluable patients

The following 4 patients are temporarily excluded because of the known rigid-registration initialization bug:

```text
013d407166
03c6b06952
06138bc5af
069059b7ef
```

Create the 11-patient manifest:

```powershell
python -c "import pandas as pd; exclude={'013d407166','03c6b06952','06138bc5af','069059b7ef'}; df=pd.read_csv('outputs/cohort_a_15/cohort_a_subset_manifest.csv', dtype=str); out=df[~df['patient_id'].astype(str).isin(exclude)].copy(); out.to_csv('outputs/tracking_pipeline_11/selected_manifest_11.csv', index=False); print('Patients:', out['patient_id'].nunique()); print('Rows:', len(out))"
```

Create the 11-patient pair file:

```powershell
python -c "import pandas as pd; exclude={'013d407166','03c6b06952','06138bc5af','069059b7ef'}; df=pd.read_csv('outputs/cohort_a_15/cohort_a_subset_pairs.csv', dtype=str); out=df[~df['patient_id'].astype(str).isin(exclude)].copy(); out.to_csv('outputs/tracking_pipeline_11/selected_pairs_11.csv', index=False); print('Patients:', out['patient_id'].nunique()); print('Rows:', len(out))"
```

The resulting files are:

```text
outputs/tracking_pipeline_11/selected_manifest_11.csv
outputs/tracking_pipeline_11/selected_pairs_11.csv
```

The 11 selected patients are:

```text
006f52e910
01161aaa0b
02522a2b27
0266e33d3f
03b90eb112
045117b0e0
060ad92489
0640966322
06eb133bbf
06eb61b839
0777d5c17d
```

---

## Usage

Run the accuracy evaluation from the project root:

```powershell
python tools/run_tracking_evaluation_batch.py `
  --pairs outputs/tracking_pipeline_11/selected_pairs_11.csv `
  --features outputs/tracking_pipeline_11/aligned_lesion_features.csv `
  --data-root "<DATA_ROOT>" `
  --out-dir outputs/tracking_pipeline_11 `
  --matrix-dir outputs/tracking_pipeline_11/lesion_pair_cost_matrices `
  --expected-patients 11 `
  --max-bl-per-fu 3 `
  --max-map-distance-vox 30 `
  --reuse-matrices
```

This command reuses the existing cost matrices and evaluates the matching results against the Cohort A reference CSVs.

> Before running the evaluation, the `extract -> pair cost -> min-cost-flow match` pipeline should already have been completed for the selected patients.

---

## Purpose

The accuracy evaluation compares predicted lesion correspondences from the min-cost-flow matcher with the provided Cohort A ground truth.

The evaluation reports:

- correct matches
- incorrect matches
- missed matches
- tracking accuracy
- precision
- recall
- F1 score
- topology accuracy
- topology F1

The evaluation works across multiple Cohort A patients.

---

## Ground-Truth Mapping

The matcher uses temporary lesion IDs generated from connected components, while the Cohort A reference CSV uses expert lesion IDs.

Therefore, the evaluator first maps the expert ground-truth lesions to the temporary lesion IDs.

For lesion identity mapping:

```text
GT cog_bl
    -> native BL lesion centroid
    -> temporary BL lesion ID

GT cog_fu
    -> native FU lesion centroid
    -> temporary FU lesion ID
```

The Cohort A `cog_*` values are treated as voxel-index coordinates.

The evaluator also filters the reference CSV using the scan pair selected in the manifest:

```text
BL_00 -> img_id_bl = 0
FU_00 -> img_id_fu = 0
```

This is required because one Cohort A reference CSV may contain lesion correspondences from multiple BL/FU image pairs.

---

## Evaluated Patient Set

The current baseline evaluation uses **11 patients**.

Four patients were temporarily excluded because of a known rigid-registration initialization issue:

```text
013d407166
03c6b06952
06138bc5af
069059b7ef
```

These cases are tracked separately as a registration bug and are not included in the current accuracy result.

The 11 evaluated patients are:

```text
006f52e910
01161aaa0b
02522a2b27
0266e33d3f
03b90eb112
045117b0e0
060ad92489
0640966322
06eb133bbf
06eb61b839
0777d5c17d
```

---

## Current Results

All 11 selected patients completed ground-truth mapping and evaluation successfully.

| Metric | Result |
|---|---:|
| Patients evaluated | 11 / 11 |
| Patients failed | 0 / 11 |
| Correct matches | 142 |
| Incorrect matches | 15 |
| Missed matches | 14 |
| Tracking accuracy | **0.910 (91.0%)** |
| Precision | **0.904 (90.4%)** |
| Recall | **0.910 (91.0%)** |
| F1 score | **0.907 (90.7%)** |
| Topology accuracy | **0.885 (88.5%)** |
| Topology F1 | **0.882 (88.2%)** |

### Metric interpretation

Tracking accuracy is calculated from correctly recovered ground-truth links:

```text
Tracking Accuracy
= Correct Matches / (Correct Matches + Missed Matches)

= 142 / (142 + 14)
= 0.910
```

Precision is:

```text
Precision
= Correct Matches / (Correct Matches + Incorrect Matches)

= 142 / (142 + 15)
= 0.904
```

Recall is:

```text
Recall
= Correct Matches / (Correct Matches + Missed Matches)

= 142 / (142 + 14)
= 0.910
```

The current baseline therefore achieves approximately:

```text
Tracking Accuracy : 91.0%
Precision         : 90.4%
Recall            : 91.0%
F1                : 90.7%
Topology Accuracy : 88.5%
Topology F1       : 88.2%
```

---

## Output Files

The evaluation results are written to:

```text
outputs/tracking_pipeline_11/
```

Important files:

| File | Description |
|---|---|
| `lesion_matches.csv` | Final predicted lesion outcomes |
| `solver_lesion_matches.csv` | Results produced directly by the min-cost-flow matcher |
| `patient_routing.csv` | Shows whether each patient used min-cost-flow or deterministic handling |
| `lesion_tracking_summary.csv` | Overall and per-patient accuracy summary |
| `lesion_tracking_details.csv` | Detailed correct / incorrect / missed comparisons |
| `lesion_tracking_gt_mapping.csv` | Ground-truth expert lesion to temporary lesion ID mapping |
| `batch_status.csv` | Evaluation status for each patient |
| `lesion_flow_graphs/` | Saved flow-network edge tables |
| `lesion_pair_cost_matrices/` | BL x FU cost matrices |

For reporting the current baseline, the main file is:

```text
outputs/tracking_pipeline_11/lesion_tracking_summary.csv
```

For investigating individual errors, use:

```text
outputs/tracking_pipeline_11/lesion_tracking_details.csv
```

---

## Current Baseline Statement

> The current lesion-tracking baseline was evaluated on 11 Cohort A patients after excluding 4 cases affected by a known rigid-registration initialization issue. All 11 selected patients were successfully evaluated. The min-cost-flow matching pipeline achieved **91.0% tracking accuracy**, **90.4% precision**, **91.0% recall**, and **90.7% F1**. Topology classification achieved **88.5% accuracy** and **88.2% F1**.
