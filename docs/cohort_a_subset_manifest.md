# Cohort A subset manifest

This module prepares a small, reproducible subset of Cohort A patients for pipeline
development. It scans a Cohort A-style folder with `inputsTr/` plus either
`targetsTr/` or `outputsTr/`,
selects a small number of patients, and writes CSV manifests that can be loaded
directly with pandas.

## Command

```powershell
python tools/prepare_cohort_a_subset.py `
  --root data/cohort_a `
  --out-dir outputs/cohort_a_subset `
  --max-patients 5 `
  --path-mode relative-to-root
```

To select exact patients:

```powershell
python tools/prepare_cohort_a_subset.py `
  --root data/cohort_a `
  --out-dir outputs/cohort_a_subset `
  --patient-ids 0a09c8844b,0aa1883c64
```

To copy the selected files into a portable subset folder:

```powershell
python tools/prepare_cohort_a_subset.py `
  --root data/cohort_a `
  --out-dir outputs/cohort_a_subset `
  --max-patients 5 `
  --copy-files `
  --path-mode relative-to-manifest
```

The default path mode is `relative-to-root`, which writes paths such as
`inputsTr/<patient>_BL_img_00.nii.gz`. This is suitable when each cloned
environment has the same Cohort A folder structure but may store it in a
different absolute location. Use `relative-to-manifest` when creating a copied
portable subset beside the manifest files.

## Outputs

`cohort_a_subset_manifest.csv` is a long-form scan manifest. Each row is one
patient timepoint/image ID.

Columns:

- `patient_id`: unique source patient identifier.
- `timepoint`: `BL` for baseline or `FU` for follow-up.
- `image_id`: Cohort A image index, for example `00`.
- `scan_id`: stable scan identifier used by downstream code.
- `ct_path`: CT image path, from `*_img_<id>.nii.gz`.
- `pet_path`: PET/SUV path if available.
- `lesion_mask_path`: lesion mask path, from `inputsTr/`, `targetsTr/`, or `outputsTr/`.
- `reference_csv_path`: expert correspondence CSV for the patient.
- `reference_json_path`: point-reference JSON for the scan if available.
- `missing_files`: semicolon-separated missing fields.
- `is_complete`: `True` only when CT, PET, lesion mask, CSV, and JSON are all present.
- `copied`: `True` when files were copied by `--copy-files`.

`cohort_a_subset_pairs.csv` is a wide-form pair manifest. Each row is one
baseline/follow-up patient pair, with separate BL and FU CT, PET, mask, and
reference paths.

`cohort_a_subset_summary.json` records selected patient IDs, row counts, and
missing-file counts for reproducibility.

## Loading

```python
import pandas as pd

scan_manifest = pd.read_csv("outputs/cohort_a_subset/cohort_a_subset_manifest.csv")
pair_manifest = pd.read_csv("outputs/cohort_a_subset/cohort_a_subset_pairs.csv")
```

## Note on the provided sample

The current two-patient Cohort A test data is CT-only. The manifest therefore
records CT and lesion-mask paths successfully, records the expert CSV/JSON files,
and flags `pet_path` as missing. This is expected for this sample and makes the
missing modality explicit instead of silently dropping the patient.
