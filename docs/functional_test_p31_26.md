# P31-26 Functional Test - Patient Data Loading

## Purpose

Verify that a selected Cohort A patient can be loaded from the generated BL/FU pair manifest with the correct CT, PET and lesion-mask files associated with the correct timepoint.

## Test file

```text
tests/test_p31_26_patient_data_loading_functional.py
```

The test group has two layers:

1. **Deterministic synthetic Cohort A-style data** - fast tests where every path and expected image value is known in advance.
2. **Representative real Cohort A loading check** - enabled by the dedicated Makefile target after a small manifest is generated from the selected `DATA` root.

## Acceptance-criteria mapping

| Acceptance criterion | Automated evidence |
| --- | --- |
| Valid patient ID can be loaded successfully | Loads a selected patient with BL/FU CT, PET and lesion masks and checks patient/timepoint/scan IDs. |
| BL and FU imaging data are loaded from the correct paths | Compares every loaded `metadata.path` with the exact manifest path resolved against the Cohort A root. |
| CT, PET and lesion masks are associated with the correct timepoint | Each synthetic modality/timepoint has a distinct sentinel value, so BL/FU or CT/PET/mask swaps fail the test. |
| Missing or invalid files are reported with a clear error message | Checks blank manifest paths, missing files, unreadable NIfTI files and an unknown patient ID. |
| Test passes on representative Cohort A patients | Dedicated target generates a real Cohort A pair manifest and loads up to two complete representative patients. |
| Expected result: data ready for downstream processing | Checks 3-D arrays, consistent shapes, finite image data, physical spacing/orientation and valid volume metadata. |

## Recommended command

From the repository root, with the actual Cohort A directory:

```bash
make test-p31-26 DATA=./data/cohort_a MAX_PATIENTS=2
```

The target first runs the existing `manifest` step, then enables the representative real-data check and runs the P31-26 test file.

If Cohort A is stored directly under `./data` as described by the current README, use:

```bash
make test-p31-26 DATA=./data MAX_PATIENTS=2
```

## Full test suite behaviour

The normal `make test` command also discovers the P31-26 test file. Its deterministic tests always run. The real-data test is skipped unless the dedicated P31-26 target explicitly enables it, so the general suite does not require a large Cohort A dataset on every machine.

