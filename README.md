# Longitudinal Review Station

## Quick Start with Makefile

Run all commands from the project root.

### 1. Prepare the raw data

Place the Cohort A dataset under `./data`:

```text
data/
├── inputsTr/
└── targetsTr/        # or outputsTr/
```

### 2. Install dependencies

```bash
make setup
```

### 3. Build the patient manifest and start the dashboard

```bash
make run
```

This command:

1. discovers patients from `./data`;
2. generates the scan and BL/FU pair manifests;
3. starts the V5 Streamlit dashboard on port `8501`.

Open the local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

### Useful Makefile commands

```bash
make run MAX_PATIENTS=5
```

Runs the dashboard with the first five discovered patients.

```bash
make run PATIENT_IDS=006f52e910,02522a2b27
```

Builds the manifest for the specified patients only. `PATIENT_IDS` takes priority over `MAX_PATIENTS`.

```bash
make run DATA="C:/path/to/Longitudinal_CT_v2"
```

Uses a dataset outside the default `./data` directory.

```bash
make run PORT=8502
```

Starts Streamlit on a different port.

```bash
make manifest
```

Rebuilds the patient and BL/FU pair manifests without starting Streamlit.

```bash
make dashboard
```

Starts the dashboard using the existing pair manifest.

```bash
make test
```

Runs the project test suite.

Press `Ctrl+C` in the terminal to stop the dashboard.

## Software Overview

The Longitudinal Review Station is a Streamlit application for reviewing and matching lesions between baseline (BL) and follow-up (FU) Cohort A scans.

The software provides the following functions:

- automatic discovery of Cohort A patients and BL/FU scan pairs;
- loading of CT, PET and lesion-mask data;
- rigid longitudinal registration with ITKElastix;
- synchronized review of original BL, aligned BL and original FU images;
- optional lesion-mask overlays;
- extraction of individual lesion components and quantitative features;
- automatic generation of a lesion-pair cost matrix;
- lesion correspondence calculation using NetworkX minimum-cost flow;
- visualization and export of patient-level matching results;
- optional comparison with reference tracking data when ground truth is available.

The main processing flow is:

```text
Raw Cohort A data
        ↓
Patient and BL/FU pair manifest
        ↓
Rigid BL-to-FU registration
        ↓
Aligned lesion extraction and features
        ↓
Pairwise cost matrix
        ↓
Minimum-cost-flow matching
        ↓
Dashboard review and exported results
```

## Project Inputs and Outputs

### Input data

By default, the application reads raw data from:

```text
./data
```

The pair manifest is generated at:

```text
outputs/cohort_a_subset/cohort_a_subset_pairs.csv
```

Paths written into the manifest are resolved relative to the selected data root. If the data root is changed through `DATA=...`, the Makefile passes the same location to the dashboard automatically.

### Generated outputs

Patient-specific artifacts are stored under:

```text
outputs/patients/<patient_id>/
```

Depending on the completed steps, this directory may contain registration results, aligned masks, lesion features, the cost matrix and final lesion matches. Existing valid artifacts are reused so that the same patient does not need to be processed again unnecessarily.

## Dashboard Usage

### 1. Select a patient

Use the patient selector in the sidebar. The available patients come from the generated pair manifest.

The V5 dashboard automatically uses the paths supplied by the Makefile. Under normal use, you do not need to enter separate manifest, output or cost-matrix paths.

### 2. Review the BL and FU scans

The alignment area presents three related views:

- **Original BL**: the baseline image in its native space;
- **Aligned BL**: the baseline image rigidly resampled into the FU space;
- **Original FU**: the follow-up image used as the registration reference.

Use the slice control or mouse wheel, where supported, to move through the scan. The displayed BL slice is mapped to the corresponding FU location so that the three views can be compared consistently.

### 3. Display lesion masks

Enable the mask option to overlay lesion regions on the images. The left view uses the native BL mask, the centre view uses the aligned BL mask, and the right view uses the FU mask.

Mask overlays are intended to help verify:

- whether lesions were extracted correctly;
- whether BL and FU anatomy is sufficiently aligned;
- whether a predicted lesion correspondence is visually plausible.

### 4. Run or reuse alignment

For a patient without processed outputs, start the alignment action shown in the dashboard. The application performs rigid BL-to-FU registration and saves the patient-specific result.

If valid registration outputs already exist, the dashboard reuses them. Use the rerun option only when the source data, registration settings or previous result has changed.

### 5. Generate lesion matching

Open the matching section below the alignment review and run matching for the selected patient.

The dashboard automatically:

1. loads or generates the aligned lesion features;
2. calculates pairwise distance and size-difference costs;
3. generates the cost matrix for the selected patient;
4. runs the NetworkX minimum-cost-flow solver;
5. decodes and displays the predicted lesion correspondences.

No pre-generated matrix path is required. Each matrix is associated with the selected patient and stored in that patient's output directory.

### 6. Interpret matching results

The result table links BL lesion IDs with FU lesion IDs and reports the matching information used by the algorithm. Lesion IDs are assigned during connected-component extraction and remain patient-specific.

Review the table together with the image overlays. A low mathematical cost does not by itself guarantee a clinically correct correspondence, especially when registration is poor, a lesion was not extracted, or several lesions are close together.

When split handling is enabled, one BL lesion may occupy more than one matching slot so that it can be associated with multiple FU lesions. Merge events remain distinguishable for later review and refinement.

### 7. Review evaluation results

When compatible ground-truth tracking data are available, the evaluation area can compare predictions with the reference correspondences.

- **Correct**: the predicted correspondence agrees with the reference.
- **Incorrect**: a correspondence was predicted, but it links the wrong lesions.
- **Missed**: a reference correspondence was not recovered. This can also occur when a ground-truth lesion was not extracted and therefore never entered the matching graph.

Use these metrics together with patient-level visual inspection. Registration and lesion-extraction failures should be investigated separately from matching-algorithm errors.

## Recommended Review Order

For each patient:

1. confirm that the BL and FU scans load successfully;
2. inspect the rigid alignment at several anatomical levels;
3. enable masks and check that expected lesions are present;
4. run lesion matching;
5. review the predicted BL/FU links visually;
6. inspect evaluation results if reference data are available;
7. record registration, extraction or matching issues separately.

## Troubleshooting

### `data root does not exist`

Confirm that the dataset is stored under `./data`, or provide its location explicitly:

```bash
make run DATA="C:/full/path/to/Longitudinal_CT_v2"
```

### `expected raw Cohort A data under .../inputsTr`

The selected `DATA` directory must contain the `inputsTr` folder. Do not point `DATA` directly to `inputsTr`.

### Missing Python packages

Run:

```bash
make setup
```

Then start the application again with `make run`.

### Rows with missing files

This message means that one or more CT, PET, mask or reference paths could not be found while building the manifest. Check the dataset directory structure and file naming before processing the affected patients.

### Port already in use

Choose another port:

```bash
make run PORT=8502
```

### Dashboard starts but contains no patients

Rebuild the manifest and confirm that the selected data root contains valid Cohort A cases:

```bash
make manifest DATA="C:/full/path/to/Longitudinal_CT_v2"
make dashboard DATA="C:/full/path/to/Longitudinal_CT_v2"
```

## Notes

- Run Make from the project root so that relative paths resolve correctly.
- Use forward slashes in Windows command-line paths when possible.
- Generated outputs are patient-specific and should remain under the configured patient output root.
- The dashboard is a research prototype and its predicted correspondences require visual validation.
