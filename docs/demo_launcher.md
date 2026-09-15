# Demo launcher (`run_demo.ps1`)

This launcher provides a Windows-friendly entry point for the current P31 pipeline and Streamlit dashboard. It was added because the repository is composed of multiple CLI tools whose input/output paths otherwise have to be entered manually.

## What it does

`run_demo.ps1` can run either Cohort A or Cohort B. In full mode it:

1. detects or asks for the cohort and data root;
2. checks for non-ASCII Windows paths before ITK/Elastix is started;
3. creates a small patient manifest (default: 2 patients);
4. runs rigid registration, aligned lesion-feature extraction, pair-cost generation, and min-cost-flow matching;
5. for Cohort A, attempts expert ground-truth evaluation (non-fatal if a selected case cannot be evaluated);
6. generates the per-patient tracking summary;
7. exports environment variables used to auto-fill the Streamlit sidebar;
8. launches the dashboard.

The pipeline still uses the legacy filename `selected_pairs_11.csv` internally. The filename is retained for compatibility and does **not** imply that the launcher selected 11 patients.

## Simplest use

From the repository root in PowerShell:

```powershell
.\run_demo.ps1
```

Alternatively, double-click `run_demo.bat`; it invokes PowerShell with a process-local execution-policy bypass.

If only `data/cohort_a` or only `data/cohort_b` exists, the launcher selects that cohort automatically. If both exist, it asks which cohort to use. The default patient count is 2.

## Explicit Cohort A two-patient demo

```powershell
.\run_demo.ps1 -Cohort A -Patients 2
```

Cohort A additionally attempts the expert-reference evaluation, so correct/incorrect match counts and tracking accuracy can appear in the dashboard when ground truth is available.

To skip expert evaluation:

```powershell
.\run_demo.ps1 -Cohort A -Patients 2 -SkipEvaluation
```

## Explicit Cohort B two-patient demo

```powershell
.\run_demo.ps1 -Cohort B -Patients 2
```

Cohort B skips the Cohort A expert-reference evaluation. Tracking counts and matching visualisation remain available; ground-truth accuracy fields show N/A.

## Select exact patients

```powershell
.\run_demo.ps1 -Cohort A -PatientIds "patientA,patientB"
```

`-PatientIds` overrides the patient-count selection used by the manifest preparation tool.

## Re-open the dashboard without rerunning the pipeline

For the same cohort/patient-count output tag:

```powershell
.\run_demo.ps1 -Cohort A -Patients 2 -Mode dashboard
```

This is useful during presentation practice because registration and matching do not need to be rerun each time.

## Run processing without opening Streamlit

```powershell
.\run_demo.ps1 -Cohort A -Patients 2 -NoDashboard
```

## Output layout

A two-patient Cohort A launch uses:

```text
outputs/
  demo_a2/
    manifest/
      cohort_a_subset_pairs.csv
      ...
    tracking/
      selected_pairs_11.csv
      aligned_lesion_features.csv
      lesion_pair_costs.csv
      lesion_pair_cost_matrices/
      lesion_matches.csv
      lesion_tracking_summary.csv        # when Cohort A GT evaluation is available
      lesion_tracking_details.csv        # when Cohort A GT evaluation is available
      tracking_patient_summary.csv
```

Rigid registration artefacts continue to use the shared `outputs/registration/<patient_id>/` directory so valid previous registrations can be reused.

## Dashboard auto-fill

The launcher passes these values to the Streamlit process through environment variables:

- pair manifest;
- data root;
- registration output root;
- matching results CSV;
- aligned lesion features CSV;
- cost matrix directory;
- pair-cost details CSV;
- optional ground-truth evaluation details and summary CSVs;
- dataset label (Cohort A or Cohort B).

The dashboard reads these variables as sidebar defaults. The tested-subset heading is now dataset-aware instead of being hard-coded to Cohort A.

## Windows path warning

The launcher refuses to start if the repository or dataset root contains non-ASCII characters. This prevents the ITK/Elastix `No ImageIO is registered to handle the given file` failure previously observed with Chinese directory names on Windows.

### Viewer render speed

The launcher sets `P31_VIEWER_MAX_SIDE=560` by default. This caps the longest side of the three interactive CT/PET panels and keeps slice-slider interaction responsive while preserving physical voxel aspect ratio.

For a higher-resolution screenshot session, start the launcher with for example:

```powershell
.\run_demo.ps1 -Cohort A -Patients 2 -Mode dashboard -ViewerRenderMaxSide 900
```

For the recorded demo, the default `560` is recommended.

## Viewer cache / timing options

The demo launcher now configures a bounded nearby-slice cache for the Streamlit multiplanar viewer.

```powershell
# Default: neighbour prefetch is disabled; each requested slice renders directly
.\run_demo.ps1 -Cohort A -Patients 2 -Mode dashboard

# Diagnostic timing output in the terminal
.\run_demo.ps1 -Cohort A -Patients 2 -Mode dashboard -ViewerTiming

# Custom prefetch radius
.\run_demo.ps1 -Cohort A -Patients 2 -Mode dashboard -ViewerPrefetchRadius 6
```

`ViewerPrefetchRadius` is capped between 0 and 24. The cache is display-only; no pipeline outputs are changed.

### Viewer performance defaults (V4.7)

The launcher now sets `ViewerPrefetchRadius` to `0` by default.  This avoids a large slider jump
blocking while many neighbouring Cohort A slices are synchronously rendered.  V4.7 also replaces
`np.take` with direct basic indexing for NIfTI slice extraction, which is especially important for
large Fortran-contiguous arrays.  JPEG caching remains enabled for already viewed slices.
