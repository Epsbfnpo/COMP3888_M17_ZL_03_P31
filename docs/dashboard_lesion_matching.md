# Dashboard lesion correspondence

The existing Streamlit multiplanar viewer can now load and run the lesion
min-cost-flow matcher for the currently selected patient.

## Launch

From the repository root:

```powershell
streamlit run tools/align_longitudinal_patient_v4_mapped.py
```

## Inputs

The **Lesion matching** sidebar section accepts:

- **Matching results CSV**: the combined `lesion_matches.csv` produced by the
  tracking pipeline. The dashboard filters this file by the selected patient.
- **Aligned lesion features CSV**: the matching pipeline's feature table. Its
  `FU_RAS_mm` voxel centroids connect each BL/FU lesion ID to the common
  registered image grid.
- **Cost matrix directory**: a directory containing labelled matrices named
  `<patient_id>_cost_matrix.csv`.
- The same disappearing, new-lesion, merge, and maximum-merge-size settings
  supported by `src/lesion_min_cost_flow.py`.

Saved results are loaded automatically. To recompute the selected patient,
click **Run / re-run lesion matching**. The dashboard runs the existing
NetworkX min-cost-flow implementation and replaces only that patient's rows in
the combined results CSV; results for all other patients are preserved.

## Display and errors

The lesion correspondence panel is patient-scoped and sits in the same view as
the patient's registered BL and FU scans. It shows BL/FU lesion counts, counts
for `MATCHED`, `MERGING`, `DISAPPEARING`, and `NEW`, and the lesion-level result
table. Selecting another patient immediately loads that patient's rows.

Choose a row under **Inspect correspondence on aligned scans** to display a
focused Registered-BL/FU pair. Both panels use the shared FU grid. Each lesion
is shown on its own centroid slice with a high-contrast spatial marker, so a
pair remains inspectable when the two centroids fall on different slices.
`DISAPPEARING` and `NEW` outcomes clearly show the absent side instead of
inventing a location.

Missing result files, a selected patient with no saved rows, a missing cost
matrix, malformed CSV data, and solver failures are displayed in the dashboard
with actionable messages instead of an unhandled exception. Missing or
inconsistent aligned features leave the result table usable while disabling
only the image-linking panel with a specific warning.
