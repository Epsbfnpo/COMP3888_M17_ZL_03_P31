# Functional Tests — P31-27, P31-29 and P31-31

These tests are deterministic, synthetic functional tests for the three assigned Jira testing stories. They do not require Cohort A or Cohort B data, so failures are easier to reproduce than dataset-dependent manual tests.

## Run commands

Run all three assigned stories:

```bash
make test-assigned
```

Run one Jira story only:

```bash
make test-p31-27
make test-p31-29
make test-p31-31
```

The existing project-wide suite remains:

```bash
make test
```

## Functional Test: Lesion Extraction and Feature Generation

Test file: `tests/test_p31_27_lesion_features_functional.py`

| Acceptance criterion | Automated evidence |
| --- | --- |
| Individual connected lesions are identified | Synthetic BL mask contains 2 components and FU mask contains 3; exactly 5 feature rows must be produced. |
| Unique lesion/component ID | The expected BL/FU IDs are asserted and the exported `lesion_id` column must be unique. |
| Centroid coordinates | A known 2×2×2 lesion has voxel centroid `(1.5, 1.5, 1.5)` and world centroid `(3, 3, 3)` mm under a 2 mm affine. |
| Lesion volume | 2 mm isotropic voxels have volume 0.008 mL; an 8-voxel lesion must have volume 0.064 mL. |
| BL/FU features can be exported | Features are exported to CSV and reloaded; required downstream columns are checked. |
| Empty/invalid masks handled correctly | An empty BL mask creates no fake BL records while valid FU lesions remain; a probability-map-like mask is rejected with `LesionMaskError`. |

Expected result: one valid feature record per extracted lesion.

## Functional Test: Lesion Pair Cost Generation

Test file: `tests/test_p31_29_pair_cost_functional.py`

| Acceptance criterion | Automated evidence |
| --- | --- |
| All valid BL/FU candidate combinations | 2 BL × 3 FU lesions must create exactly 6 candidate rows. |
| Distance cost calculated correctly | A `(0,0,0)` to `(3,4,12)` example must produce 13 mm distance and the configured normalised distance cost. |
| Size-difference cost | 2 mL vs 8 mL must produce a fractional size difference of 0.75. |
| PET only when comparable | SUV values are used when present; missing SUV is ignored; raw PET with mismatched units is not used. |
| Distance gating | A 100 mm candidate is removed by a 20 mm gate and represented as `inf` in the labelled matrix. |
| Labelled BL × FU matrix | Exported CSV row labels are BL lesion IDs and columns are FU lesion IDs. The exported matrix is then passed directly to `match_patient_cost_matrix`. |

Expected result: a valid labelled matrix that the matching algorithm can consume directly.

## Functional Test: Visualization and Tracking Summary

Test file: `tests/test_p31_31_visualization_summary_functional.py`

| Acceptance criterion | Automated evidence |
| --- | --- |
| Axial view | A synthetic RAS volume is mapped to the axial axis and a valid 2-D slice is extracted. |
| Coronal view | Same test for the coronal anatomical plane. |
| Sagittal view | Same test for the sagittal anatomical plane. |
| Lesion mask overlay | `_overlay_mask` must change lesion pixels while preserving non-lesion grayscale pixels. |
| BL/FU lesion locations linked to results | MATCHED and MERGING resolve both endpoints; NEW resolves FU only; DISAPPEARING resolves BL only. |
| Required outcome types displayed | `OUTCOME_STYLES` must explicitly contain MATCHED, NEW, DISAPPEARING and MERGING. |
| Correct tracking summary | A deterministic mixture of MATCHED/MERGING/NEW/DISAPPEARING rows is checked against exact BL, FU, matched, appearing, disappearing and merge counts. |

Expected result: the viewer support functions can represent the required anatomical views and outcomes, and the selected-patient tracking summary reports the exact expected counts.

## Maintenance fix included

The latest application file is `tools/align_longitudinal_patient_v5_mapped.py`, but two older tests still imported the deleted V4 dashboard. Their imports were updated to V5 so `make test` does not fail simply because of the dashboard rename.
