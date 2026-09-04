# Longitudinal CT Alignment

This document describes the current **ITKElastix rigid-only alignment workflow** used in the COMP3888 P31 longitudinal review pipeline.

> **Current dashboard version:** V4 — Rigid-mapped Original BL + masks  
> **Registration policy:** Rigid only  
> **Fixed/reference image:** Follow-up CT (FU)  
> **Moving image:** Baseline CT (BL)

---

## Quick Start

Run the Streamlit alignment dashboard **from the repository root**:

```bash
streamlit run tools/align_longitudinal_patient_v4_mapped.py
```

If Streamlit is installed inside the project virtual environment on Windows, the equivalent command may be:

```powershell
.\.venv\Scripts\streamlit.exe run tools/align_longitudinal_patient_v4_mapped.py
```

The dashboard uses these default paths:

```text
Pair manifest:
outputs/cohort_a_subset/cohort_a_subset_pairs.csv

Registration output root:
outputs/registration
```

If the manifest contains paths that require an external dataset root, provide it in **Data root** or set one of the supported environment variables:

```text
DATA_ROOT
COHORT_B_ROOT
COHORT_A_ROOT
```

---

## Streamlit Sidebar: What Each Field Means

The left sidebar is the main control panel for selecting data, running alignment, and changing the viewer.

### Data

#### `Pair manifest`

Path to the BL/FU pair manifest.

Default:

```text
outputs/cohort_a_subset/cohort_a_subset_pairs.csv
```

The manifest tells the application which Baseline and Follow-up scans belong to the same patient and where the corresponding CT and lesion-mask files are located.

---

#### `Data root`

Optional root directory for the imaging dataset.

Use this when paths stored in the manifest are relative to the original dataset directory.

It can normally be left blank when:

- the manifest already contains usable paths; or
- `DATA_ROOT`, `COHORT_B_ROOT`, or `COHORT_A_ROOT` is already configured.

---

#### `Registration output root`

Directory where registration results are stored.

Default:

```text
outputs/registration
```

Each patient receives a separate directory, for example:

```text
outputs/registration/
└── 0a09c8844b/
    ├── registered_baseline_ct.nii.gz
    ├── registered_baseline_lesion_mask.nii.gz
    └── TransformParameters.0.txt
```

`registered_baseline_lesion_mask.nii.gz` is generated when lesion-mask display is requested and a BL mask is available.

---

### Patient

#### `Patient`

Selects the patient pair loaded from the pair manifest.

The dashboard loads:

- Original BL CT
- FU CT
- BL lesion mask, if available
- FU lesion mask, if available
- previously generated rigid-registration results, if available

---

### Alignment

The dashboard currently displays:

```text
Conservative baseline: Rigid only
```

#### `Run / re-run alignment`

Runs ITKElastix registration for the selected patient.

The current V4 dashboard explicitly runs:

```python
stages=("rigid",)
```

Therefore the active dashboard does **not** run affine or deformable registration.

Running the button again overwrites the previous registration output for that patient.

---

### Viewer

#### `CT window`

Changes only the CT display window.

Available presets are:

| Option | HU range | Typical purpose |
|---|---:|---|
| Soft tissue | -160 to 240 | General soft-tissue review |
| Lung | -1000 to 400 | Lung structures |
| Bone | -500 to 1500 | Bone structures |
| Wide | -1000 to 2000 | Broad overview |

Changing the CT window does **not** change the registration result.

---

#### `Show lesion masks`

Controls whether lesion masks are overlaid on the CT images.

When enabled:

- the Original BL panel shows the original BL lesion mask;
- the Registered BL panel shows the BL lesion mask transformed into FU space;
- the FU panel shows the original FU lesion mask.

The BL mask uses the **same saved rigid transform** as the BL CT.

Because masks are discrete labels, the transformed BL mask uses **nearest-neighbour interpolation** so that registration does not create fractional label values.

---

## Viewer Panels

The main alignment viewer contains three columns.

| Panel | Meaning |
|---|---|
| **Original BL CT** | Original, untouched Baseline CT in its native geometry |
| **Registered BL CT** | Baseline CT after rigid registration and resampling into FU geometry |
| **FU CT** | Original Follow-up CT used as the fixed/reference image |

### 1. Original BL CT

This is the actual original Baseline image.

It is **not** a reconstructed image and is **not** a slice selected using image-similarity search.

The dashboard uses the saved Elastix rigid transform to map the centre of the currently selected FU slice into the Original-BL physical coordinate system.

It then displays the nearest native BL axial slice.

The caption includes information similar to:

```text
Original data · rigid-mapped native slice 61 (mapped z=61.4)
```

This means the current FU slice centre maps to approximately BL slice `61.4`, so native BL slice `61` is displayed.

### Important limitation

When rotation is present, an entire FU axial plane can correspond to an oblique plane in the native BL image.

Therefore the Original BL panel should be interpreted as the **nearest native axial slice for visual reference**, not as a mathematically exact resampled plane.

---

### 2. Registered BL CT

This is the Baseline CT after ITKElastix rigid registration.

It has been resampled into the **FU image geometry**.

Therefore:

```text
Registered BL slice N
```

and:

```text
FU slice N
```

refer to the same FU-space output slice.

This makes direct visual comparison much more meaningful than comparing raw BL and FU slice indices.

---

### 3. FU CT

This is the unchanged Follow-up CT.

FU is the **fixed/reference image** in registration.

Its geometry defines the coordinate system used by:

- Registered BL CT
- Registered BL lesion mask
- downstream aligned BL lesion coordinates

---

## FU-Space Axial Slice Slider

The viewer slider is labelled:

```text
FU-space axial slice
```

The selected index directly controls:

- Registered BL CT slice
- FU CT slice

because both are in the same FU geometry.

For the Original BL panel, the same slider position is passed through the saved rigid spatial mapping to determine the nearest corresponding native BL slice.

Conceptually:

```text
Selected FU slice
       │
       ├──────────────► FU CT
       │
       ├──────────────► Registered BL CT
       │
       └── rigid FU→BL point mapping
                         │
                         ▼
                  Original BL slice
```

---

# Why Alignment Is Needed

Baseline and Follow-up scans of the same patient are not guaranteed to share the same voxel coordinates.

Differences can come from:

- patient position;
- patient rotation;
- scan origin;
- scan field of view;
- number of slices;
- voxel spacing;
- image direction;
- scanner acquisition differences.

Therefore raw voxel indices should not be treated as directly comparable.

For example:

```text
BL lesion centroid: (168, 210, 71)
FU lesion centroid: (175, 205, 43)
```

A large difference in voxel coordinates does not necessarily mean the lesions are anatomically far apart.

Alignment first establishes a common spatial frame.

---

# Registration Configuration

The current workflow uses:

```text
Fixed image  = FU CT
Moving image = BL CT
Transform    = 3-D rigid Euler transform
```

A 3-D rigid transform has six degrees of freedom:

```text
Rotation:
Rx
Ry
Rz

Translation:
Tx
Ty
Tz
```

The rigid transform can:

- rotate the BL scan;
- translate the BL scan.

It does **not** intentionally perform:

- scaling;
- stretching;
- shearing;
- local warping;
- deformable registration.

This conservative choice matches the current project scope and helps preserve genuine longitudinal anatomical and lesion changes.

---

# ITKElastix Registration Process

The registration module reads the original NIfTI files directly using ITK so that important spatial metadata is preserved:

- spacing;
- origin;
- direction.

The active registration configuration uses:

```text
Metric:
AdvancedMattesMutualInformation

Number of resolutions:
3

Maximum iterations:
256

Number of spatial samples:
4096

Automatic transform initialisation:
GeometricalCenter

Use direction cosines:
true
```

The general process is:

```text
Original BL CT
      +
Original FU CT
      │
      ▼
ITKElastix rigid registration
      │
      ▼
Find best global rotation + translation
      │
      ├────────► Save TransformParameters.0.txt
      │
      ▼
Resample BL into FU geometry
      │
      ▼
registered_baseline_ct.nii.gz
```

---

# Why the Registered BL Can Look Different in Size

Rigid registration itself does **not** stretch the anatomy.

However, the Registered BL image is resampled onto the FU voxel grid.

For example, the original images could have different:

```text
BL:
shape   = 512 × 512 × 180
spacing = 1 × 1 × 3 mm

FU:
shape   = 512 × 512 × 260
spacing = 1 × 1 × 2 mm
```

After registration, the Registered BL adopts the FU output geometry.

Therefore the displayed image can appear differently sampled even though the rigid transform itself contains only rotation and translation.

---

# Registration Output

For each patient, the main rigid-registration output is:

```text
outputs/registration/<patient_id>/
├── registered_baseline_ct.nii.gz
└── TransformParameters.0.txt
```

When lesion-mask overlay is used:

```text
registered_baseline_lesion_mask.nii.gz
```

may also be generated.

## `registered_baseline_ct.nii.gz`

BL CT resampled into FU geometry.

The application validates that its:

- array shape; and
- affine geometry

match the FU CT before enabling aligned slice comparison.

---

## `TransformParameters.0.txt`

Saved Elastix rigid transform.

This file contains the spatial relationship established during registration, including the Euler transform parameters.

It is important because it can be reused without running registration again.

The transform is used for:

- producing Registered BL;
- transforming the BL lesion mask;
- mapping FU-space positions back to Original-BL space;
- future coordinate transformation before lesion matching.

---

# Transform Direction

Elastix transform direction can be confusing.

In this project:

```text
Fixed  = FU
Moving = BL
```

The saved Elastix transform maps a physical point from the **fixed-image domain to the moving-image domain**:

```text
FU physical coordinate
        │
        ▼
saved Elastix transform
        │
        ▼
Original BL physical coordinate
```

The current V4 viewer uses this mapping directly when locating the Original BL native slice.

Therefore, for the viewer's FU → Original-BL point mapping, a separate numerical inverse is not required.

This is different from the more intuitive visual statement:

```text
"BL is registered into FU space"
```

Both statements are valid in their respective contexts:

- **image output:** BL is resampled into FU geometry;
- **Elastix point mapping:** output/fixed FU points are mapped to locations in moving BL for sampling.

---

# Original BL Mapping

For each FU axial slice, the V4 viewer:

1. finds the centre voxel of the FU slice;
2. converts it to a physical/world coordinate;
3. converts NIfTI/nibabel RAS coordinates to ITK/Elastix LPS coordinates;
4. applies the saved rigid transform;
5. converts the result back from LPS to RAS;
6. converts the physical location into Original-BL voxel coordinates;
7. reads the resulting continuous BL `z` coordinate;
8. displays the nearest native BL axial slice.

Conceptually:

```text
FU voxel index
      │
      ▼
FU RAS physical coordinate
      │
      ▼
RAS → LPS
      │
      ▼
Elastix rigid transform
      │
      ▼
BL LPS physical coordinate
      │
      ▼
LPS → RAS
      │
      ▼
Original-BL voxel coordinate
      │
      ▼
nearest native BL axial slice
```

This is why the Original BL panel is spatially related to the selected FU slice rather than simply using the same raw slice number.

---

# Lesion Mask Alignment

The BL lesion mask starts in Original-BL geometry.

After rigid registration:

```text
Original BL mask
       │
       │ same saved rigid transform
       ▼
Registered BL mask
       │
       ▼
FU geometry
```

The transformed mask uses:

```text
FinalNearestNeighborInterpolator
FinalBSplineInterpolationOrder = 0
DefaultPixelValue = 0
```

This preserves the discrete lesion labels.

Using ordinary continuous interpolation could create invalid values such as:

```text
0.25
0.62
0.91
```

between label `0` and label `1`, which would be inappropriate for a segmentation mask.

---

# Alignment vs Lesion Matching

Alignment and lesion matching are separate tasks.

## Alignment asks:

> Where is the Baseline anatomy located relative to the Follow-up scan?

ITKElastix answers this by estimating a global rigid spatial transform.

## Matching asks:

> Which BL lesion corresponds to which FU lesion?

That is a downstream problem.

The planned architecture is:

```text
BL/FU scans
    │
    ▼
Rigid CT alignment
    │
    ▼
Transform BL mask / lesion coordinates
    │
    ▼
Extract aligned lesion features
    │
    ├── centroid distance
    ├── volume
    ├── overlap
    └── other features
    │
    ▼
Construct candidate edge costs
    │
    ▼
NetworkX min_cost_flow
    │
    ▼
BL ↔ FU lesion correspondence
```

ITKElastix therefore does **not** decide which lesions match.

It improves the spatial coordinates used by the matcher.

---

# Why Rigid Only?

The current project scope deliberately uses a conservative rigid-only baseline.

Rigid registration corrects:

```text
✓ global translation
✓ global rotation
```

It does not correct:

```text
✗ breathing deformation
✗ organ deformation
✗ local anatomical movement
✗ tumour growth or shrinkage
✗ body-shape change
```

This is useful for longitudinal analysis because the registration algorithm should not aggressively deform anatomy in order to make two time points look identical.

A real biological change should remain visible rather than being "explained away" by a flexible deformation model.

---

# Current Limitations

The current alignment should not be interpreted as perfect anatomical correspondence.

Important limitations include:

1. **Rigid-only model**  
   Local organ and tissue deformation is not corrected.

2. **Original BL panel is a nearest native slice**  
   If rotation makes the corresponding BL plane oblique, one native axial slice cannot exactly represent the full plane.

3. **Registration quality can vary by scan pair**  
   Large changes in field of view, positioning, anatomy, or acquisition may reduce alignment quality.

4. **Alignment is not lesion matching**  
   A spatially close BL/FU lesion pair is only a candidate until the downstream matching algorithm evaluates it.

5. **Visual similarity is not the only validation criterion**  
   The saved transform, geometry checks, lesion-mask behaviour, and later matching performance should also be validated.

---

# Recommended Client Explanation

A concise explanation for client meetings is:

> We added an ITKElastix rigid-registration step before lesion matching. The Follow-up CT is used as the reference image and the Baseline CT is aligned into that space using only global rotation and translation. We deliberately do not use scaling or deformable registration, so the alignment remains conservative and interpretable. The same rigid transform is reused for the Baseline lesion mask, while the original BL data is preserved. The dashboard shows Original BL, Registered BL, and FU side by side, and the saved transform is also used to locate the corresponding Original-BL position for each FU-space slice. This alignment step does not perform lesion matching; it provides a common spatial frame for the later min-cost-flow matcher.

---

# Code Locations

Current relevant modules:

```text
src/registration.py
    ITKElastix registration implementation.

tools/align_longitudinal_patient_v4_mapped.py
    Current Streamlit V4 alignment dashboard.

outputs/registration/<patient_id>/
    Per-patient registration outputs.
```

The underlying `registration.py` module still supports both `rigid` and `affine` stage names for reuse and experimentation.

However, the **current V4 Streamlit workflow explicitly requests only**:

```python
stages=("rigid",)
```

Therefore the user-facing alignment workflow documented here is **rigid only**.

---

# Summary

The current alignment workflow can be summarised as:

```text
Original BL CT
      │
      │ ITKElastix rigid registration
      ▼
Registered BL CT ─────────────┐
                              │
Original BL mask              │
      │                       │
      │ same rigid transform  │
      ▼                       │
Registered BL mask            │
                              ▼
                       Common FU geometry
                              ▲
                              │
                           FU CT
                           FU mask
                              │
                              ▼
                     Lesion feature comparison
                              │
                              ▼
                      Min-cost-flow matching
```

The core principle is:

> **Registration normalises spatial position; it does not determine lesion correspondence.**
