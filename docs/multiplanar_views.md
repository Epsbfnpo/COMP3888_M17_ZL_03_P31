# P31-17 — Support Coronal and Sagittal Views

This iteration extends the existing Streamlit longitudinal alignment viewer with linked multiplanar viewing.

The dashboard now supports:

- **Axial** — transverse/head-to-foot slicing;
- **Coronal** — frontal/front-to-back slicing;
- **Sagittal** — side-to-side slicing.

## Run

From the repository root:

```powershell
python -m streamlit run tools/align_longitudinal_patient_v4_mapped.py
```

The existing rigid BL → FU registration must already exist for the selected patient, or it can be created with **Run / re-run alignment**.

The sidebar now contains:

```text
Modality
  CT
  PET

View plane
  Axial
  Coronal
  Sagittal

Show lesion masks
```

The slice slider automatically changes to the number of slices in the selected anatomical plane.

Each plane keeps its own Streamlit slice state, so switching Axial → Coronal → Sagittal does not leave an invalid index from the previous plane.



## Anatomical plane handling

The implementation does not assume that NIfTI array axes are always stored in RAS order.

`src/multiplanar_view.py` uses the NIfTI orientation codes exposed by the loader to find the voxel axis normal to each anatomical plane:

```text
Axial    -> Superior / Inferior axis (S/I)
Coronal  -> Anterior / Posterior axis (A/P)
Sagittal -> Left / Right axis (L/R)
```

For example, a volume stored as:

```text
(R, A, S)
```

uses:

```text
Sagittal -> axis 0
Coronal  -> axis 1
Axial    -> axis 2
```

but a valid volume stored in a different axis order is handled from its own orientation metadata rather than by hard-coded axis numbers.



## BL / FU spatial consistency

FU CT is the fixed/reference image for rigid registration.

Therefore:

```text
Registered BL CT
Registered BL lesion mask
Registered BL PET
FU CT
FU lesion mask
FU PET on CT grid
```

are viewed on the FU geometry.

For a selected plane and slice index:

```text
Registered BL slice N
FU slice N
```

refer to the same FU-space anatomical plane.

This remains true when switching among Axial, Coronal, and Sagittal views.

### Original BL panel

Original BL remains in its native BL geometry.

For visual reference, the centre of the selected FU anatomical plane is mapped through the saved Elastix rigid transform:

```text
FU physical point
      ↓
Elastix fixed(FU) -> moving(BL) transform
      ↓
Original BL physical point
      ↓
Original BL voxel coordinate
      ↓
nearest native BL plane
```

This generalises the earlier axial-only mapping to all three anatomical planes.

If rotation is present, a FU plane can correspond to an oblique plane in native BL space. The Original BL panel is therefore the nearest native plane for context; **Registered BL and FU are the exact aligned comparison panels**.



## CT display

CT keeps the existing window presets:

```text
Soft tissue
Lung
Bone
Wide
```

The whole volume is windowed once and cached as `uint8`, so moving the slice slider only extracts a 2-D slice instead of repeating HU normalisation.



## PET display

When BL and FU PET are available, the dashboard can switch to **PET**.

### FU PET

FU PET is resampled onto the FU CT grid using the existing NIfTI world geometry and linear interpolation.

This is a grid resampling operation, not a new registration.

### BL PET

The already-saved BL → FU rigid transform is applied to BL PET with continuous/linear interpolation and written to:

```text
outputs/registration/<patient_id>/registered_baseline_pet.nii.gz
```

The transform therefore remains the same CT-derived longitudinal registration used elsewhere in the pipeline.

### Original BL PET

For the native BL reference panel, BL PET is resampled onto the native BL CT grid. This allows the native BL lesion mask to remain spatially consistent with the displayed PET image.

PET intensities are display-normalised with robust percentiles. This changes only the viewer brightness and does not modify source PET data or downstream quantitative features.

If a patient does not contain both BL and FU PET, selecting PET gives a clear error and the user can switch back to CT.



## Lesion masks

`Show lesion masks` works in all three planes and with both display modalities.

The rules are unchanged:

- native BL mask overlays the native BL-grid panel;
- BL mask transformed with nearest-neighbour interpolation overlays Registered BL;
- FU mask overlays FU;
- positive lesion voxels are displayed with the existing red/orange fill and bright boundary.

The same selected anatomical plane and slice index are used for image and mask extraction.



## Physical display aspect correction

Coronal and sagittal medical-image slices can look stretched if the viewer
treats each voxel as a square screen pixel. NIfTI voxels are often anisotropic:
the physical millimetres represented by one row pixel may differ from those
represented by one column pixel.

P31-17 therefore applies a **display-only physical aspect correction** after
image/mask overlay composition. The source NIfTI files, registration transform,
affines, lesion measurements, and matcher inputs are unchanged.

For each selected anatomical plane, the viewer derives the two displayed pixel
spacings from the volume metadata and the slice rotation used by the dashboard.
For a standard RAS volume this corresponds to:

```text
Axial    -> display row = A/P spacing, display column = L/R spacing
Coronal  -> display row = S/I spacing, display column = L/R spacing
Sagittal -> display row = S/I spacing, display column = A/P spacing
```

The rendered height-to-width ratio is then based on physical extent:

```text
physical height = displayed rows    × row spacing (mm)
physical width  = displayed columns × column spacing (mm)
```

This prevents a `200 × 409` coronal/sagittal matrix from automatically being
drawn with a raw `200:409` screen aspect when its voxel spacing implies a
different anatomical proportion.

### BL and FU grids

- **Original BL** uses the native BL voxel spacing.
- **Registered BL** uses the FU/reference voxel spacing.
- **FU** uses the FU/reference voxel spacing.

Registered BL and FU therefore retain exactly the same display aspect, as they
share the same FU geometry. Original BL may legitimately have a different field
of view or physical extent; the correction does not force all three panels to
be the same height.

Lesion overlays are composed before aspect correction, so masks and images are
resized together and cannot drift apart visually. The same correction is used
for CT and PET display modes.



## Tests

New pure multiplanar tests are in:

```text
tests/test_multiplanar_view.py
```

They cover:

- anatomical plane → voxel-axis resolution from NIfTI orientation codes;
- correct slice count for Axial / Coronal / Sagittal;
- slice extraction from all three voxel axes;
- centre-world-coordinate updates for each plane;
- invalid plane and out-of-range slice handling;
- display row/column spacing after slice rotation;
- physical-aspect resampling for anisotropic and isotropic pixels.

Run locally after installing `requirements.txt`:

```powershell
python -m pytest -q tests/test_multiplanar_view.py
```

Then smoke-test the real Streamlit dashboard on at least one registered Cohort B patient by switching all three planes, CT/PET, and lesion-mask overlay.

## Interactive slider performance

The multiplanar slider uses a Streamlit fragment, so changing the slice index reruns only the viewer fragment rather than the complete matching/evaluation dashboard. CT/PET volumes and registration outputs remain cached.

For interactive rendering, physical-aspect correction now uses Pillow's compiled bilinear resize and does **not** upscale native medical-image slices before sending them to the browser. The longest rendered side is capped at 560 pixels by default, which is sufficient for the three-column dashboard and avoids repeatedly creating 900-pixel intermediate images. This changes display rendering only; NIfTI data, registration geometry, masks, lesion coordinates, matching and evaluation outputs are unchanged.

For a larger screenshot render, set `P31_VIEWER_MAX_SIDE` before starting Streamlit, for example `$env:P31_VIEWER_MAX_SIDE = "900"`. The interactive default remains 560 for responsiveness.

## Interactive slider performance (V4.6)

The multiplanar slider runs inside a Streamlit fragment, so moving the slice control does not need to recompute the tracking pipeline. V4.6 additionally caches **display-only JPEG panels** for already viewed slices and pre-renders a bounded neighbourhood around a cache miss (default: disabled (0 slices)). This is designed for presentation/demo use and does not alter NIfTI volumes, masks, registration transforms, lesion features, costs, or matching results.

The viewer cache is session-local and bounded. It is cleared after a new registration. Physical voxel aspect correction is still applied before a panel is encoded, so the performance optimisation does not reintroduce the stretched coronal/sagittal display problem.

Launcher controls:

```powershell
# Normal demo: cache nearby slices automatically
.\run_demo.ps1 -Cohort A -Patients 2 -Mode dashboard

# Print fragment/cache timing lines in the terminal for diagnosis
.\run_demo.ps1 -Cohort A -Patients 2 -Mode dashboard -ViewerTiming

# Change the prefetch neighbourhood if required (0 disables prefetch)
.\run_demo.ps1 -Cohort A -Patients 2 -Mode dashboard -ViewerPrefetchRadius 10
```

With `-ViewerTiming`, terminal lines such as the following are expected:

```text
[P31 viewer] ... slice=138 cache=HIT current_render=0.0ms ... fragment_python=...
```

A cache hit confirms that slice extraction, mask overlay, physical-aspect correction, and image encoding were reused. There is still a small browser/server round-trip because Streamlit widgets are server-driven. A fully client-side radiology-style scroll viewer would require a custom JavaScript component (or a specialised web viewer such as OHIF/Cornerstone), not Java.

### Streamlit console messages

V4.6 also removes two sources of noisy console output:

- deprecated `use_container_width=True` calls were replaced with the current `width="stretch"` API;
- the mixed-type cost-detail `Value` column is converted to display strings before `st.dataframe`, avoiding PyArrow's automatic fallback for values such as `np.bool_` mixed with floats/strings.

The previous PyArrow traceback was recoverable and did not corrupt results, but it added unnecessary work on full dashboard reruns and made performance diagnosis harder.


## V4.7 slider performance fix: Fortran-order NIfTI slicing

Cohort A exposed a second performance issue that was not visible with smaller Cohort B studies.
NiBabel/NIfTI arrays are often **Fortran-contiguous**.  The earlier viewer used `np.take` to extract a
2-D slice from a 3-D volume.  On a large Fortran-order array, `np.take(..., axis=...)` can be orders
of magnitude slower than direct NumPy indexing and may take seconds for one mask slice.

V4.7 now uses basic indexing (`data[:, :, z]`, `data[:, y, :]`, or `data[x, :, :]`) to obtain a cheap
2-D view first, then rotates/flips and copies only that 2-D slice for display.  This applies to CT/PET
and lesion masks without changing image values, geometry, registration, matching, or evaluation.

Synchronous neighbour prefetch is now **disabled by default** (`P31_VIEWER_PREFETCH_RADIUS=0`).
Large jumps therefore render only the requested slice instead of blocking while 20 neighbouring
slices are also generated.  The existing JPEG cache is retained, so revisiting a slice is still an
immediate cache hit.  Prefetch remains opt-in for experiments via `-ViewerPrefetchRadius N`.
