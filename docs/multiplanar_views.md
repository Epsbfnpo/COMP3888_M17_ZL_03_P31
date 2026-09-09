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
