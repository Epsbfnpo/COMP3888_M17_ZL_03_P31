from __future__ import annotations

"""
Batch rigid alignment + aligned lesion-feature extraction for Cohort A.

This tool is intended to prepare the input consumed by:

    tools/generate_lesion_pair_costs.py
    tools/run_tracking_evaluation_batch.py

For every selected patient it:

1. loads BL/FU CT and lesion masks from cohort_a_subset_pairs.csv;
2. performs BL -> FU CT registration using the project's rigid-only
   src.registration.register_patient_ct();
3. reuses a previous *rigid-only* registration when safe;
4. extracts BL connected components in the ORIGINAL BL mask and assigns
   stable temporary lesion IDs before any resampling;
5. resamples that integer component-label image into FU geometry with
   nearest-neighbour interpolation using the rigid Elastix transform;
6. extracts FU components in the native FU mask;
7. exports one combined CSV containing:
       - aligned/FU-space centroids for matching, and
       - native BL/FU centroids for ground-truth identity mapping.

Important
---------
The expert correspondence CSV is NOT used by this script.  Ground truth must
remain evaluation-only and must not leak into feature generation.

The batch output uses temporary lesion IDs such as:
    0a09c8844b_BL_L001
    0a09c8844b_FU_L001

These are working component identifiers, not expert lesion IDs and not
predicted correspondences.

Example
-------
python tools/generate_aligned_lesion_features_batch.py ^
  --pairs outputs/cohort_a_15/cohort_a_subset_pairs.csv ^
  --data-root "C:\\Users\\Bill5\\Desktop\\Longitudinal_CT_v2" ^
  --out outputs/aligned_lesion_features.csv ^
  --expected-patients 15
"""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cohort_a_loading import (  # noqa: E402
    CohortALoadError,
    LoadedVolume,
    load_nifti_volume,
    load_pair_manifest,
    load_patient_pair,
)
from src.lesion_components import (  # noqa: E402
    EmptyLesionMaskError,
    LesionMaskError,
    extract_individual_lesions,
    lesion_rows,
)
from src.registration import (  # noqa: E402
    RegistrationError,
    register_patient_ct,
)


class AlignedFeatureBatchError(RuntimeError):
    """Raised when the aligned-feature batch cannot proceed safely."""


FEATURE_COLUMNS = [
    "patient_id",
    "timepoint",
    "lesion_id",
    "temporary_lesion_id",
    "component_index",
    "source_label",

    # Aligned/FU-space features used by the matching cost generator.
    "voxel_count",
    "volume_ml",
    "centroid_x_vox",
    "centroid_y_vox",
    "centroid_z_vox",
    "centroid_x_mm",
    "centroid_y_mm",
    "centroid_z_mm",
    "coordinate_space",

    # Native-timepoint identity features used only for ground-truth mapping.
    # BL values refer to the original BL image; FU values refer to the
    # original FU image.
    "native_voxel_count",
    "native_volume_ml",
    "native_centroid_x_vox",
    "native_centroid_y_vox",
    "native_centroid_z_vox",
    "native_centroid_x_mm",
    "native_centroid_y_mm",
    "native_centroid_z_mm",
    "native_coordinate_space",

    "connectivity",
    "mask_path",
    "native_mask_path",
    "mask_role",
    "registration_stage",
    "transform_parameter_path",
]


def _normalise_patient_ids(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.add(text)
            result.append(text)

    return result


def _parse_patient_ids(text: str) -> list[str]:
    return _normalise_patient_ids(part for part in str(text).split(","))


def _select_patients(
    manifest: pd.DataFrame,
    requested: list[str] | None,
) -> list[str]:
    if "patient_id" not in manifest.columns:
        raise AlignedFeatureBatchError(
            "Pair manifest is missing required column: patient_id"
        )

    manifest_ids = _normalise_patient_ids(manifest["patient_id"])

    duplicated = (
        manifest["patient_id"]
        .astype(str)
        .str.strip()
        .duplicated(keep=False)
    )
    if duplicated.any():
        duplicate_ids = sorted(
            manifest.loc[duplicated, "patient_id"].astype(str).unique()
        )
        raise AlignedFeatureBatchError(
            "Expected one BL/FU pair row per patient, but duplicate patient_id "
            "value(s) were found: " + ", ".join(duplicate_ids)
        )

    if requested:
        available = set(manifest_ids)
        missing = [patient_id for patient_id in requested if patient_id not in available]
        if missing:
            raise AlignedFeatureBatchError(
                "Requested patient ID(s) not present in pair manifest: "
                + ", ".join(missing)
            )
        return requested

    return manifest_ids


def _geometry_matches(
    a: LoadedVolume,
    b: LoadedVolume,
    *,
    affine_atol: float = 1e-3,
) -> bool:
    return (
        a.data.shape == b.data.shape
        and np.allclose(
            np.asarray(a.metadata.affine, dtype=float),
            np.asarray(b.metadata.affine, dtype=float),
            rtol=0.0,
            atol=affine_atol,
        )
    )


def _validate_native_pair_geometry(
    *,
    patient_id: str,
    bl_ct: LoadedVolume,
    fu_ct: LoadedVolume,
    bl_mask: LoadedVolume,
    fu_mask: LoadedVolume,
) -> None:
    if not _geometry_matches(bl_ct, bl_mask):
        raise AlignedFeatureBatchError(
            f"{patient_id}: BL CT and BL lesion mask do not share the same "
            "shape/affine geometry."
        )

    if not _geometry_matches(fu_ct, fu_mask):
        raise AlignedFeatureBatchError(
            f"{patient_id}: FU CT and FU lesion mask do not share the same "
            "shape/affine geometry."
        )


def _transform_file_is_rigid(transform_path: Path) -> bool:
    """Best-effort guard against accidentally reusing an old affine result."""
    try:
        text = transform_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    # Elastix rigid registration normally persists an Euler transform.
    # Keep this permissive enough for version differences.
    transform_match = re.search(
        r'\(Transform\s+"?([^"\s\)]+)"?\)',
        text,
        flags=re.IGNORECASE,
    )
    if transform_match is None:
        return False

    transform_name = transform_match.group(1).lower()
    return "euler" in transform_name or "rigid" in transform_name


def _rigid_registration_is_ready(
    registration_dir: Path,
    *,
    followup_ct: LoadedVolume,
) -> bool:
    """
    Reuse only a clean rigid-only registration.

    If TransformParameters.1.txt exists, the folder came from an old multi-stage
    run (e.g. rigid + affine), so this batch intentionally re-runs rigid-only.
    """
    registered_path = registration_dir / "registered_baseline_ct.nii.gz"
    transform0 = registration_dir / "TransformParameters.0.txt"
    transform1 = registration_dir / "TransformParameters.1.txt"

    if (
        not registered_path.exists()
        or not transform0.exists()
        or transform1.exists()
        or not _transform_file_is_rigid(transform0)
    ):
        return False

    try:
        registered = load_nifti_volume(
            registered_path,
            role="existing rigid-registered baseline CT",
        )
    except Exception:
        return False

    return _geometry_matches(registered, followup_ct)


def _run_or_reuse_rigid_registration(
    *,
    pair,
    registration_dir: Path,
    rerun_registration: bool,
    number_of_resolutions: int,
    maximum_iterations: int,
    number_of_spatial_samples: int,
    log_to_console: bool,
    enable_low_overlap_fallback: bool,
    low_overlap_fallback_ratio: float,
) -> tuple[Path, Path, bool, str]:
    """
    Run rigid registration with a conservative two-pass policy.

    Pass 1 uses the normal rigid registration parameters.
    Pass 2 is attempted only when Pass 1 raises RegistrationError, and lowers
    Elastix RequiredRatioOfValidSamples for scans with limited BL/FU overlap.

    Returns:
        registered_ct_path,
        rigid_transform_parameter_path,
        reused,
        registration_mode
    """
    assert pair.followup.ct is not None

    ready = _rigid_registration_is_ready(
        registration_dir,
        followup_ct=pair.followup.ct,
    )

    if ready and not rerun_registration:
        return (
            (registration_dir / "registered_baseline_ct.nii.gz").resolve(),
            (registration_dir / "TransformParameters.0.txt").resolve(),
            True,
            "REUSED_RIGID",
        )

    try:
        result = register_patient_ct(
            pair,
            output_dir=registration_dir,
            stages=("rigid",),
            number_of_resolutions=number_of_resolutions,
            maximum_iterations=maximum_iterations,
            number_of_spatial_samples=number_of_spatial_samples,
            required_ratio_of_valid_samples=None,
            log_to_console=log_to_console,
            overwrite=True,
        )
        registration_mode = "STANDARD_RIGID"

    except RegistrationError as first_error:
        if not enable_low_overlap_fallback:
            raise

        print(
            f"  Standard rigid registration failed for {pair.patient_id}."
        )
        print(
            "  Retrying with low-overlap rigid fallback: "
            f"RequiredRatioOfValidSamples={low_overlap_fallback_ratio:g}"
        )

        try:
            result = register_patient_ct(
                pair,
                output_dir=registration_dir,
                stages=("rigid",),
                number_of_resolutions=number_of_resolutions,
                maximum_iterations=maximum_iterations,
                number_of_spatial_samples=number_of_spatial_samples,
                required_ratio_of_valid_samples=low_overlap_fallback_ratio,
                log_to_console=log_to_console,
                overwrite=True,
            )
            registration_mode = "LOW_OVERLAP_FALLBACK"

        except RegistrationError as fallback_error:
            raise RegistrationError(
                f"Rigid registration failed for patient '{pair.patient_id}' "
                "in both the standard pass and low-overlap fallback. "
                f"Standard error: {first_error}. "
                f"Fallback error: {fallback_error}"
            ) from fallback_error

    if len(result.transform_parameter_paths) != 1:
        raise AlignedFeatureBatchError(
            f"{pair.patient_id}: expected exactly one rigid transform parameter "
            f"file, got {len(result.transform_parameter_paths)}."
        )

    transform_path = result.transform_parameter_paths[0]
    if not _transform_file_is_rigid(transform_path):
        raise AlignedFeatureBatchError(
            f"{pair.patient_id}: generated transform does not look like a "
            f"rigid/Euler transform: {transform_path}"
        )

    return (
        result.registered_ct_path.resolve(),
        transform_path.resolve(),
        False,
        registration_mode,
    )


def _save_label_volume_like(
    *,
    label_data: np.ndarray,
    reference_volume: LoadedVolume,
    output_path: Path,
) -> LoadedVolume:
    """
    Save an integer component-label image using the reference volume geometry.

    This is used to persist the *native* BL connected-component IDs before
    registration. Transformix then resamples these IDs with nearest-neighbour
    interpolation, so two different BL lesions remain two different labels even
    if they become spatially adjacent in FU space.
    """
    data = np.asarray(label_data)

    if data.ndim != 3:
        raise AlignedFeatureBatchError(
            f"Component label image must be 3-D; got shape {data.shape}."
        )
    if np.any(data < 0):
        raise AlignedFeatureBatchError(
            "Component label image contains negative labels."
        )

    max_label = int(data.max(initial=0))
    dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
    data = data.astype(dtype, copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import nibabel as nib

        reference_img = reference_volume.image
        header = reference_img.header.copy()
        header.set_data_dtype(dtype)

        output_img = nib.Nifti1Image(
            data,
            np.asarray(reference_volume.metadata.affine, dtype=float),
            header=header,
        )

        try:
            output_img.set_qform(
                np.asarray(reference_volume.metadata.affine, dtype=float),
                code=int(reference_img.header["qform_code"]),
            )
        except Exception:
            pass

        try:
            output_img.set_sform(
                np.asarray(reference_volume.metadata.affine, dtype=float),
                code=int(reference_img.header["sform_code"]),
            )
        except Exception:
            pass

        nib.save(output_img, str(output_path))

    except Exception as exc:
        raise AlignedFeatureBatchError(
            f"Could not save component label image '{output_path}': {exc}"
        ) from exc

    saved = load_nifti_volume(
        output_path,
        role="native BL connected-component label mask",
        preserve_dtype=True,
    )

    if not _geometry_matches(saved, reference_volume):
        raise AlignedFeatureBatchError(
            "Saved component label image does not preserve the reference geometry."
        )

    return saved


def _patch_transform_parameter_file_for_label_mask(
    source_path: Path,
    destination_path: Path,
) -> None:
    """
    Copy an Elastix transform parameter file and force nearest-neighbour
    interpolation for a discrete lesion-label mask.

    We patch a copy rather than the registration file itself so the original
    CT transform remains auditable and untouched.
    """
    text = source_path.read_text(encoding="utf-8", errors="strict")

    replacements = {
        "FinalBSplineInterpolationOrder": "0",
        "DefaultPixelValue": "0",
        "WriteResultImage": '"true"',
    }

    for key, value in replacements.items():
        pattern = re.compile(
            rf"\({re.escape(key)}\s+[^\)]*\)",
            flags=re.IGNORECASE,
        )
        replacement = f"({key} {value})"
        if pattern.search(text):
            text = pattern.sub(replacement, text)
        else:
            text += "\n" + replacement + "\n"

    # A float result is safe during Transformix. Values are rounded back to
    # integer labels after nearest-neighbour resampling.
    pixel_pattern = re.compile(
        r"\(ResultImagePixelType\s+[^\)]*\)",
        flags=re.IGNORECASE,
    )
    if pixel_pattern.search(text):
        text = pixel_pattern.sub('(ResultImagePixelType "float")', text)
    else:
        text += '\n(ResultImagePixelType "float")\n'

    destination_path.write_text(text, encoding="utf-8")


def _transform_bl_mask_to_fu_space(
    *,
    patient_id: str,
    baseline_mask: LoadedVolume,
    followup_ct: LoadedVolume,
    rigid_transform_path: Path,
    output_path: Path,
    reuse_existing: bool,
) -> LoadedVolume:
    """
    Resample the BL label mask into FU geometry using the existing rigid
    transform and nearest-neighbour interpolation.
    """
    if reuse_existing and output_path.exists():
        try:
            existing = load_nifti_volume(
                output_path,
                role=f"{patient_id} existing aligned BL lesion mask",
                preserve_dtype=True,
            )
            if _geometry_matches(existing, followup_ct):
                return existing
        except Exception:
            pass

    try:
        import itk
    except ImportError as exc:
        raise AlignedFeatureBatchError(
            "ITKElastix is required to transform lesion masks. Install it with "
            "'python -m pip install itk-elastix'."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{patient_id}_mask_transform_") as tmp:
        tmp_dir = Path(tmp)
        patched_transform = tmp_dir / "TransformParameters.mask.txt"

        _patch_transform_parameter_file_for_label_mask(
            rigid_transform_path,
            patched_transform,
        )

        try:
            parameter_object = itk.ParameterObject.New()
            parameter_object.AddParameterFile(str(patched_transform))

            # Read from disk to preserve physical spacing/origin/direction.
            input_mask = itk.imread(str(baseline_mask.metadata.path), itk.F)

            transformix = itk.TransformixFilter.New(input_mask)
            transformix.SetTransformParameterObject(parameter_object)

            if hasattr(transformix, "SetLogToConsole"):
                transformix.SetLogToConsole(False)
            if hasattr(transformix, "SetLogToFile"):
                transformix.SetLogToFile(False)

            transformix.UpdateLargestPossibleRegion()
            transformed = transformix.GetOutput()

            temporary_float_path = tmp_dir / "aligned_bl_mask_float.nii.gz"
            itk.imwrite(transformed, str(temporary_float_path))

        except Exception as exc:
            raise AlignedFeatureBatchError(
                f"{patient_id}: Transformix failed while resampling the BL "
                f"lesion mask into FU space: {exc}"
            ) from exc

        transformed_float = load_nifti_volume(
            temporary_float_path,
            role=f"{patient_id} transformed BL lesion mask",
            preserve_dtype=False,
        )

        if not _geometry_matches(transformed_float, followup_ct):
            raise AlignedFeatureBatchError(
                f"{patient_id}: transformed BL lesion mask does not match FU "
                "CT shape/affine geometry."
            )

        values = np.asarray(transformed_float.data, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise AlignedFeatureBatchError(
                f"{patient_id}: transformed BL lesion mask contains NaN/Inf."
            )

        rounded = np.rint(values)
        if not np.allclose(values, rounded, rtol=0.0, atol=1e-4):
            max_error = float(np.max(np.abs(values - rounded)))
            raise AlignedFeatureBatchError(
                f"{patient_id}: transformed BL mask contains non-integer values "
                f"after nearest-neighbour resampling (max rounding error "
                f"{max_error:.6g})."
            )

        if np.any(rounded < 0):
            raise AlignedFeatureBatchError(
                f"{patient_id}: transformed BL lesion mask contains negative labels."
            )

        max_label = int(rounded.max(initial=0))
        if max_label <= np.iinfo(np.uint16).max:
            mask_dtype = np.uint16
        else:
            mask_dtype = np.uint32

        label_data = rounded.astype(mask_dtype, copy=False)

        # Save using nibabel and explicitly use the FU NIfTI affine/header.
        # This ensures downstream centroid coordinates are reported in FU RAS mm.
        try:
            import nibabel as nib

            reference_img = followup_ct.image
            header = reference_img.header.copy()
            header.set_data_dtype(mask_dtype)

            output_img = nib.Nifti1Image(
                label_data,
                np.asarray(followup_ct.metadata.affine, dtype=float),
                header=header,
            )

            try:
                output_img.set_qform(
                    np.asarray(followup_ct.metadata.affine, dtype=float),
                    code=int(reference_img.header["qform_code"]),
                )
            except Exception:
                pass

            try:
                output_img.set_sform(
                    np.asarray(followup_ct.metadata.affine, dtype=float),
                    code=int(reference_img.header["sform_code"]),
                )
            except Exception:
                pass

            nib.save(output_img, str(output_path))

        except Exception as exc:
            raise AlignedFeatureBatchError(
                f"{patient_id}: could not save aligned BL lesion mask "
                f"'{output_path}': {exc}"
            ) from exc

    aligned = load_nifti_volume(
        output_path,
        role=f"{patient_id} aligned BL lesion mask",
        preserve_dtype=True,
    )

    if not _geometry_matches(aligned, followup_ct):
        raise AlignedFeatureBatchError(
            f"{patient_id}: saved aligned BL lesion mask failed the FU geometry check."
        )

    return aligned


def _feature_rows_from_patient(
    *,
    patient_id: str,
    native_bl_result,
    aligned_bl_component_mask: LoadedVolume | None,
    fu_mask: LoadedVolume,
    connectivity: int,
    transform_parameter_path: Path,
) -> tuple[list[dict[str, object]], int, int]:
    """
    Build matching features while preserving native lesion identity.

    BL identity:
        Connected components are created ONCE in the original BL mask.
        Their component IDs are then resampled as integer labels into FU space.
        We do not run connected-components again after registration.

    FU identity:
        Components are extracted directly from the native FU mask.

    This prevents two distinct BL lesions that become adjacent after resampling
    from being accidentally merged into one temporary lesion ID.
    """
    output_rows: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # BL: IDs originate in native BL space; positions used by the matcher
    # are measured from the resampled labelled mask in FU space.
    # ------------------------------------------------------------------
    bl_count = 0

    if native_bl_result is not None:
        if aligned_bl_component_mask is None:
            raise AlignedFeatureBatchError(
                f"{patient_id}: native BL lesions exist but no aligned BL "
                "component-label mask is available."
            )

        aligned_labels = np.asarray(
            aligned_bl_component_mask.data,
            dtype=np.int64,
        )

        spacing = np.asarray(
            aligned_bl_component_mask.metadata.spacing_mm,
            dtype=float,
        )
        voxel_volume_ml = float(np.prod(spacing) / 1000.0)
        affine = np.asarray(
            aligned_bl_component_mask.metadata.affine,
            dtype=float,
        )

        expected_labels = {
            int(lesion.component_index)
            for lesion in native_bl_result.lesions
        }
        actual_labels = {
            int(v)
            for v in np.unique(aligned_labels)
            if int(v) > 0
        }

        unexpected = sorted(actual_labels - expected_labels)
        if unexpected:
            raise AlignedFeatureBatchError(
                f"{patient_id}: aligned BL component mask contains unexpected "
                f"label(s): {unexpected}"
            )

        lost_labels: list[int] = []

        for native_lesion in native_bl_result.lesions:
            label = int(native_lesion.component_index)
            coords = np.argwhere(aligned_labels == label)

            if len(coords) == 0:
                lost_labels.append(label)
                continue

            aligned_centroid_vox = coords.mean(axis=0)
            homogeneous = np.array(
                [
                    aligned_centroid_vox[0],
                    aligned_centroid_vox[1],
                    aligned_centroid_vox[2],
                    1.0,
                ],
                dtype=float,
            )
            aligned_centroid_mm = (affine @ homogeneous)[:3]

            aligned_voxel_count = int(len(coords))

            output_rows.append(
                {
                    "patient_id": patient_id,
                    "timepoint": "BL",
                    "lesion_id": native_lesion.temporary_id,
                    "temporary_lesion_id": native_lesion.temporary_id,
                    "component_index": native_lesion.component_index,
                    "source_label": native_lesion.source_label,

                    "voxel_count": aligned_voxel_count,
                    "volume_ml": float(
                        aligned_voxel_count * voxel_volume_ml
                    ),
                    "centroid_x_vox": float(aligned_centroid_vox[0]),
                    "centroid_y_vox": float(aligned_centroid_vox[1]),
                    "centroid_z_vox": float(aligned_centroid_vox[2]),
                    "centroid_x_mm": float(aligned_centroid_mm[0]),
                    "centroid_y_mm": float(aligned_centroid_mm[1]),
                    "centroid_z_mm": float(aligned_centroid_mm[2]),
                    "coordinate_space": "FU_RAS_mm",

                    "native_voxel_count": native_lesion.voxel_count,
                    "native_volume_ml": native_lesion.volume_ml,
                    "native_centroid_x_vox": native_lesion.centroid_voxel[0],
                    "native_centroid_y_vox": native_lesion.centroid_voxel[1],
                    "native_centroid_z_vox": native_lesion.centroid_voxel[2],
                    "native_centroid_x_mm": native_lesion.centroid_world_mm[0],
                    "native_centroid_y_mm": native_lesion.centroid_world_mm[1],
                    "native_centroid_z_mm": native_lesion.centroid_world_mm[2],
                    "native_coordinate_space": "BL_NATIVE",

                    "connectivity": connectivity,
                    "mask_path": str(
                        aligned_bl_component_mask.metadata.path
                    ),
                    "native_mask_path": "",
                    "mask_role": (
                        "BL_native_component_IDs_rigidly_resampled_to_FU"
                    ),
                    "registration_stage": "rigid",
                    "transform_parameter_path": str(
                        transform_parameter_path
                    ),
                }
            )

        if lost_labels:
            raise AlignedFeatureBatchError(
                f"{patient_id}: {len(lost_labels)} native BL lesion component(s) "
                "disappeared completely when resampled into FU geometry: "
                + ", ".join(str(v) for v in lost_labels)
                + ". The registration/FOV should be reviewed instead of "
                  "silently changing lesion identity."
            )

        bl_count = len(native_bl_result.lesions)

    # ------------------------------------------------------------------
    # FU: native FU space is already the fixed/reference space, so native
    # and aligned coordinates are the same.
    # ------------------------------------------------------------------
    try:
        fu_result = extract_individual_lesions(
            fu_mask,
            patient_id=patient_id,
            timepoint="FU",
            connectivity=connectivity,
        )
    except EmptyLesionMaskError:
        fu_result = None
        fu_count = 0
    else:
        fu_count = fu_result.lesion_count

        for lesion in fu_result.lesions:
            output_rows.append(
                {
                    "patient_id": patient_id,
                    "timepoint": "FU",
                    "lesion_id": lesion.temporary_id,
                    "temporary_lesion_id": lesion.temporary_id,
                    "component_index": lesion.component_index,
                    "source_label": lesion.source_label,

                    "voxel_count": lesion.voxel_count,
                    "volume_ml": lesion.volume_ml,
                    "centroid_x_vox": lesion.centroid_voxel[0],
                    "centroid_y_vox": lesion.centroid_voxel[1],
                    "centroid_z_vox": lesion.centroid_voxel[2],
                    "centroid_x_mm": lesion.centroid_world_mm[0],
                    "centroid_y_mm": lesion.centroid_world_mm[1],
                    "centroid_z_mm": lesion.centroid_world_mm[2],
                    "coordinate_space": "FU_RAS_mm",

                    "native_voxel_count": lesion.voxel_count,
                    "native_volume_ml": lesion.volume_ml,
                    "native_centroid_x_vox": lesion.centroid_voxel[0],
                    "native_centroid_y_vox": lesion.centroid_voxel[1],
                    "native_centroid_z_vox": lesion.centroid_voxel[2],
                    "native_centroid_x_mm": lesion.centroid_world_mm[0],
                    "native_centroid_y_mm": lesion.centroid_world_mm[1],
                    "native_centroid_z_mm": lesion.centroid_world_mm[2],
                    "native_coordinate_space": "FU_NATIVE",

                    "connectivity": connectivity,
                    "mask_path": str(fu_mask.metadata.path),
                    "native_mask_path": str(fu_mask.metadata.path),
                    "mask_role": "FU_reference_mask",
                    "registration_stage": "reference",
                    "transform_parameter_path": "",
                }
            )

    return output_rows, bl_count, fu_count

def _validate_feature_table(
    features: pd.DataFrame,
    *,
    successful_patient_ids: list[str],
) -> None:
    if features.empty:
        raise AlignedFeatureBatchError("No aligned lesion features were produced.")

    missing = sorted(set(FEATURE_COLUMNS) - set(features.columns))
    if missing:
        raise AlignedFeatureBatchError(
            "Internal error: output feature table is missing column(s): "
            + ", ".join(missing)
        )

    expected_spaces = set(features["coordinate_space"].astype(str))
    if expected_spaces != {"FU_RAS_mm"}:
        raise AlignedFeatureBatchError(
            "All output lesion coordinates must be in FU_RAS_mm."
        )

    duplicate = features.duplicated(
        ["patient_id", "timepoint", "lesion_id"],
        keep=False,
    )
    if duplicate.any():
        example = features.loc[
            duplicate,
            ["patient_id", "timepoint", "lesion_id"],
        ].iloc[0]
        raise AlignedFeatureBatchError(
            "Duplicate lesion ID in aligned feature output: "
            f"{example['patient_id']} {example['timepoint']} "
            f"{example['lesion_id']}"
        )

    for column in (
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
        "volume_ml",
        "native_centroid_x_vox",
        "native_centroid_y_vox",
        "native_centroid_z_vox",
        "native_volume_ml",
    ):
        values = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise AlignedFeatureBatchError(
                f"Output feature column {column} contains non-finite values."
            )

    if (pd.to_numeric(features["volume_ml"]) <= 0).any():
        raise AlignedFeatureBatchError(
            "Output volume_ml must be > 0 for every lesion."
        )

    present_ids = set(features["patient_id"].astype(str))
    missing_patients = [
        patient_id
        for patient_id in successful_patient_ids
        if patient_id not in present_ids
    ]
    if missing_patients:
        raise AlignedFeatureBatchError(
            "Successful patients missing from final feature table: "
            + ", ".join(missing_patients)
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rigidly align Cohort A BL CT/masks to FU space for multiple "
            "patients and export one combined aligned lesion-feature CSV."
        )
    )

    parser.add_argument(
        "--pairs",
        required=True,
        help="Path to the selected cohort_a_subset_pairs.csv.",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help=(
            "Cohort A root used to resolve relative manifest paths. If omitted, "
            "COHORT_A_ROOT is used when available."
        ),
    )
    parser.add_argument(
        "--patient-ids",
        default="",
        help=(
            "Optional comma-separated patient IDs. If omitted, every patient "
            "in --pairs is processed."
        ),
    )
    parser.add_argument(
        "--expected-patients",
        type=int,
        default=15,
        help=(
            "Expected number of selected patients. Default: 15. Use 0 to "
            "disable the count check."
        ),
    )

    parser.add_argument(
        "--out",
        default="outputs/aligned_lesion_features.csv",
        help="Combined aligned lesion-feature CSV.",
    )
    parser.add_argument(
        "--status-out",
        default="",
        help=(
            "Per-patient batch status CSV. Default: same directory/name as "
            "--out with '_status.csv' suffix."
        ),
    )
    parser.add_argument(
        "--registration-root",
        default="outputs/registration",
        help=(
            "Root directory for per-patient rigid registration outputs. "
            "Existing clean rigid-only results are reused by default."
        ),
    )
    parser.add_argument(
        "--aligned-mask-root",
        default="outputs/aligned_masks",
        help="Root directory for BL masks resampled into FU geometry.",
    )
    parser.add_argument(
        "--rerun-registration",
        action="store_true",
        help="Force rigid CT registration to run again for every patient.",
    )
    parser.add_argument(
        "--rerun-mask-transform",
        action="store_true",
        help="Force BL lesion-mask resampling to run again for every patient.",
    )

    parser.add_argument(
        "--connectivity",
        choices=[6, 18, 26],
        type=int,
        default=18,
        help="3-D connected-component connectivity. Default: 18.",
    )

    # Same conservative settings already used by src.registration.
    parser.add_argument(
        "--number-of-resolutions",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--maximum-iterations",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--number-of-spatial-samples",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--registration-log-to-console",
        action="store_true",
        help="Show detailed Elastix optimisation logs.",
    )
    parser.add_argument(
        "--low-overlap-fallback-ratio",
        type=float,
        default=0.05,
        help=(
            "RequiredRatioOfValidSamples used only after the normal rigid pass "
            "fails. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--disable-low-overlap-fallback",
        action="store_true",
        help="Disable the second-pass low-overlap rigid fallback.",
    )

    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.expected_patients < 0:
        raise AlignedFeatureBatchError("--expected-patients must be >= 0.")

    if args.number_of_resolutions < 1:
        raise AlignedFeatureBatchError("--number-of-resolutions must be >= 1.")

    if args.maximum_iterations < 1:
        raise AlignedFeatureBatchError("--maximum-iterations must be >= 1.")

    if args.number_of_spatial_samples < 128:
        raise AlignedFeatureBatchError(
            "--number-of-spatial-samples must be >= 128."
        )

    if (
        not math.isfinite(args.low_overlap_fallback_ratio)
        or not (0.0 < args.low_overlap_fallback_ratio <= 1.0)
    ):
        raise AlignedFeatureBatchError(
            "--low-overlap-fallback-ratio must be in the interval (0, 1]."
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        _validate_args(args)

        pairs_path = Path(args.pairs).expanduser().resolve()
        if not pairs_path.exists():
            raise FileNotFoundError(f"Pair manifest does not exist: {pairs_path}")

        manifest = load_pair_manifest(pairs_path)

        requested = _parse_patient_ids(args.patient_ids)
        patient_ids = _select_patients(
            manifest,
            requested=requested or None,
        )

        if not patient_ids:
            raise AlignedFeatureBatchError("No patients selected.")

        if (
            args.expected_patients > 0
            and len(patient_ids) != args.expected_patients
        ):
            raise AlignedFeatureBatchError(
                f"Expected {args.expected_patients} patients but selected "
                f"{len(patient_ids)}. Check --pairs/--patient-ids, or set "
                "--expected-patients 0 to disable this guard."
            )

        root_text = str(args.data_root).strip()
        if not root_text:
            root_text = os.environ.get("COHORT_A_ROOT", "").strip()

        data_root = (
            Path(root_text).expanduser().resolve()
            if root_text
            else None
        )

        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if str(args.status_out).strip():
            status_path = Path(args.status_out).expanduser().resolve()
        else:
            status_path = out_path.with_name(
                out_path.stem + "_status.csv"
            )
        status_path.parent.mkdir(parents=True, exist_ok=True)

        registration_root = Path(args.registration_root).expanduser().resolve()
        aligned_mask_root = Path(args.aligned_mask_root).expanduser().resolve()

        registration_root.mkdir(parents=True, exist_ok=True)
        aligned_mask_root.mkdir(parents=True, exist_ok=True)

        print("Generate aligned lesion features - batch")
        print("----------------------------------------")
        print(f"Patients          : {len(patient_ids)}")
        print(f"Pair manifest     : {pairs_path}")
        print(f"Data root         : {data_root if data_root else '<manifest/env resolution>'}")
        print(f"Registration      : rigid only")
        print(
            "Low-overlap retry : "
            + (
                "disabled"
                if args.disable_low_overlap_fallback
                else f"enabled (ratio={args.low_overlap_fallback_ratio:g})"
            )
        )
        print(f"Connectivity      : {args.connectivity}")
        print(f"Registration root : {registration_root}")
        print(f"Aligned mask root : {aligned_mask_root}")
        print(f"Feature output    : {out_path}")
        print()

        all_feature_rows: list[dict[str, object]] = []
        status_rows: list[dict[str, object]] = []
        successful_patient_ids: list[str] = []

        for index, patient_id in enumerate(patient_ids, start=1):
            print(
                f"[{index:02d}/{len(patient_ids):02d}] "
                f"{patient_id} - loading..."
            )

            registration_dir = registration_root / patient_id
            patient_mask_dir = aligned_mask_root / patient_id
            native_component_mask_path = (
                patient_mask_dir
                / "baseline_native_component_labels.nii.gz"
            )
            aligned_mask_path = (
                patient_mask_dir
                / "baseline_component_labels_in_fu_space.nii.gz"
            )

            status_row: dict[str, object] = {
                "patient_id": patient_id,
                "status": "FAILED",
                "registration_reused": False,
                "registration_mode": "",
                "longitudinal_case": "",
                "registered_bl_ct_path": "",
                "rigid_transform_path": "",
                "native_bl_component_mask_path": "",
                "aligned_bl_mask_path": "",
                "bl_lesions": pd.NA,
                "fu_lesions": pd.NA,
                "feature_rows": 0,
                "error": "",
            }

            try:
                pair = load_patient_pair(
                    pairs_path,
                    patient_id,
                    data_root=data_root,
                    modalities=("ct", "lesion_mask"),
                    allow_missing=False,
                )

                if pair.baseline.ct is None or pair.followup.ct is None:
                    raise AlignedFeatureBatchError(
                        f"{patient_id}: BL/FU CT was unexpectedly not loaded."
                    )
                if (
                    pair.baseline.lesion_mask is None
                    or pair.followup.lesion_mask is None
                ):
                    raise AlignedFeatureBatchError(
                        f"{patient_id}: BL/FU lesion mask was unexpectedly not loaded."
                    )

                _validate_native_pair_geometry(
                    patient_id=patient_id,
                    bl_ct=pair.baseline.ct,
                    fu_ct=pair.followup.ct,
                    bl_mask=pair.baseline.lesion_mask,
                    fu_mask=pair.followup.lesion_mask,
                )

                print(
                    f"[{index:02d}/{len(patient_ids):02d}] "
                    f"{patient_id} - rigid registration..."
                )

                (
                    registered_ct_path,
                    transform_path,
                    registration_reused,
                    registration_mode,
                ) = _run_or_reuse_rigid_registration(
                    pair=pair,
                    registration_dir=registration_dir,
                    rerun_registration=args.rerun_registration,
                    number_of_resolutions=args.number_of_resolutions,
                    maximum_iterations=args.maximum_iterations,
                    number_of_spatial_samples=args.number_of_spatial_samples,
                    log_to_console=args.registration_log_to_console,
                    enable_low_overlap_fallback=(
                        not args.disable_low_overlap_fallback
                    ),
                    low_overlap_fallback_ratio=args.low_overlap_fallback_ratio,
                )

                print(
                    f"[{index:02d}/{len(patient_ids):02d}] "
                    f"{patient_id} - rigid registration complete "
                    f"[{registration_mode}]"
                )

                # ------------------------------------------------------
                # Preserve BL lesion identity BEFORE registration.
                # ------------------------------------------------------
                try:
                    native_bl_result = extract_individual_lesions(
                        pair.baseline.lesion_mask,
                        patient_id=patient_id,
                        timepoint="BL",
                        connectivity=args.connectivity,
                    )
                except EmptyLesionMaskError:
                    native_bl_result = None

                aligned_bl_mask = None
                native_component_mask = None

                if native_bl_result is not None:
                    print(
                        f"[{index:02d}/{len(patient_ids):02d}] "
                        f"{patient_id} - preserve BL component IDs..."
                    )

                    native_component_mask = _save_label_volume_like(
                        label_data=native_bl_result.labelled_mask,
                        reference_volume=pair.baseline.lesion_mask,
                        output_path=native_component_mask_path,
                    )

                    print(
                        f"[{index:02d}/{len(patient_ids):02d}] "
                        f"{patient_id} - BL component labels -> FU space..."
                    )

                    aligned_bl_mask = _transform_bl_mask_to_fu_space(
                        patient_id=patient_id,
                        baseline_mask=native_component_mask,
                        followup_ct=pair.followup.ct,
                        rigid_transform_path=transform_path,
                        output_path=aligned_mask_path,
                        reuse_existing=not args.rerun_mask_transform,
                    )
                else:
                    print(
                        f"[{index:02d}/{len(patient_ids):02d}] "
                        f"{patient_id} - BL mask empty; no BL transform needed."
                    )

                patient_rows, bl_count, fu_count = _feature_rows_from_patient(
                    patient_id=patient_id,
                    native_bl_result=native_bl_result,
                    aligned_bl_component_mask=aligned_bl_mask,
                    fu_mask=pair.followup.lesion_mask,
                    connectivity=args.connectivity,
                    transform_parameter_path=transform_path,
                )

                all_feature_rows.extend(patient_rows)
                successful_patient_ids.append(patient_id)

                status_row.update(
                    {
                        "status": "OK",
                        "registration_reused": registration_reused,
                        "registration_mode": registration_mode,
                        "registered_bl_ct_path": str(registered_ct_path),
                        "rigid_transform_path": str(transform_path),
                        "native_bl_component_mask_path": (
                            str(native_component_mask.metadata.path)
                            if native_component_mask is not None
                            else ""
                        ),
                        "aligned_bl_mask_path": (
                            str(aligned_bl_mask.metadata.path)
                            if aligned_bl_mask is not None
                            else ""
                        ),
                        "bl_lesions": bl_count,
                        "fu_lesions": fu_count,
                        "feature_rows": len(patient_rows),
                        "error": "",
                    }
                )

                if bl_count > 0 and fu_count == 0:
                    longitudinal_case = "ALL_DISAPPEARING_CANDIDATE"
                elif bl_count == 0 and fu_count > 0:
                    longitudinal_case = "ALL_NEW_CANDIDATE"
                elif bl_count == 0 and fu_count == 0:
                    longitudinal_case = "NO_LESIONS_BOTH_TIMEPOINTS"
                else:
                    longitudinal_case = "BL_AND_FU_LESIONS"

                status_row["longitudinal_case"] = longitudinal_case

                print(
                    f"[{index:02d}/{len(patient_ids):02d}] "
                    f"{patient_id} - OK "
                    f"(BL={bl_count}, FU={fu_count}, rows={len(patient_rows)}, "
                    f"case={longitudinal_case})"
                )

            except (
                CohortALoadError,
                RegistrationError,
                LesionMaskError,
                AlignedFeatureBatchError,
                FileNotFoundError,
                ValueError,
                RuntimeError,
            ) as exc:
                status_row["error"] = f"{type(exc).__name__}: {exc}"
                print(
                    f"[{index:02d}/{len(patient_ids):02d}] "
                    f"{patient_id} - FAILED: {status_row['error']}"
                )

            except Exception as exc:
                # Keep the other patients' outputs instead of losing the whole
                # batch, but record unexpected failures clearly.
                status_row["error"] = f"{type(exc).__name__}: {exc}"
                print(
                    f"[{index:02d}/{len(patient_ids):02d}] "
                    f"{patient_id} - FAILED (unexpected): {status_row['error']}"
                )

            status_rows.append(status_row)

            # Persist progress after every patient so a long 15-patient batch
            # does not lose earlier work if a later registration fails.
            pd.DataFrame(status_rows).to_csv(status_path, index=False)

            if all_feature_rows:
                progress_features = pd.DataFrame(
                    all_feature_rows,
                    columns=FEATURE_COLUMNS,
                )
                progress_features.to_csv(out_path, index=False)

        status = pd.DataFrame(status_rows)
        features = pd.DataFrame(
            all_feature_rows,
            columns=FEATURE_COLUMNS,
        )

        if not features.empty:
            _validate_feature_table(
                features,
                successful_patient_ids=successful_patient_ids,
            )

            patient_order = {
                patient_id: index
                for index, patient_id in enumerate(patient_ids)
            }
            timepoint_order = {"BL": 0, "FU": 1}

            features["_patient_order"] = features["patient_id"].map(patient_order)
            features["_timepoint_order"] = features["timepoint"].map(timepoint_order)

            features = (
                features.sort_values(
                    [
                        "_patient_order",
                        "_timepoint_order",
                        "component_index",
                    ],
                    kind="mergesort",
                )
                .drop(columns=["_patient_order", "_timepoint_order"])
                .reset_index(drop=True)
            )

        features.to_csv(out_path, index=False)
        status.to_csv(status_path, index=False)

        ok_count = int((status["status"] == "OK").sum())
        failed_count = int((status["status"] == "FAILED").sum())

        print()
        print("=" * 78)
        print("BATCH COMPLETE")
        print("=" * 78)
        print(f"Patients OK       : {ok_count}/{len(patient_ids)}")
        print(f"Patients failed   : {failed_count}/{len(patient_ids)}")
        print(f"Lesion rows       : {len(features)}")
        print(f"Aligned features  : {out_path}")
        print(f"Batch status      : {status_path}")
        print(f"Registration root : {registration_root}")
        print(f"Aligned masks     : {aligned_mask_root}")

        if failed_count:
            print()
            print(
                "Some patients failed. Successful patient features were still "
                "written, but do NOT run the 15-patient accuracy report until "
                "batch status shows 15/15 OK."
            )
            raise SystemExit(2)

    except (
        CohortALoadError,
        RegistrationError,
        LesionMaskError,
        AlignedFeatureBatchError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
