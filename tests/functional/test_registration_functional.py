from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.cohort_a_loading import load_nifti_volume
from src.registration import (
    RegistrationError,
    register_ct_pair,
    warp_baseline_mask_to_fu,
)


def _save_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def _make_synthetic_case(tmp_path: Path):
    """
    Build a small deterministic CT phantom.

    BL and FU contain the same anatomy, but BL has a different physical origin.
    A successful rigid registration therefore has to place BL into FU space
    rather than merely copying BL geometry.
    """
    shape = (32, 32, 24)
    x, y, z = np.indices(shape, dtype=np.float32)

    body = (
        ((x - 15.5) / 11.0) ** 2
        + ((y - 15.5) / 9.0) ** 2
        + ((z - 11.5) / 9.0) ** 2
        <= 1.0
    )
    bone = (
        ((x - 15.5) / 3.2) ** 2
        + ((y - 15.5) / 3.2) ** 2
        + ((z - 11.5) / 7.0) ** 2
        <= 1.0
    )
    organ = (
        ((x - 20.0) / 4.0) ** 2
        + ((y - 14.0) / 3.0) ** 2
        + ((z - 12.0) / 4.0) ** 2
        <= 1.0
    )
    lesion = (
        (x - 20.0) ** 2
        + (y - 14.0) ** 2
        + (z - 12.0) ** 2
        <= 2.2**2
    )

    ct = np.full(shape, -1000.0, dtype=np.float32)
    ct[body] = 30.0
    ct[bone] = 550.0
    ct[organ] = 100.0
    ct[lesion] = 180.0

    mask = lesion.astype(np.uint16)

    fu_affine = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    bl_affine = fu_affine.copy()
    bl_affine[:3, 3] = (6.0, -4.0, 5.0)

    bl_ct_path = _save_nifti(tmp_path / "bl_ct.nii.gz", ct, bl_affine)
    fu_ct_path = _save_nifti(tmp_path / "fu_ct.nii.gz", ct, fu_affine)
    bl_mask_path = _save_nifti(tmp_path / "bl_mask.nii.gz", mask, bl_affine)
    fu_mask_path = _save_nifti(tmp_path / "fu_mask.nii.gz", mask, fu_affine)

    bl_ct = load_nifti_volume(bl_ct_path, role="synthetic BL CT")
    fu_ct = load_nifti_volume(fu_ct_path, role="synthetic FU CT")
    bl_mask = load_nifti_volume(
        bl_mask_path,
        role="synthetic BL lesion mask",
        preserve_dtype=True,
    )
    fu_mask = load_nifti_volume(
        fu_mask_path,
        role="synthetic FU lesion mask",
        preserve_dtype=True,
    )

    return bl_ct, fu_ct, bl_mask, fu_mask


def _dice(binary_a: np.ndarray, binary_b: np.ndarray) -> float:
    a = np.asarray(binary_a, dtype=bool)
    b = np.asarray(binary_b, dtype=bool)
    denominator = int(a.sum()) + int(b.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.logical_and(a, b).sum()) / float(denominator)


def test_bl_ct_registration_produces_valid_fu_space_outputs(tmp_path: Path) -> None:
    """
    FT-REG-01

    Acceptance coverage:
      - BL CT registration completes successfully.
      - Registered BL CT has the same shape as FU CT.
      - Registered BL CT has matching FU affine geometry.
      - Required rigid transform parameters are generated.
      - A BL lesion mask can be transformed into FU reference space.
      - The resulting registration has meaningful anatomical overlap.
    """
    bl_ct, fu_ct, bl_mask, fu_mask = _make_synthetic_case(tmp_path)

    registration_dir = tmp_path / "registration"

    result = register_ct_pair(
        bl_ct,
        fu_ct,
        patient_id="functional_registration_case",
        output_dir=registration_dir,
        stages=("rigid",),
        initialization_method="adaptive",
        number_of_resolutions=2,
        maximum_iterations=96,
        number_of_spatial_samples=2048,
        minimum_body_dice=0.50,
        log_to_console=False,
        overwrite=True,
    )

    # 1. Registration completed and output exists.
    assert result.registered_ct_path.is_file()

    # Latest adaptive behaviour: equal scan extents should use geometric-centre
    # initialization rather than the anatomy-profile fallback.
    assert result.initialization_method == "geometrical_center"

    # 2. Registered BL CT uses the FU voxel grid.
    registered_bl = result.registered_baseline_ct
    assert registered_bl.data.shape == fu_ct.data.shape

    # 3. Registered BL CT uses the FU affine geometry.
    np.testing.assert_allclose(
        registered_bl.metadata.affine,
        fu_ct.metadata.affine,
        rtol=0.0,
        atol=1e-3,
    )

    # The current registration implementation also exposes a body-overlap
    # quality check. The synthetic pair should align very closely.
    assert result.body_dice is not None
    assert result.body_dice >= 0.90

    # 4. Exactly one rigid transform file must be generated.
    assert len(result.transform_parameter_paths) == 1
    transform_path = result.transform_parameter_paths[0]
    assert transform_path.is_file()

    transform_text = transform_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    assert "(Transform " in transform_text
    assert "(TransformParameters " in transform_text
    assert "(Size " in transform_text
    assert "(Spacing " in transform_text
    assert "(Origin " in transform_text
    assert "(Direction " in transform_text

    # 5. A downstream lesion mask can use the same FU reference space.
    registered_mask_path = warp_baseline_mask_to_fu(
        bl_mask,
        transform_path=transform_path,
        registered_mask_path=tmp_path / "registered_bl_mask.nii.gz",
        force=True,
    )
    assert registered_mask_path.is_file()

    registered_mask = load_nifti_volume(
        registered_mask_path,
        role="registered BL lesion mask",
        preserve_dtype=True,
    )

    assert registered_mask.data.shape == fu_mask.data.shape
    np.testing.assert_allclose(
        registered_mask.metadata.affine,
        fu_mask.metadata.affine,
        rtol=0.0,
        atol=1e-3,
    )

    # Nearest-neighbour mask resampling should preserve discrete labels.
    assert set(np.unique(registered_mask.data)).issubset({0, 1})

    # The transformed BL lesion should land on the FU lesion.
    assert _dice(registered_mask.data > 0, fu_mask.data > 0) >= 0.90


def test_registration_failure_has_understandable_error(tmp_path: Path) -> None:
    """
    Verify the failure-path acceptance criterion: registration errors should
    identify what failed rather than exposing only a low-level library error.
    """
    bl_ct, fu_ct, _, _ = _make_synthetic_case(tmp_path)

    missing_path = tmp_path / "missing_baseline_ct.nii.gz"
    broken_bl = replace(
        bl_ct,
        metadata=replace(bl_ct.metadata, path=missing_path),
    )

    with pytest.raises(
        RegistrationError,
        match=r"Baseline CT source file does not exist",
    ):
        register_ct_pair(
            broken_bl,
            fu_ct,
            patient_id="functional_registration_failure",
            output_dir=tmp_path / "registration_failure",
            stages=("rigid",),
        )
