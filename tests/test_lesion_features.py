from __future__ import annotations
import csv
from dataclasses import replace
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
from src.cohort_a_loading import LoadedVolume, VolumeMetadata
from src.lesion_features import (
    FEATURE_COLUMNS,
    _check_geometry,
    export_lesion_features,
    extract_aligned_lesion_features,
)

def make_volume(
    data: np.ndarray,
    *,
    affine: np.ndarray | None = None,
    units: str = "mm",
    name: str = "volume.nii.gz",
) -> LoadedVolume:
    """Create a small in-memory volume for testing."""
    data = np.asarray(data)
    affine = np.eye(4) if affine is None else np.asarray(affine, dtype=float)
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units(units)
    metadata = VolumeMetadata(
        path=Path(name),
        shape=tuple(int(value) for value in data.shape),
        spacing_mm=tuple(float(value) for value in image.header.get_zooms()[:3]),
        affine=affine,
        orientation=tuple(str(value) for value in nib.aff2axcodes(affine)),
        dtype=str(data.dtype),
    )
    return LoadedVolume(data=data, image=image, metadata=metadata)

def make_masks() -> tuple[LoadedVolume, LoadedVolume, LoadedVolume]:
    shape = (5, 5, 5)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    reference = make_volume(np.zeros(shape), affine=affine, name="fu_ct.nii.gz")
    baseline = np.zeros(shape, dtype=np.uint8)
    baseline[1:3, 1, 1] = 1
    followup = np.zeros(shape, dtype=np.uint8)
    followup[2, 2, 2] = 1
    return (
        make_volume(baseline, affine=affine, name="bl_mask.nii.gz"),
        make_volume(followup, affine=affine, name="fu_mask.nii.gz"),
        reference,
    )


def test_check_geometry_accepts_matching_mm_volume() -> None:
    reference = make_volume(np.zeros((3, 3, 3)))
    volume = make_volume(np.ones((3, 3, 3)))

    _check_geometry(volume, reference, "mask")

@pytest.mark.parametrize(
    ("data", "affine"),
    [
        (np.zeros((3, 3)), np.eye(4)),
        (np.zeros((3, 3, 3)), np.zeros((4, 4))),
        (np.zeros((3, 3, 3)), np.diag([1.0, 1.0, 1.0, 2.0])),
    ],
)
def test_check_geometry_rejects_invalid_spatial_geometry(
    data: np.ndarray,
    affine: np.ndarray,
) -> None:
    reference = make_volume(np.zeros((3, 3, 3)))
    volume = make_volume(data)
    volume = replace(volume, metadata=replace(volume.metadata, affine=affine))
    with pytest.raises(ValueError, match="valid 3-D spatial geometry"):
        _check_geometry(volume, reference, "mask")

def test_check_geometry_rejects_a_different_reference_grid() -> None:
    reference = make_volume(np.zeros((3, 3, 3)))
    shifted_affine = np.eye(4)
    shifted_affine[0, 3] = 1.0
    volume = make_volume(np.zeros((3, 3, 3)), affine=shifted_affine)
    with pytest.raises(ValueError, match="resampled onto the FU reference grid"):
        _check_geometry(volume, reference, "BL mask")


def test_check_geometry_rejects_non_mm_units() -> None:
    reference = make_volume(np.zeros((3, 3, 3)))
    volume = make_volume(np.zeros((3, 3, 3)), units="micron")

    with pytest.raises(ValueError, match="spatial units must be mm"):
        _check_geometry(volume, reference, "mask")

def test_extract_features_requires_patient_id() -> None:
    baseline, followup, reference = make_masks()
    with pytest.raises(ValueError, match="patient_id is required"):
        extract_aligned_lesion_features(
            baseline,
            followup,
            reference=reference,
            patient_id="   ",
        )

def test_extract_features_calculates_geometry_without_pet() -> None:
    baseline, followup, reference = make_masks()
    rows = extract_aligned_lesion_features(
        baseline,
        followup,
        reference=reference,
        patient_id="patient_1",
        connectivity=6,
    )

    assert len(rows) == 2
    baseline_row = rows[0]
    assert baseline_row["lesion_id"] == "patient_1_BL_L001"
    assert baseline_row["voxel_count"] == 2
    #2 mm isotropic voxel occupy 0.008 mL
    assert baseline_row["volume_ml"] == pytest.approx(0.016)
    assert baseline_row["coordinate_space"] == "FU_RAS_mm"
    assert baseline_row["pet_units"] is None
    assert baseline_row["pet_mean"] is None
    assert baseline_row["pet_max"] is None


def test_extract_features_calculates_pet_and_suv_values() -> None:
    baseline, followup, reference = make_masks()
    baseline_pet_data = np.zeros(reference.data.shape, dtype=np.float32)
    baseline_pet_data[1, 1, 1] = 2.0
    baseline_pet_data[2, 1, 1] = 6.0
    baseline_pet = make_volume(
        baseline_pet_data,
        affine=reference.metadata.affine,
        name="bl_pet.nii.gz",
    )
    rows = extract_aligned_lesion_features(
        baseline,
        followup,
        reference=reference,
        patient_id="patient_1",
        baseline_pet=baseline_pet,
        baseline_pet_units="suv",
        connectivity=6,
    )

    baseline_row = next(row for row in rows if row["timepoint"] == "BL")
    assert baseline_row["pet_units"] == "suv"
    assert baseline_row["pet_mean"] == pytest.approx(4.0)
    assert baseline_row["pet_max"] == pytest.approx(6.0)
    assert baseline_row["suvmean"] == pytest.approx(4.0)
    assert baseline_row["suvmax"] == pytest.approx(6.0)
def test_non_suv_pet_does_not_fill_suv_fields() -> None:
    baseline, followup, reference = make_masks()
    pet = make_volume(
        np.ones(reference.data.shape),
        affine=reference.metadata.affine,
    )
    rows = extract_aligned_lesion_features(
        baseline,
        followup,
        reference=reference,
        patient_id="patient_1",
        baseline_pet=pet,
        baseline_pet_units="Bq/mL",
    )

    baseline_row = next(row for row in rows if row["timepoint"] == "BL")
    assert baseline_row["pet_mean"] == pytest.approx(1.0)
    assert baseline_row["suvmean"] is None
    assert baseline_row["suvmax"] is None


def test_extract_features_rejects_non_finite_pet_values_in_a_lesion() -> None:
    baseline, followup, reference = make_masks()
    pet_data = np.ones(reference.data.shape)
    pet_data[1, 1, 1] = np.nan
    pet = make_volume(pet_data, affine=reference.metadata.affine)
    with pytest.raises(ValueError, match="PET contains non-finite lesion values"):
        extract_aligned_lesion_features(
            baseline,
            followup,
            reference=reference,
            patient_id="patient_1",
            baseline_pet=pet,
        )


def test_empty_masks_return_no_rows() -> None:
    _, _, reference = make_masks()
    empty = make_volume(
        np.zeros(reference.data.shape, dtype=np.uint8),
        affine=reference.metadata.affine,
    )
    rows = extract_aligned_lesion_features(
        empty,
        empty,
        reference=reference,
        patient_id="patient_1",
    )
    assert rows == []

def test_export_features_creates_parent_directory_and_csv(tmp_path: Path) -> None:
    row = {column: None for column in FEATURE_COLUMNS}
    row.update(patient_id="patient_1", timepoint="BL", lesion_id="patient_1_BL_L001")
    output = tmp_path / "nested" / "lesion_features.csv"
    export_lesion_features([row], output)

    with output.open(newline="", encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert output.is_file()
    assert list(exported[0]) == FEATURE_COLUMNS
    assert exported[0]["lesion_id"] == "patient_1_BL_L001"
    assert exported[0]["pet_mean"] == ""
