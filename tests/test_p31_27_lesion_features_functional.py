"""P31-27 functional tests: lesion extraction and aligned feature generation.

These tests use small deterministic synthetic volumes so every expected lesion,
centroid and volume is known in advance.  No project dataset is required.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.cohort_a_loading import LoadedVolume, VolumeMetadata
from src.lesion_components import LesionMaskError
from src.lesion_features import extract_aligned_lesion_features, export_lesion_features


def _volume(data: np.ndarray, affine: np.ndarray, name: str) -> LoadedVolume:
    data = np.asarray(data)
    spacing = tuple(float(abs(affine[i, i])) for i in range(3))
    # extract_aligned_lesion_features only needs the spatial-unit method from
    # the image header.  Keeping the test in-memory makes it fast and stable.
    header = SimpleNamespace(get_xyzt_units=lambda: ("mm", "unknown"))
    image = SimpleNamespace(
        header=header,
        affine=np.asarray(affine, dtype=float),
        shape=data.shape,
        get_data_dtype=lambda: data.dtype,
    )
    metadata = VolumeMetadata(
        path=Path(name),
        shape=tuple(int(v) for v in data.shape),
        spacing_mm=spacing,
        affine=np.asarray(affine, dtype=float),
        orientation=("R", "A", "S"),
        dtype=str(data.dtype),
    )
    return LoadedVolume(data=data, image=image, metadata=metadata)


def _synthetic_case() -> tuple[LoadedVolume, LoadedVolume, LoadedVolume]:
    shape = (8, 8, 8)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    reference = _volume(np.zeros(shape, dtype=np.float32), affine, "fu_ct.nii.gz")

    bl = np.zeros(shape, dtype=np.uint8)
    bl[1:3, 1:3, 1:3] = 1       # 8 voxels, centroid (1.5, 1.5, 1.5)
    bl[5, 5, 5] = 1             # 1 voxel

    fu = np.zeros(shape, dtype=np.uint8)
    fu[2:4, 1:3, 1:3] = 1       # 8 voxels
    fu[5:7, 5, 5] = 1           # 2 voxels
    fu[1, 6, 6] = 1             # 1 voxel

    return (
        _volume(bl, affine, "registered_bl_mask.nii.gz"),
        _volume(fu, affine, "fu_mask.nii.gz"),
        reference,
    )


def test_p31_27_extracts_one_feature_record_per_connected_lesion() -> None:
    bl_mask, fu_mask, reference = _synthetic_case()

    rows = extract_aligned_lesion_features(
        bl_mask,
        fu_mask,
        reference=reference,
        patient_id="patient_test",
        connectivity=6,
    )
    table = pd.DataFrame(rows)

    # BL has two components and FU has three: one output row per lesion.
    assert len(table) == 5
    assert (table["timepoint"] == "BL").sum() == 2
    assert (table["timepoint"] == "FU").sum() == 3

    # IDs must be unique and timepoint-specific.
    assert table["lesion_id"].is_unique
    assert set(table.loc[table.timepoint == "BL", "lesion_id"]) == {
        "patient_test_BL_L001",
        "patient_test_BL_L002",
    }
    assert set(table.loc[table.timepoint == "FU", "lesion_id"]) == {
        "patient_test_FU_L001",
        "patient_test_FU_L002",
        "patient_test_FU_L003",
    }

    # Centroid and volume are checked against known synthetic geometry.
    # 2 mm isotropic voxels = 8 mm^3 = 0.008 mL each.
    bl_first = table.loc[table.lesion_id == "patient_test_BL_L001"].iloc[0]
    assert bl_first["centroid_x_vox"] == pytest.approx(1.5)
    assert bl_first["centroid_y_vox"] == pytest.approx(1.5)
    assert bl_first["centroid_z_vox"] == pytest.approx(1.5)
    assert bl_first["centroid_x_mm"] == pytest.approx(3.0)
    assert bl_first["centroid_y_mm"] == pytest.approx(3.0)
    assert bl_first["centroid_z_mm"] == pytest.approx(3.0)
    assert bl_first["volume_ml"] == pytest.approx(8 * 0.008)
    assert set(table["coordinate_space"]) == {"FU_RAS_mm"}


def test_p31_27_exports_bl_fu_features_for_downstream_processing(tmp_path: Path) -> None:
    bl_mask, fu_mask, reference = _synthetic_case()
    rows = extract_aligned_lesion_features(
        bl_mask,
        fu_mask,
        reference=reference,
        patient_id="patient_test",
        connectivity=6,
    )

    output = tmp_path / "aligned_lesion_features.csv"
    export_lesion_features(rows, output)

    assert output.is_file()
    exported = pd.read_csv(output)
    assert len(exported) == 5
    assert {"patient_id", "timepoint", "lesion_id", "volume_ml"}.issubset(exported.columns)
    assert {"centroid_x_mm", "centroid_y_mm", "centroid_z_mm"}.issubset(exported.columns)
    assert set(exported["timepoint"]) == {"BL", "FU"}


def test_p31_27_empty_mask_is_handled_without_fabricating_lesions() -> None:
    bl_mask, fu_mask, reference = _synthetic_case()
    empty_bl = _volume(
        np.zeros_like(bl_mask.data),
        bl_mask.metadata.affine,
        "empty_bl_mask.nii.gz",
    )

    rows = extract_aligned_lesion_features(
        empty_bl,
        fu_mask,
        reference=reference,
        patient_id="patient_test",
        connectivity=6,
    )

    # Empty BL contributes no fake record; valid FU lesions are still exported.
    assert rows
    assert all(row["timepoint"] == "FU" for row in rows)
    assert len(rows) == 3


def test_p31_27_invalid_probability_mask_is_rejected_clearly() -> None:
    bl_mask, fu_mask, reference = _synthetic_case()
    invalid = np.zeros_like(bl_mask.data, dtype=np.float32)
    invalid[1, 1, 1] = 0.4
    invalid_bl = _volume(invalid, bl_mask.metadata.affine, "invalid_bl_mask.nii.gz")

    with pytest.raises(LesionMaskError, match="probability map"):
        extract_aligned_lesion_features(
            invalid_bl,
            fu_mask,
            reference=reference,
            patient_id="patient_test",
            connectivity=6,
        )
