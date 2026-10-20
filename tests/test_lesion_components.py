from __future__ import annotations
from types import SimpleNamespace

import numpy as np
import pytest

from src.lesion_components import (
    EmptyLesionMaskError,
    LesionMaskError,
    _temporary_components,
    component_structure,
    extract_individual_lesions,
    lesion_rows,
    validate_lesion_mask,
)

def make_volume(
    data: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    affine: np.ndarray | None = None,
) -> SimpleNamespace:
    """Build the small part of LoadedVolume used by this module."""
    metadata = SimpleNamespace(
        spacing_mm=spacing,
        affine=np.eye(4) if affine is None else affine,
    )
    return SimpleNamespace(data=data, metadata=metadata)

@pytest.mark.parametrize(
    ("connectivity", "neighbour_count"),
    [(6, 7), (18, 19), (26, 27)],
)
def test_component_structure_uses_expected_neighbourhood(
    connectivity: int, neighbour_count: int
) -> None:
    structure = component_structure(connectivity)
    assert structure.shape == (3, 3, 3)
    assert int(structure.sum()) == neighbour_count

def test_component_structure_rejects_unsupported_connectivity() -> None:
    with pytest.raises(ValueError, match="6, 18, or 26"):
        component_structure(12)

def test_validate_mask_accepts_boolean_and_integer_valued_float_masks() -> None:
    boolean_mask = np.zeros((2, 2, 2), dtype=bool)
    boolean_mask[0, 0, 0] = True
    float_mask = boolean_mask.astype(np.float32) * 2.0
    boolean_result = validate_lesion_mask(boolean_mask)
    float_result = validate_lesion_mask(float_mask)

    assert boolean_result.dtype == np.int32
    assert float_result.dtype == np.int32
    np.testing.assert_array_equal(float_result, float_mask)

@pytest.mark.parametrize(
    ("mask", "error", "message"),
    [
        (np.zeros((2, 2)), LesionMaskError, "3-D"),
        (np.full((2, 2, 2), "lesion"), LesionMaskError, "numeric labels"),
        (np.array([[[np.nan]]]), LesionMaskError, "NaN or infinite"),
        (np.array([[[-1]]]), LesionMaskError, "negative labels"),
        (np.array([[[0.25]]]), LesionMaskError, "probability map"),
        (
            np.array([[[np.iinfo(np.int32).max + 1]]], dtype=np.int64),
            LesionMaskError,
            "int32 range",
        ),
        (np.zeros((2, 2, 2)), EmptyLesionMaskError, "empty"),
    ],
)
def test_validate_mask_rejects_invalid_input(
    mask: np.ndarray, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        validate_lesion_mask(mask)

def test_connectivity_changes_whether_corner_touching_voxels_merge() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.int32)
    mask[0, 0, 0] = 1
    mask[1, 1, 1] = 1
    _, mapping_6 = _temporary_components(mask, connectivity=6)
    labels_26, mapping_26 = _temporary_components(mask, connectivity=26)

    assert mapping_6 == [(1, 1), (2, 1)]
    assert mapping_26 == [(1, 1)]
    assert labels_26[0, 0, 0] == labels_26[1, 1, 1] == 1

def test_touching_source_labels_are_kept_as_separate_lesions() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.int32)
    mask[1, 1, 1] = 1
    mask[1, 1, 2] = 2

    labels, mapping = _temporary_components(mask, connectivity=26)
    assert mapping == [(1, 1), (2, 2)]
    assert labels[1, 1, 1] == 1
    assert labels[1, 1, 2] == 2

def test_extract_individual_lesions_calculates_geometry_and_ids() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.int16)
    mask[1, 1, 1] = 3
    mask[1, 1, 2] = 3
    mask[4, 4, 4] = 3
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, -5.0],
            [0.0, 0.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    volume = make_volume(mask, spacing=(2.0, 3.0, 4.0), affine=affine)
    result = extract_individual_lesions(
        volume, patient_id="patient_7", timepoint="bl", connectivity=6
    )

    assert result.patient_id == "patient_7"
    assert result.timepoint == "BL"
    assert result.connectivity == 6
    assert result.lesion_count == 2
    assert [lesion.temporary_id for lesion in result.lesions] == [
        "patient_7_BL_L001",
        "patient_7_BL_L002",
    ]

    first, second = result.lesions
    assert first.source_label == second.source_label == 3
    assert first.voxel_count == 2
    assert first.volume_ml == pytest.approx(0.048)
    assert first.centroid_voxel == pytest.approx((1.0, 1.0, 1.5))
    #affine included voxel scaling and world-space translation
    assert first.centroid_world_mm == pytest.approx((12.0, -2.0, 7.0))
    assert second.voxel_count == 1
    assert second.volume_ml == pytest.approx(0.024)
    assert set(np.unique(result.labelled_mask)) == {0, 1, 2}

def test_lesion_rows_flattens_multiple_results() -> None:
    baseline_mask = np.zeros((2, 2, 2), dtype=np.uint8)
    baseline_mask[0, 0, 0] = 1
    followup_mask = np.zeros((2, 2, 2), dtype=np.uint8)
    followup_mask[1, 1, 1] = 2

    baseline = extract_individual_lesions(
        make_volume(baseline_mask), patient_id="p1", timepoint="BL"
    )
    followup = extract_individual_lesions(
        make_volume(followup_mask), patient_id="p1", timepoint="FU"
    )
    rows = lesion_rows([baseline, followup])

    assert len(rows) == 2
    assert rows[0] == {
        "patient_id": "p1",
        "timepoint": "BL",
        "temporary_lesion_id": "p1_BL_L001",
        "component_index": 1,
        "source_label": 1,
        "voxel_count": 1,
        "volume_ml": 0.001,
        "centroid_x_vox": 0.0,
        "centroid_y_vox": 0.0,
        "centroid_z_vox": 0.0,
        "centroid_x_mm": 0.0,
        "centroid_y_mm": 0.0,
        "centroid_z_mm": 0.0,
        "connectivity": 18,
    }
    assert rows[1]["timepoint"] == "FU"
    assert rows[1]["source_label"] == 2


def test_lesion_rows_returns_empty_list_when_there_are_no_results() -> None:
    assert lesion_rows([]) == []
