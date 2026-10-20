from types import SimpleNamespace
import numpy as np
import pytest
from src import correspondence_visualization as cv
def location(**changes):
    values = {
        "mask_path": None,
        "component_index": 2,
        "mask_role": "regular",
        "timepoint": "FU",
        "connectivity": 18,
        "centroid_voxel": (1.0, 1.5, 1.0),
        "lesion_id": "L2",
    }
    values.update(changes)
    return SimpleNamespace(**values)

def test_selected_mask_loads_and_extracts_component(monkeypatch):
    raw = SimpleNamespace(data=np.zeros((3, 3, 3)))
    labels = np.zeros((3, 3, 3), dtype=int)
    labels[1, 1:3, 1] = 2
    monkeypatch.setattr(cv, "load_nifti_volume", lambda *args, **kwargs: raw)
    extract = SimpleNamespace(labelled_mask=labels)
    monkeypatch.setattr(cv, "extract_individual_lesions", lambda *args, **kwargs: extract)
    monkeypatch.setattr(cv, "validate_image_mask_alignment", lambda *args: None)

    result = cv.selected_lesion_mask(location(mask_path="mask.nii.gz"), object())

    np.testing.assert_array_equal(result, labels == 2)

def test_native_component_ids_skip_extraction(monkeypatch):
    labels = np.zeros((2, 2, 2), dtype=int)
    labels[1, 1, 1] = 4
    fallback = SimpleNamespace(data=labels)
    #native label are already component ids, so no re-label is needed
    def unexpected_call(*args, **kwargs):
        pytest.fail("extract_individual_lesions should not be called")

    monkeypatch.setattr(cv, "extract_individual_lesions", unexpected_call)
    monkeypatch.setattr(cv, "validate_image_mask_alignment", lambda *args: None)
    item = location(
        component_index=4,
        centroid_voxel=(1, 1, 1),
        mask_role="BL_native_component_IDs_rigidly_resampled_to_FU",
    )

    result = cv.selected_lesion_mask(item, object(), fallback)
    np.testing.assert_array_equal(result, labels == 4)


def test_missing_mask_raises_error():
    with pytest.raises(ValueError, match="No segmentation mask"):
        cv.selected_lesion_mask(location(), object())

def test_missing_component_index_raises_error(monkeypatch):
    fallback = SimpleNamespace(data=np.zeros((2, 2, 2)))
    monkeypatch.setattr(cv, "validate_image_mask_alignment", lambda *args: None)
    with pytest.raises(ValueError, match="component_index"):
        cv.selected_lesion_mask(location(component_index=None), object(), fallback)


def test_centroid_mismatch_raises_error(monkeypatch):
    labels = np.zeros((2, 2, 2), dtype=int)
    labels[0, 0, 0] = 2
    fallback = SimpleNamespace(data=labels)
    extracted = SimpleNamespace(labelled_mask=labels)
    monkeypatch.setattr(cv, "extract_individual_lesions", lambda *args, **kwargs: extracted)
    monkeypatch.setattr(cv, "validate_image_mask_alignment", lambda *args: None)
    with pytest.raises(ValueError, match="centroid disagree"):
        cv.selected_lesion_mask(
            location(centroid_voxel=(1, 1, 1)), object(), fallback
        )

def test_lesion_display_slice(monkeypatch):
    mask = np.zeros((2, 3, 4), dtype=bool)
    shown = np.ones((2, 3), dtype=bool)
    monkeypatch.setattr(cv, "select_lesion_slice", lambda mask, axis: 2)
    monkeypatch.setattr(
        cv, "extract_display_slice", lambda mask, axis, index: shown
    )
    
    result, index = cv.lesion_display_slice(mask, axis=1)
    assert result is shown
    assert index == 2
