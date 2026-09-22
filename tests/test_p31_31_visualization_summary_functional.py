"""P31-31 functional tests: visualisation support and tracking summaries."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.cohort_a_loading import LoadedVolume, VolumeMetadata
from src.correspondence_visualization import OUTCOME_STYLES
from src.matching_dashboard import resolve_match_image_focus
from src.multiplanar_view import extract_display_slice, plane_info
from src.tracking_summary import summarise_patient_tracking
from tools.align_longitudinal_patient_v5_mapped import _overlay_mask


def _volume(shape: tuple[int, int, int] = (8, 10, 12)) -> LoadedVolume:
    affine = np.diag([1.0, 1.5, 2.0, 1.0])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    header = SimpleNamespace(get_xyzt_units=lambda: ("mm", "unknown"))
    image = SimpleNamespace(
        header=header,
        affine=affine,
        shape=shape,
        get_data_dtype=lambda: data.dtype,
    )
    metadata = VolumeMetadata(
        path=Path("synthetic_ct.nii.gz"),
        shape=shape,
        spacing_mm=(1.0, 1.5, 2.0),
        affine=affine,
        orientation=("R", "A", "S"),
        dtype=str(data.dtype),
    )
    return LoadedVolume(data=data, image=image, metadata=metadata)


def test_p31_31_axial_coronal_and_sagittal_views_are_extractable() -> None:
    volume = _volume()
    expected_axes = {"Axial": 2, "Coronal": 1, "Sagittal": 0}

    for plane, expected_axis in expected_axes.items():
        info = plane_info(volume, plane)
        assert info.voxel_axis == expected_axis
        index = info.slice_count // 2
        image = extract_display_slice(volume.data, axis=info.voxel_axis, index=index)
        assert image.ndim == 2
        assert image.size > 0


def test_p31_31_lesion_mask_overlay_is_applied_to_corresponding_pixels() -> None:
    grayscale = np.full((12, 12), 100, dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 5:9] = True

    rendered = _overlay_mask(grayscale, mask)

    assert rendered.shape == (12, 12, 3)
    assert rendered.dtype == np.uint8
    # Outside the mask the grayscale RGB value is preserved.
    assert np.array_equal(rendered[0, 0], np.array([100, 100, 100], dtype=np.uint8))
    # Inside the mask at least one channel must change because of the overlay.
    assert not np.array_equal(rendered[5, 6], np.array([100, 100, 100], dtype=np.uint8))


def _features() -> pd.DataFrame:
    rows = []
    for tp, ids in (("BL", ["b1", "b2", "b3", "b4"]), ("FU", ["f1", "f2", "f3"])):
        for index, lesion_id in enumerate(ids, start=1):
            rows.append(
                {
                    "patient_id": "p",
                    "timepoint": tp,
                    "lesion_id": lesion_id,
                    "centroid_x_vox": float(index),
                    "centroid_y_vox": float(index + 1),
                    "centroid_z_vox": float(index + 2),
                    "coordinate_space": "FU_RAS_mm",
                }
            )
    return pd.DataFrame(rows)


def test_p31_31_match_results_link_to_bl_fu_locations_for_all_outcomes() -> None:
    features = _features()
    cases = [
        ({"bl_lesion_id": "b1", "fu_lesion_id": "f1", "match_type": "MATCHED"}, True, True),
        ({"bl_lesion_id": "b2", "fu_lesion_id": "f2", "match_type": "MERGING"}, True, True),
        ({"bl_lesion_id": None, "fu_lesion_id": "f3", "match_type": "NEW"}, False, True),
        ({"bl_lesion_id": "b4", "fu_lesion_id": None, "match_type": "DISAPPEARING"}, True, False),
    ]

    for raw_match, has_bl, has_fu in cases:
        focus = resolve_match_image_focus(pd.Series(raw_match), features)
        assert (focus.baseline is not None) is has_bl
        assert (focus.followup is not None) is has_fu
        assert focus.match_type == raw_match["match_type"]


def test_p31_31_dashboard_has_explicit_styles_for_all_required_outcomes() -> None:
    required = {"MATCHED", "NEW", "DISAPPEARING", "MERGING"}
    assert required.issubset(OUTCOME_STYLES)
    for outcome in required:
        label, colour = OUTCOME_STYLES[outcome]
        assert label.strip()
        assert len(colour) == 3


def test_p31_31_tracking_summary_reports_correct_counts() -> None:
    matches = pd.DataFrame(
        [
            {"patient_id": "p", "bl_lesion_id": "b1", "fu_lesion_id": "f1", "match_type": "MATCHED"},
            {"patient_id": "p", "bl_lesion_id": "b2", "fu_lesion_id": "f2", "match_type": "MERGING"},
            {"patient_id": "p", "bl_lesion_id": "b3", "fu_lesion_id": "f2", "match_type": "MERGING"},
            {"patient_id": "p", "bl_lesion_id": "b4", "fu_lesion_id": None, "match_type": "DISAPPEARING"},
            {"patient_id": "p", "bl_lesion_id": None, "fu_lesion_id": "f3", "match_type": "NEW"},
        ]
    )

    summary = summarise_patient_tracking(matches, "p")

    assert summary["bl_lesions"] == 4
    assert summary["fu_lesions"] == 3
    assert summary["matched_lesions"] == 3
    assert summary["matched_fu_lesions"] == 2
    assert summary["appearing_lesions"] == 1
    assert summary["disappearing_lesions"] == 1
    assert summary["unmatched_lesions"] == 2
    assert summary["merging_links"] == 2
    assert summary["merged_fu_lesions"] == 1
