from __future__ import annotations

import numpy as np
import pytest

from tools.align_longitudinal_patient_v4_mapped import _centroid_marker_slice


@pytest.mark.parametrize(
    ("axis", "expected_slice", "expected_shape"),
    [(0, 4, (30, 20)), (1, 8, (30, 10)), (2, 17, (20, 10))],
)
def test_centroid_marker_uses_selected_anatomical_axis(
    axis: int,
    expected_slice: int,
    expected_shape: tuple[int, int],
) -> None:
    marker, slice_index = _centroid_marker_slice(
        (10, 20, 30),
        (4.0, 8.0, 17.0),
        axis=axis,
        radius=2,
    )

    assert slice_index == expected_slice
    assert marker.shape == expected_shape
    assert marker.dtype == np.bool_
    assert marker.sum() > 1


def test_centroid_marker_rejects_location_outside_scan() -> None:
    with pytest.raises(ValueError, match="outside image shape"):
        _centroid_marker_slice((10, 10, 10), (12.0, 2.0, 3.0), axis=0)
