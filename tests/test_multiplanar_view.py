from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.cohort_a_loading import LoadedVolume, VolumeMetadata
from src.multiplanar_view import (
    MultiplanarViewError,
    anatomical_axis,
    display_pixel_spacing_mm,
    extract_display_slice,
    physical_aspect_resize,
    plane_info,
    slice_centre_world_mm,
)


def _volume(
    shape=(4, 5, 6),
    orientation=("R", "A", "S"),
    spacing_mm=(1.0, 1.0, 1.0),
) -> LoadedVolume:
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    affine = np.eye(4, dtype=float)
    image = nib.Nifti1Image(data, affine)
    metadata = VolumeMetadata(
        path=Path("synthetic.nii.gz"),
        shape=tuple(shape),
        spacing_mm=tuple(float(v) for v in spacing_mm),
        affine=affine,
        orientation=orientation,
        dtype="float32",
    )
    return LoadedVolume(data=data, image=image, metadata=metadata)


def test_named_planes_use_anatomical_orientation_codes():
    assert anatomical_axis(("R", "A", "S"), "Sagittal") == 0
    assert anatomical_axis(("R", "A", "S"), "Coronal") == 1
    assert anatomical_axis(("R", "A", "S"), "Axial") == 2


    assert anatomical_axis(("S", "R", "P"), "Axial") == 0
    assert anatomical_axis(("S", "R", "P"), "Sagittal") == 1
    assert anatomical_axis(("S", "R", "P"), "Coronal") == 2


def test_plane_info_changes_slice_count_with_selected_view():
    volume = _volume((4, 5, 6))
    assert plane_info(volume, "Sagittal").slice_count == 4
    assert plane_info(volume, "Coronal").slice_count == 5
    assert plane_info(volume, "Axial").slice_count == 6


def test_extract_display_slice_uses_selected_axis():
    data = np.arange(4 * 5 * 6).reshape((4, 5, 6))

    sagittal = extract_display_slice(data, axis=0, index=2)
    coronal = extract_display_slice(data, axis=1, index=3)
    axial = extract_display_slice(data, axis=2, index=4)

    assert sagittal.shape == (6, 5)
    assert coronal.shape == (6, 4)
    assert axial.shape == (5, 4)




def test_extract_display_slice_handles_fortran_order_without_np_take(monkeypatch):
    # NiBabel commonly exposes Fortran-contiguous NIfTI arrays.  The viewer must
    # use basic slicing rather than np.take, which is extremely slow on large
    # F-order 3-D arrays.
    data = np.asfortranarray(np.arange(8 * 9 * 10).reshape((8, 9, 10)))

    def _forbidden_take(*args, **kwargs):
        raise AssertionError("extract_display_slice must not call np.take")

    monkeypatch.setattr(np, "take", _forbidden_take)
    axial = extract_display_slice(data, axis=2, index=4)
    expected = np.ascontiguousarray(np.flipud(np.rot90(data[:, :, 4])))
    np.testing.assert_array_equal(axial, expected)
    assert axial.flags.c_contiguous


def test_slice_centre_world_tracks_selected_plane_index():
    volume = _volume((5, 7, 9))

    axial = slice_centre_world_mm(volume, plane="Axial", index=3)
    coronal = slice_centre_world_mm(volume, plane="Coronal", index=4)
    sagittal = slice_centre_world_mm(volume, plane="Sagittal", index=1)

    np.testing.assert_allclose(axial, [2.0, 3.0, 3.0])
    np.testing.assert_allclose(coronal, [2.0, 4.0, 4.0])
    np.testing.assert_allclose(sagittal, [1.0, 3.0, 4.0])


def test_invalid_plane_and_slice_are_rejected():
    volume = _volume()
    with pytest.raises(MultiplanarViewError):
        plane_info(volume, "Diagonal")
    with pytest.raises(IndexError):
        extract_display_slice(volume.data, axis=2, index=999)


def test_display_pixel_spacing_matches_rotated_slice_axes():
    volume = _volume(
        shape=(4, 5, 6),
        orientation=("R", "A", "S"),
        spacing_mm=(2.0, 3.0, 5.0),
    )


    assert display_pixel_spacing_mm(volume, "Sagittal") == (5.0, 3.0)
    assert display_pixel_spacing_mm(volume, "Coronal") == (5.0, 2.0)
    assert display_pixel_spacing_mm(volume, "Axial") == (3.0, 2.0)


def test_physical_aspect_resize_uses_mm_not_raw_pixel_shape():
    square_pixels = np.zeros((100, 100, 3), dtype=np.uint8)


    corrected = physical_aspect_resize(
        square_pixels,
        row_spacing_mm=2.0,
        column_spacing_mm=1.0,
        max_side=200,
    )

    assert corrected.shape == (200, 100, 3)
    assert corrected.dtype == np.uint8


def test_physical_aspect_resize_preserves_isotropic_aspect():
    image = np.zeros((120, 80), dtype=np.uint8)
    corrected = physical_aspect_resize(
        image,
        row_spacing_mm=1.5,
        column_spacing_mm=1.5,
        max_side=300,
    )


    assert corrected.shape == (300, 200)


def test_invalid_physical_spacing_is_rejected():
    image = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(MultiplanarViewError):
        physical_aspect_resize(
            image,
            row_spacing_mm=0.0,
            column_spacing_mm=1.0,
        )


def test_physical_aspect_fast_mode_does_not_upscale():
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    corrected = physical_aspect_resize(
        image,
        row_spacing_mm=2.0,
        column_spacing_mm=1.0,
        max_side=560,
        allow_upscale=False,
    )
    # Physical shape is square (200*2 mm == 400*1 mm), so only the taller
    # pixel dimension is reduced; no axis is inflated beyond the input box.
    assert corrected.shape == (200, 200, 3)


def test_physical_aspect_fast_mode_respects_render_cap():
    image = np.zeros((1000, 800, 3), dtype=np.uint8)
    corrected = physical_aspect_resize(
        image,
        row_spacing_mm=1.0,
        column_spacing_mm=1.0,
        max_side=560,
        allow_upscale=False,
    )
    assert max(corrected.shape[:2]) <= 560
    assert corrected.shape[0] / corrected.shape[1] == pytest.approx(1000 / 800, rel=0.01)
