from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage as ndi

from .cohort_a_loading import LoadedVolume, VolumeMetadata, load_nifti_volume


class MultiplanarViewError(ValueError):
    """Raised when a requested anatomical plane cannot be displayed safely."""


PLANE_NAMES = ("Axial", "Coronal", "Sagittal")

# An anatomical plane is identified by the world axis normal to that plane.
_PLANE_AXIS_CODES: Mapping[str, frozenset[str]] = {
    "Axial": frozenset({"S", "I"}),
    "Coronal": frozenset({"A", "P"}),
    "Sagittal": frozenset({"L", "R"}),
}


@dataclass(frozen=True)
class PlaneInfo:
    name: str
    voxel_axis: int
    slice_count: int


def normalise_plane(plane: str) -> str:
    text = str(plane).strip().lower()
    mapping = {name.lower(): name for name in PLANE_NAMES}
    try:
        return mapping[text]
    except KeyError as exc:
        raise MultiplanarViewError(
            "plane must be one of: " + ", ".join(PLANE_NAMES)
        ) from exc


def anatomical_axis(orientation: tuple[str, str, str], plane: str) -> int:
    
    #Make the voxel array axis normal to the anatomical plane.
    #This intentionally uses NIfTI orientation codes, rather than assuming that array axis 0/1/2 always means sagittal/coronal/axial. 
    #for effective volumes stored in different voxel axis sequences, it is still correct.

    plane_name = normalise_plane(plane)
    accepted = _PLANE_AXIS_CODES[plane_name]

    matches = [
        index
        for index, code in enumerate(tuple(str(v).upper() for v in orientation))
        if code in accepted
    ]
    if len(matches) != 1:
        raise MultiplanarViewError(
            f"Could not identify a unique {plane_name.lower()} axis from "
            f"orientation {orientation!r}."
        )
    return int(matches[0])


def plane_info(volume: LoadedVolume, plane: str) -> PlaneInfo:
    plane_name = normalise_plane(plane)
    axis = anatomical_axis(volume.metadata.orientation, plane_name)
    return PlaneInfo(
        name=plane_name,
        voxel_axis=axis,
        slice_count=int(volume.data.shape[axis]),
    )


def extract_display_slice(
    volume: np.ndarray,
    *,
    axis: int,
    index: int,
) -> np.ndarray:
    #Extract one 2-D slice and apply the dashboard's display orientation.
    data = np.asarray(volume)
    if data.ndim != 3:
        raise MultiplanarViewError(f"Expected a 3-D volume, got {data.shape}.")
    if axis not in (0, 1, 2):
        raise MultiplanarViewError("axis must be 0, 1, or 2")
    if not 0 <= int(index) < data.shape[axis]:
        raise IndexError(
            f"Slice index {index} is outside [0, {data.shape[axis] - 1}] "
            f"for axis {axis}."
        )

    raw_slice = np.take(data, int(index), axis=axis)
    return np.ascontiguousarray(np.flipud(np.rot90(raw_slice)))




def display_pixel_spacing_mm(
    volume: LoadedVolume,
    plane: str,
) -> tuple[float, float]:
    #Return ``(row_mm, column_mm)`` for a displayed anatomical slice.
    #Extract a local 2D slice and apply rot90 and flippud to the dashboard. 
    #Rotation swaps the voxel axes in two planes, so the displayed row spacing is the spacing between the second remaining voxel axis, 
    #and the displayed column spacing is the spacing between the first remaining voxel axis.
    #Keeping this calculation next to the slice direction assist program can prevent the Streamlit viewer from treating anisotropic medical image voxels as square screen pixels.

    info = plane_info(volume, plane)
    spacing = tuple(float(abs(v)) for v in volume.metadata.spacing_mm)
    if len(spacing) != 3 or any((not np.isfinite(v) or v <= 0.0) for v in spacing):
        raise MultiplanarViewError(
            f"Invalid voxel spacing {volume.metadata.spacing_mm!r} for "
            f"{volume.metadata.path}."
        )

    remaining_axes = [axis for axis in range(3) if axis != info.voxel_axis]
    #np.rot90 swaps these two axes; flipud only changes direction/sign.
    row_mm = spacing[remaining_axes[1]]
    column_mm = spacing[remaining_axes[0]]
    return float(row_mm), float(column_mm)


def physical_aspect_resize(
    image: np.ndarray,
    *,
    row_spacing_mm: float,
    column_spacing_mm: float,
    max_side: int = 900,
) -> np.ndarray:
    #Resample a 2-D display image to preserve its physical millimetre aspect.
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise MultiplanarViewError(
            f"Expected a 2-D grayscale/RGB display image, got {arr.shape}."
        )
    if arr.ndim == 3 and arr.shape[2] not in (3, 4):
        raise MultiplanarViewError(
            f"Expected RGB/RGBA channels in the last dimension, got {arr.shape}."
        )

    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        return np.ascontiguousarray(arr)

    row_mm = float(row_spacing_mm)
    col_mm = float(column_spacing_mm)
    if (
        not np.isfinite(row_mm)
        or not np.isfinite(col_mm)
        or row_mm <= 0.0
        or col_mm <= 0.0
    ):
        raise MultiplanarViewError(
            "row_spacing_mm and column_spacing_mm must be positive finite values."
        )
    if int(max_side) <= 0:
        raise MultiplanarViewError("max_side must be a positive integer.")

    physical_h = h * row_mm
    physical_w = w * col_mm
    ratio = physical_h / physical_w

    if ratio >= 1.0:
        target_h = int(max_side)
        target_w = max(1, int(round(target_h / ratio)))
    else:
        target_w = int(max_side)
        target_h = max(1, int(round(target_w * ratio)))

    zoom_h = target_h / h
    zoom_w = target_w / w
    zoom = (zoom_h, zoom_w) if arr.ndim == 2 else (zoom_h, zoom_w, 1.0)

    resized = ndi.zoom(arr, zoom, order=1, mode="nearest", prefilter=False)

    #ndimage normally preserves dtype, but clamp/cast explicitly for the
    #uint8 RGB arrays used by the dashboard.
    if arr.dtype == np.uint8 and resized.dtype != np.uint8:
        resized = np.clip(np.rint(resized), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(resized)


def slice_centre_world_mm(
    volume: LoadedVolume,
    *,
    plane: str,
    index: int,
) -> np.ndarray:
    """Return RAS world coordinates of the centre voxel on a selected plane."""
    info = plane_info(volume, plane)
    if not 0 <= int(index) < info.slice_count:
        raise IndexError(
            f"Slice index {index} is outside [0, {info.slice_count - 1}] "
            f"for {info.name}."
        )

    voxel = np.asarray(
        [(size - 1) / 2.0 for size in volume.data.shape] + [1.0],
        dtype=float,
    )
    voxel[info.voxel_axis] = float(index)
    world = np.asarray(volume.metadata.affine, dtype=float) @ voxel
    return world[:3]


def geometry_matches(a: LoadedVolume, b: LoadedVolume, *, atol: float = 1e-3) -> bool:
    return (
        a.data.shape == b.data.shape
        and np.allclose(
            a.metadata.affine,
            b.metadata.affine,
            rtol=0.0,
            atol=atol,
        )
    )


def resample_to_reference(
    moving: LoadedVolume,
    reference: LoadedVolume,
    *,
    order: int = 1,
    output_path: str | Path | None = None,
    role: str = "resampled volume",
) -> LoadedVolume:
    
    #Resample a co-registered volume onto a reference NIfTI grid in world space.
    #This is used for resampling the PET ->CT grid within a certain time point. It does not estimate new registration conversions; 
    #It only uses the existing NIfTI affine to sample the motion images on the reference voxel grid.
    if order not in (0, 1, 2, 3, 4, 5):
        raise MultiplanarViewError("resampling order must be between 0 and 5")

    if geometry_matches(moving, reference):
        return moving

    try:
        result = resample_from_to(
            moving.image,
            (reference.data.shape, reference.metadata.affine),
            order=order,
        )
    except Exception as exc:
        raise MultiplanarViewError(
            f"Could not resample {moving.metadata.path.name} to the reference grid: {exc}"
        ) from exc

    if output_path is None:
        data = result.get_fdata(dtype=np.float32)
        zooms = result.header.get_zooms()
        metadata = VolumeMetadata(
            path=Path(f"<{role}>"),
            shape=tuple(int(v) for v in result.shape),
            spacing_mm=tuple(float(v) for v in zooms[:3]),
            affine=np.asarray(result.affine, dtype=float).copy(),
            orientation=tuple(str(v) for v in nib.aff2axcodes(result.affine)),  # type: ignore[arg-type]
            dtype=str(result.get_data_dtype()),
        )
        return LoadedVolume(
            data=np.asarray(data),
            image=result,
            metadata=metadata,
        )

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(path))
    return load_nifti_volume(path, role=role)
