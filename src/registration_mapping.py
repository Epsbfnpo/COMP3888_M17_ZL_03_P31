"""Coordinate mapping helpers for longitudinal rigid registration.

This module converts FU-space slice locations back to the native baseline (BL)
grid by rebuilding the saved Elastix Euler transform.  It deliberately has no
Streamlit dependency so the same mapping can be reused by dashboards, batch
tools, and tests.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.multiplanar_view import anatomical_axis


def _ras_to_lps(point_xyz: np.ndarray) -> np.ndarray:
    """Convert a three-dimensional point from NIfTI RAS to ITK LPS."""
    point = np.asarray(point_xyz, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"Expected a 3-D RAS point, got shape {point.shape}.")
    return np.asarray((-point[0], -point[1], point[2]), dtype=float)


def _lps_to_ras(point_xyz: np.ndarray) -> np.ndarray:
    """Convert a three-dimensional point from ITK LPS to NIfTI RAS."""
    point = np.asarray(point_xyz, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"Expected a 3-D LPS point, got shape {point.shape}.")
    return np.asarray((-point[0], -point[1], point[2]), dtype=float)


def _parameter_value(
    parameter_map: Any,
    name: str,
    *,
    default: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Read one Elastix parameter and normalise its values to strings."""
    try:
        values = parameter_map[name]
    except Exception as exc:
        if default is not None:
            return default
        raise ValueError(
            f"Rigid transform parameter '{name}' is missing."
        ) from exc
    return tuple(str(value) for value in values)


def _build_elastix_euler_transform(transform_path: str | Path):
    """Rebuild a saved fixed-FU to moving-BL Elastix Euler transform."""
    path = Path(transform_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Rigid transform not found: {path}")

    try:
        import itk
    except ImportError as exc:
        raise RuntimeError(
            "ITKElastix is required to map FU positions back to Original BL."
        ) from exc

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(str(path))
    parameter_map = parameter_object.GetParameterMap(0)

    transform_name = _parameter_value(parameter_map, "Transform")[0].strip('"')
    if transform_name != "EulerTransform":
        raise ValueError(
            "Original-BL slice mapping currently supports only the rigid "
            f"EulerTransform, but the saved transform is {transform_name!r}."
        )

    initial = _parameter_value(
        parameter_map,
        "InitialTransformParameterFileName",
        default=("NoInitialTransform",),
    )[0].strip('"')
    if initial not in {
        "NoInitialTransform",
        "NoInitialTransformParameterFileName",
    }:
        raise ValueError(
            "The saved rigid transform references an additional initial "
            "transform. This mapper refuses to ignore a transform chain: "
            f"{initial}"
        )

    parameters = tuple(
        float(value)
        for value in _parameter_value(parameter_map, "TransformParameters")
    )
    if len(parameters) != 6:
        raise ValueError(
            "Expected six Euler rigid parameters (rx, ry, rz, tx, ty, tz); "
            f"found {len(parameters)}."
        )

    centre = tuple(
        float(value)
        for value in _parameter_value(parameter_map, "CenterOfRotationPoint")
    )
    if len(centre) != 3:
        raise ValueError(
            "Expected a 3-D CenterOfRotationPoint in the rigid transform."
        )

    compute_zyx = (
        _parameter_value(parameter_map, "ComputeZYX", default=("false",))[0]
        .strip('"')
        .lower()
        == "true"
    )

    rigid = itk.Euler3DTransform[itk.D].New()
    rigid.SetCenter(centre)
    rigid.SetComputeZYX(compute_zyx)
    rigid.SetRotation(parameters[0], parameters[1], parameters[2])
    rigid.SetTranslation((parameters[3], parameters[4], parameters[5]))
    return rigid


@lru_cache(maxsize=32)
def fu_to_bl_slice_map(
    *,
    bl_shape: tuple[int, int, int],
    bl_affine_flat: tuple[float, ...],
    bl_orientation: tuple[str, str, str],
    fu_shape: tuple[int, int, int],
    fu_affine_flat: tuple[float, ...],
    fu_orientation: tuple[str, str, str],
    plane_name: str,
    transform_path: str,
    transform_modified_ns: int,
) -> tuple[float, ...]:
    """Map every FU plane centre to a continuous native-BL slice index.

    ``transform_modified_ns`` is intentionally part of the cache key so a
    re-run registration invalidates the cached mapping even when its path is
    unchanged.
    """
    del transform_modified_ns

    if len(bl_shape) != 3 or any(int(size) <= 0 for size in bl_shape):
        raise ValueError(f"Expected a positive 3-D BL shape, got {bl_shape}.")
    if len(fu_shape) != 3 or any(int(size) <= 0 for size in fu_shape):
        raise ValueError(f"Expected a positive 3-D FU shape, got {fu_shape}.")

    bl_affine = np.asarray(bl_affine_flat, dtype=float)
    fu_affine = np.asarray(fu_affine_flat, dtype=float)
    if bl_affine.size != 16 or fu_affine.size != 16:
        raise ValueError("BL and FU affines must each contain 16 values.")
    bl_affine = bl_affine.reshape(4, 4)
    fu_affine = fu_affine.reshape(4, 4)

    bl_axis = anatomical_axis(bl_orientation, plane_name)
    fu_axis = anatomical_axis(fu_orientation, plane_name)
    try:
        bl_affine_inv = np.linalg.inv(bl_affine)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Original BL affine is not invertible.") from exc

    rigid = _build_elastix_euler_transform(transform_path)
    fu_centre = np.asarray(
        [(size - 1) / 2.0 for size in fu_shape] + [1.0],
        dtype=float,
    )
    mapped_axis: list[float] = []

    for slice_index in range(fu_shape[fu_axis]):
        fu_voxel = fu_centre.copy()
        fu_voxel[fu_axis] = float(slice_index)

        # nibabel uses RAS physical coordinates; Elastix/ITK uses LPS.
        fu_world_ras = (fu_affine @ fu_voxel)[:3]
        fu_world_lps = _ras_to_lps(fu_world_ras)

        # Elastix transform direction is fixed (FU) -> moving (native BL).
        bl_world_lps = np.asarray(
            rigid.TransformPoint(tuple(float(value) for value in fu_world_lps)),
            dtype=float,
        )
        bl_world_ras = _lps_to_ras(bl_world_lps)
        bl_world_h = np.asarray((*bl_world_ras, 1.0), dtype=float)
        bl_voxel = bl_affine_inv @ bl_world_h
        mapped_axis.append(float(bl_voxel[bl_axis]))

    return tuple(mapped_axis)


def mapped_native_bl_slice(
    mapped_axis_by_fu_slice: tuple[float, ...],
    *,
    fu_slice_index: int,
    bl_slice_count: int,
) -> tuple[int, float, bool]:
    """Return nearest BL slice, continuous index, and outside-FOV status."""
    if bl_slice_count <= 0:
        raise ValueError("BL slice count must be positive.")
    if not 0 <= fu_slice_index < len(mapped_axis_by_fu_slice):
        raise IndexError(
            f"FU slice {fu_slice_index} is outside the precomputed mapping."
        )

    continuous_index = float(mapped_axis_by_fu_slice[fu_slice_index])
    if not np.isfinite(continuous_index):
        raise ValueError(
            f"Mapped BL index for FU slice {fu_slice_index} is not finite."
        )

    last_index = bl_slice_count - 1
    outside_fov = (
        continuous_index < -0.5
        or continuous_index > last_index + 0.5
    )
    index = int(np.clip(np.rint(continuous_index), 0, last_index))
    return index, continuous_index, outside_fov


__all__ = ["fu_to_bl_slice_map", "mapped_native_bl_slice"]
