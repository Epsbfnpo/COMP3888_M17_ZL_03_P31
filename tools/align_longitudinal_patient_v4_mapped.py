from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Allow:
#   streamlit run tools/align_longitudinal_patient_v4_mapped.py
# from the repository root without installing src as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cohort_a_loading import (  # noqa: E402
    LoadedVolume,
    load_nifti_volume,
    load_pair_manifest,
    load_patient_pair,
)
from src.registration import RegistrationError, register_patient_ct  # noqa: E402


DEFAULT_PAIRS = "outputs/cohort_a_subset/cohort_a_subset_pairs.csv"
DEFAULT_OUTPUT_ROOT = "outputs/registration"
APP_VERSION = "V4 · Rigid-mapped Original BL + masks"


@st.cache_data(show_spinner=False)
def _patient_ids(manifest_path: str, modified_ns: int) -> tuple[str, ...]:
    """Read patient IDs from a pair manifest."""
    del modified_ns  # Used only to invalidate the cache if the manifest changes.
    manifest = load_pair_manifest(manifest_path)
    return tuple(
        str(value).strip()
        for value in manifest["patient_id"].tolist()
        if str(value).strip()
    )


@st.cache_resource(show_spinner=False)
def _load_pair_cached(
    manifest_path: str,
    patient_id: str,
    data_root: str | None,
    manifest_modified_ns: int,
):
    """Load BL/FU CT and lesion masks once and keep them in server memory."""
    del manifest_modified_ns
    return load_patient_pair(
        manifest_path,
        patient_id,
        data_root=data_root,
        modalities=("ct", "lesion_mask"),
        allow_missing=True,
    )


@st.cache_resource(show_spinner=False)
def _load_registered_ct_cached(
    registered_path: str,
    modified_ns: int,
) -> LoadedVolume:
    """Load the registered CT once and reuse it between Streamlit reruns."""
    del modified_ns
    return load_nifti_volume(
        registered_path,
        role="registered baseline CT",
    )


@st.cache_resource(show_spinner=False)
def _load_registered_mask_cached(
    registered_mask_path: str,
    modified_ns: int,
) -> LoadedVolume:
    """Load a rigidly transformed BL lesion mask once."""
    del modified_ns
    return load_nifti_volume(
        registered_mask_path,
        role="registered baseline lesion mask",
        preserve_dtype=True,
    )



def _validate_image_mask_geometry(
    image: LoadedVolume,
    mask: LoadedVolume,
    *,
    label: str,
) -> None:
    if image.data.shape != mask.data.shape:
        raise ValueError(
            f"{label} CT/mask shape mismatch: "
            f"{image.data.shape} vs {mask.data.shape}."
        )

    if not np.allclose(
        image.metadata.affine,
        mask.metadata.affine,
        rtol=0.0,
        atol=1e-3,
    ):
        raise ValueError(f"{label} CT/mask affine geometry does not match.")


def _registered_mask_needs_update(
    registered_mask_path: Path,
    baseline_mask_path: Path,
    transform_path: Path,
) -> bool:
    if not registered_mask_path.exists():
        return True

    try:
        output_time = registered_mask_path.stat().st_mtime_ns
        return (
            baseline_mask_path.stat().st_mtime_ns > output_time
            or transform_path.stat().st_mtime_ns > output_time
        )
    except OSError:
        return True


def _warp_baseline_mask_to_fu(
    baseline_mask: LoadedVolume,
    *,
    transform_path: Path,
    registered_mask_path: Path,
) -> Path:
    """
    Apply the saved rigid BL→FU transform to the BL lesion mask.

    Nearest-neighbour interpolation is mandatory for label masks so the
    transformation does not create fractional lesion labels.
    """
    if not transform_path.exists():
        raise FileNotFoundError(
            f"Rigid transform not found: {transform_path}"
        )

    registered_mask_path.parent.mkdir(parents=True, exist_ok=True)

    if not _registered_mask_needs_update(
        registered_mask_path,
        baseline_mask.metadata.path,
        transform_path,
    ):
        return registered_mask_path

    try:
        import itk
    except ImportError as exc:
        raise RuntimeError(
            "ITKElastix is required to transform lesion masks."
        ) from exc

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(str(transform_path))

    # Preserve label semantics.
    parameter_object.SetParameter(
        0,
        "ResampleInterpolator",
        "FinalNearestNeighborInterpolator",
    )
    parameter_object.SetParameter(
        0,
        "FinalBSplineInterpolationOrder",
        "0",
    )
    parameter_object.SetParameter(
        0,
        "DefaultPixelValue",
        "0",
    )

    moving_mask = itk.imread(
        str(baseline_mask.metadata.path),
        itk.US,
    )

    transformix = itk.TransformixFilter.New(moving_mask)
    transformix.SetTransformParameterObject(parameter_object)
    transformix.SetLogToConsole(False)

    if hasattr(transformix, "SetLogToFile"):
        transformix.SetLogToFile(False)

    transformix.UpdateLargestPossibleRegion()
    result_mask = transformix.GetOutput()

    itk.imwrite(result_mask, str(registered_mask_path))
    return registered_mask_path


def _overlay_mask(
    grayscale: np.ndarray,
    mask: np.ndarray | None,
    *,
    alpha: float = 0.48,
) -> np.ndarray:
    """
    Overlay positive mask voxels on a uint8 CT slice.

    Returns RGB uint8 data suitable for st.image.
    """
    base = np.asarray(grayscale, dtype=np.uint8)

    if base.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale slice, got {base.shape}.")

    rgb = np.repeat(base[..., None], 3, axis=2)

    if mask is None:
        return rgb

    lesion = np.asarray(mask) > 0
    if lesion.shape != base.shape:
        raise ValueError(
            f"Mask slice shape {lesion.shape} does not match CT slice "
            f"shape {base.shape}."
        )

    if not np.any(lesion):
        return rgb

    # Warm red/orange overlay. Keep some underlying CT visible.
    overlay_colour = np.asarray([255.0, 64.0, 32.0], dtype=np.float32)
    pixels = rgb[lesion].astype(np.float32)
    pixels = (1.0 - alpha) * pixels + alpha * overlay_colour
    rgb[lesion] = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)

    # Add a bright boundary so small lesions remain visible.
    up = np.zeros_like(lesion)
    down = np.zeros_like(lesion)
    left = np.zeros_like(lesion)
    right = np.zeros_like(lesion)

    up[1:] = lesion[:-1]
    down[:-1] = lesion[1:]
    left[:, 1:] = lesion[:, :-1]
    right[:, :-1] = lesion[:, 1:]

    interior = lesion & up & down & left & right
    boundary = lesion & ~interior
    rgb[boundary] = np.asarray([255, 220, 64], dtype=np.uint8)

    return rgb


def _mask_slice(
    volume: np.ndarray,
    index: int,
) -> np.ndarray:
    """Extract/orient a mask slice using the same convention as the CT."""
    if not 0 <= index < volume.shape[2]:
        raise IndexError(
            f"Mask slice index {index} is outside "
            f"[0, {volume.shape[2] - 1}]."
        )

    axial = np.asarray(volume[:, :, index]) > 0
    return np.ascontiguousarray(np.flipud(np.rot90(axial)))

def _normalise_optional_path(text: str) -> str | None:
    value = str(text).strip()
    return value if value else None


def _registration_paths(
    output_root: str | Path,
    patient_id: str,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_root).expanduser() / str(patient_id)
    registered_path = output_dir / "registered_baseline_ct.nii.gz"
    registered_mask_path = output_dir / "registered_baseline_lesion_mask.nii.gz"
    return output_dir, registered_path, registered_mask_path


def _registration_is_ready(output_dir: Path, registered_path: Path) -> bool:
    """Rigid-only output files required by the current stable pipeline."""
    return (
        registered_path.exists()
        and (output_dir / "TransformParameters.0.txt").exists()
    )


def _window_values(name: str) -> tuple[float, float]:
    presets = {
        "Soft tissue": (-160.0, 240.0),
        "Lung": (-1000.0, 400.0),
        "Bone": (-500.0, 1500.0),
        "Wide": (-1000.0, 2000.0),
    }
    return presets[name]


def _geometry_matches(a: LoadedVolume, b: LoadedVolume) -> bool:
    return (
        a.data.shape == b.data.shape
        and np.allclose(
            a.metadata.affine,
            b.metadata.affine,
            rtol=0.0,
            atol=1e-3,
        )
    )


def _prepare_display_volume(
    data: np.ndarray,
    *,
    low_hu: float,
    high_hu: float,
) -> np.ndarray:
    """
    Window a whole CT volume once and convert it to uint8.

    This is intentionally done outside the slice fragment so moving the slider
    only selects/rotates a uint8 slice instead of repeating HU clipping and
    normalisation on every interaction.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D CT volume, got {data.shape}.")
    if high_hu <= low_hu:
        raise ValueError("CT window upper bound must be greater than lower bound.")

    # Work in float32 to reduce temporary memory compared with float64.
    image = np.asarray(data, dtype=np.float32).copy()
    np.nan_to_num(
        image,
        copy=False,
        nan=low_hu,
        posinf=high_hu,
        neginf=low_hu,
    )
    np.clip(image, low_hu, high_hu, out=image)

    image -= low_hu
    image *= 255.0 / (high_hu - low_hu)

    return np.rint(image).astype(np.uint8)


def _file_modified_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _display_cache_signature(
    patient_id: str,
    registered_path: Path,
    registered_modified_ns: int,
    bl_ct: LoadedVolume,
    fu_ct: LoadedVolume,
    window_name: str,
) -> tuple[object, ...]:
    """
    Build a signature for the single in-session display-volume cache.

    Only the current patient/window is retained, so switching among several
    patients does not accumulate multiple full CT display volumes in memory.
    """
    return (
        str(patient_id),
        str(registered_path.resolve()),
        int(registered_modified_ns),
        str(bl_ct.metadata.path),
        _file_modified_ns(bl_ct.metadata.path),
        str(fu_ct.metadata.path),
        _file_modified_ns(fu_ct.metadata.path),
        str(window_name),
    )


def _get_display_volumes(
    *,
    patient_id: str,
    registered_path: Path,
    registered_modified_ns: int,
    bl_ct: LoadedVolume,
    registered_bl: LoadedVolume,
    fu_ct: LoadedVolume,
    window_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Return windowed uint8 Original-BL / Registered-BL / FU volumes.

    These are prepared once per patient/window and kept in session state.
    Slider movement therefore only extracts three already-windowed slices.
    """
    low_hu, high_hu = _window_values(window_name)

    signature = _display_cache_signature(
        patient_id,
        registered_path,
        registered_modified_ns,
        bl_ct,
        fu_ct,
        window_name,
    )

    cache = st.session_state.get("_alignment_display_cache")
    if cache is None or cache.get("signature") != signature:
        with st.spinner(f"Preparing {window_name.lower()} display volumes..."):
            original_display = _prepare_display_volume(
                bl_ct.data,
                low_hu=low_hu,
                high_hu=high_hu,
            )
            registered_display = _prepare_display_volume(
                registered_bl.data,
                low_hu=low_hu,
                high_hu=high_hu,
            )
            fu_display = _prepare_display_volume(
                fu_ct.data,
                low_hu=low_hu,
                high_hu=high_hu,
            )

        st.session_state["_alignment_display_cache"] = {
            "signature": signature,
            "original": original_display,
            "registered": registered_display,
            "followup": fu_display,
            "low_hu": low_hu,
            "high_hu": high_hu,
        }
        cache = st.session_state["_alignment_display_cache"]

    return (
        cache["original"],
        cache["registered"],
        cache["followup"],
        float(cache["low_hu"]),
        float(cache["high_hu"]),
    )

def _axial_uint8_slice(volume: np.ndarray, index: int) -> np.ndarray:
    """
    Extract one already-windowed axial slice and orient it for display.

    No HU conversion is done here; this function is deliberately lightweight
    because it runs whenever the slice slider changes.
    """
    if not 0 <= index < volume.shape[2]:
        raise IndexError(
            f"Slice index {index} is outside [0, {volume.shape[2] - 1}]."
        )

    axial = volume[:, :, index]
    return np.ascontiguousarray(np.flipud(np.rot90(axial)))


def _slice_world_position_mm(volume: LoadedVolume, index: int) -> np.ndarray:
    """Return the world coordinate of the centre voxel on an axial slice."""
    shape = volume.data.shape
    voxel = np.asarray(
        [
            (shape[0] - 1) / 2.0,
            (shape[1] - 1) / 2.0,
            float(index),
            1.0,
        ],
        dtype=float,
    )
    world = np.asarray(volume.metadata.affine, dtype=float) @ voxel
    return world[:3]


def _ras_to_lps(point_xyz: np.ndarray) -> np.ndarray:
    """
    Convert nibabel/NIfTI RAS world coordinates to ITK/Elastix LPS coordinates.
    """
    point = np.asarray(point_xyz, dtype=float)
    return np.asarray([-point[0], -point[1], point[2]], dtype=float)


def _lps_to_ras(point_xyz: np.ndarray) -> np.ndarray:
    """Convert ITK/Elastix LPS world coordinates back to nibabel RAS."""
    point = np.asarray(point_xyz, dtype=float)
    return np.asarray([-point[0], -point[1], point[2]], dtype=float)


def _parameter_value(
    parameter_map,
    name: str,
    *,
    default: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    try:
        values = parameter_map[name]
    except Exception:
        if default is not None:
            return default
        raise ValueError(
            f"Rigid transform parameter '{name}' is missing."
        )

    return tuple(str(value) for value in values)


def _build_elastix_euler_transform(transform_path: str):
    """
    Reconstruct the saved Elastix 3-D Euler rigid transform.

    Important direction convention
    ------------------------------
    Elastix image-registration transforms map points from the FIXED image
    domain to the MOVING image domain. In this project:

        fixed  = FU
        moving = Original BL

    Therefore this transform can directly map a FU physical point to its
    corresponding location in Original BL space. No numerical inversion is
    required for this viewer.
    """
    try:
        import itk
    except ImportError as exc:
        raise RuntimeError(
            "ITKElastix is required to map FU positions back to Original BL."
        ) from exc

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(str(transform_path))
    parameter_map = parameter_object.GetParameterMap(0)

    transform_name = _parameter_value(
        parameter_map,
        "Transform",
    )[0].strip('"')

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
            "transform. This viewer intentionally refuses to ignore a transform "
            f"chain: {initial}"
        )

    parameters = tuple(
        float(value)
        for value in _parameter_value(
            parameter_map,
            "TransformParameters",
        )
    )

    if len(parameters) != 6:
        raise ValueError(
            "Expected six Euler rigid parameters "
            "(rx, ry, rz, tx, ty, tz); "
            f"found {len(parameters)}."
        )

    centre = tuple(
        float(value)
        for value in _parameter_value(
            parameter_map,
            "CenterOfRotationPoint",
        )
    )

    if len(centre) != 3:
        raise ValueError(
            "Expected a 3-D CenterOfRotationPoint in the rigid transform."
        )

    compute_zyx_text = _parameter_value(
        parameter_map,
        "ComputeZYX",
        default=("false",),
    )[0].strip('"').lower()
    compute_zyx = compute_zyx_text == "true"

    rigid = itk.Euler3DTransform[itk.D].New()
    rigid.SetCenter(centre)
    rigid.SetComputeZYX(compute_zyx)
    rigid.SetRotation(
        parameters[0],
        parameters[1],
        parameters[2],
    )
    rigid.SetTranslation(
        (
            parameters[3],
            parameters[4],
            parameters[5],
        )
    )

    return rigid


@st.cache_data(show_spinner=False)
def _fu_to_bl_slice_map(
    *,
    bl_shape: tuple[int, int, int],
    bl_affine_flat: tuple[float, ...],
    fu_shape: tuple[int, int, int],
    fu_affine_flat: tuple[float, ...],
    transform_path: str,
    transform_modified_ns: int,
) -> tuple[float, ...]:
    """
    Map every FU axial slice centre into the Original-BL voxel coordinate system.

    The expensive transform parsing is done once and the result is cached.
    Slider interaction then requires only an array lookup and rounding.

    Returns one continuous Original-BL z index per FU slice.
    """
    del transform_modified_ns  # cache invalidation only

    bl_affine = np.asarray(bl_affine_flat, dtype=float).reshape(4, 4)
    fu_affine = np.asarray(fu_affine_flat, dtype=float).reshape(4, 4)

    if tuple(int(v) for v in bl_shape) != bl_shape:
        raise ValueError("Invalid BL shape.")
    if tuple(int(v) for v in fu_shape) != fu_shape:
        raise ValueError("Invalid FU shape.")

    try:
        bl_affine_inv = np.linalg.inv(bl_affine)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Original BL affine is not invertible.") from exc

    rigid = _build_elastix_euler_transform(transform_path)

    mapped_z: list[float] = []

    # Use the centre voxel of each FU axial slice as the representative point.
    # This is exact for locating that centre point. If the rigid transform
    # includes rotation, the full FU slice plane may be oblique in native BL
    # space, so a single native BL axial slice is still the closest-plane
    # presentation rather than a resampled oblique plane.
    fu_x = (fu_shape[0] - 1) / 2.0
    fu_y = (fu_shape[1] - 1) / 2.0

    for slice_index in range(fu_shape[2]):
        fu_voxel = np.asarray(
            [
                fu_x,
                fu_y,
                float(slice_index),
                1.0,
            ],
            dtype=float,
        )

        # nibabel gives RAS physical coordinates.
        fu_world_ras = (fu_affine @ fu_voxel)[:3]

        # Elastix/ITK operates in LPS physical coordinates.
        fu_world_lps = _ras_to_lps(fu_world_ras)

        # Elastix transform direction is fixed(FU) -> moving(BL).
        bl_world_lps = np.asarray(
            rigid.TransformPoint(tuple(float(v) for v in fu_world_lps)),
            dtype=float,
        )

        # Convert back to nibabel RAS, then to Original-BL voxel coordinates.
        bl_world_ras = _lps_to_ras(bl_world_lps)
        bl_world_h = np.asarray(
            [
                bl_world_ras[0],
                bl_world_ras[1],
                bl_world_ras[2],
                1.0,
            ],
            dtype=float,
        )
        bl_voxel = bl_affine_inv @ bl_world_h
        mapped_z.append(float(bl_voxel[2]))

    return tuple(mapped_z)


def _mapped_native_bl_slice(
    mapped_z_by_fu_slice: tuple[float, ...],
    *,
    fu_slice_index: int,
    bl_slice_count: int,
) -> tuple[int, float, bool]:
    """
    Return the nearest Original-BL native axial slice for a FU slice.

    The continuous z coordinate comes from the saved ITKElastix rigid
    registration transform, not from raw-coordinate proximity.
    """
    if not 0 <= fu_slice_index < len(mapped_z_by_fu_slice):
        raise IndexError(
            f"FU slice {fu_slice_index} is outside the precomputed mapping."
        )

    continuous_index = float(mapped_z_by_fu_slice[fu_slice_index])
    last_index = bl_slice_count - 1

    outside_fov = (
        continuous_index < -0.5
        or continuous_index > last_index + 0.5
    )

    index = int(
        np.clip(
            np.rint(continuous_index),
            0,
            last_index,
        )
    )

    return index, continuous_index, outside_fov


@st.fragment
def _render_slice_viewer(
    original_display: np.ndarray,
    registered_display: np.ndarray,
    fu_display: np.ndarray,
    bl_ct: LoadedVolume,
    fu_ct: LoadedVolume,
    patient_id: str,
    low_hu: float,
    high_hu: float,
    *,
    mapped_z_by_fu_slice: tuple[float, ...],
    show_masks: bool,
    bl_mask: LoadedVolume | None,
    registered_bl_mask: LoadedVolume | None,
    fu_mask: LoadedVolume | None,
) -> None:
    """
    Three-column Before / After / Reference viewer.

    Registered BL and FU use the same exact FU slice index. Original BL remains
    in its native geometry. Its displayed native slice is selected from the
    saved Elastix rigid transform that maps the current FU slice centre into
    Original-BL space.
    """
    st.divider()
    st.subheader("Alignment comparison")
    st.caption(
        "Original BL = nearest native slice located using the saved rigid "
        "FU→BL point mapping. Registered BL and FU = exact same FU-space "
        "slice. " + ("Lesion masks are ON." if show_masks else "Lesion masks are OFF.")
    )

    slice_count = fu_display.shape[2]

    state_key = f"slice_{patient_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = slice_count // 2
    elif not 0 <= int(st.session_state[state_key]) < slice_count:
        st.session_state[state_key] = slice_count // 2

    slice_index = st.slider(
        "FU-space axial slice",
        min_value=0,
        max_value=slice_count - 1,
        step=1,
        key=state_key,
    )

    world_xyz = _slice_world_position_mm(fu_ct, slice_index)
    original_index, original_float_index, outside_fov = (
        _mapped_native_bl_slice(
            mapped_z_by_fu_slice,
            fu_slice_index=slice_index,
            bl_slice_count=bl_ct.data.shape[2],
        )
    )

    try:
        original_slice = _axial_uint8_slice(
            original_display,
            original_index,
        )
        registered_slice = _axial_uint8_slice(
            registered_display,
            slice_index,
        )
        fu_slice = _axial_uint8_slice(
            fu_display,
            slice_index,
        )

        original_mask_slice = None
        registered_mask_slice = None
        fu_mask_slice = None

        if show_masks:
            if bl_mask is not None:
                original_mask_slice = _mask_slice(
                    bl_mask.data,
                    original_index,
                )
            if registered_bl_mask is not None:
                registered_mask_slice = _mask_slice(
                    registered_bl_mask.data,
                    slice_index,
                )
            if fu_mask is not None:
                fu_mask_slice = _mask_slice(
                    fu_mask.data,
                    slice_index,
                )

        original_slice = _overlay_mask(
            original_slice,
            original_mask_slice,
        )
        registered_slice = _overlay_mask(
            registered_slice,
            registered_mask_slice,
        )
        fu_slice = _overlay_mask(
            fu_slice,
            fu_mask_slice,
        )

    except Exception as exc:
        st.error(f"Could not render current slices: {exc}")
        return

    st.caption(
        f"FU-space slice {slice_index + 1}/{slice_count} · "
        f"world centre ≈ "
        f"({world_xyz[0]:.1f}, {world_xyz[1]:.1f}, {world_xyz[2]:.1f}) mm · "
        f"window [{low_hu:.0f}, {high_hu:.0f}] HU"
    )

    if outside_fov:
        st.warning(
            "The rigid-mapped FU position falls outside the native BL slice range. "
            f"The Original BL panel is showing the nearest edge slice "
            f"({original_index})."
        )

    before, after, reference = st.columns(3)

    with before:
        st.markdown("**Original BL CT**")
        st.image(
            original_slice,
            caption=(
                f"Original data · rigid-mapped native slice {original_index} "
                f"(mapped z={original_float_index:.1f})"
            ),
            use_container_width=True,
        )

    with after:
        st.markdown("**Registered BL CT**")
        st.image(
            registered_slice,
            caption=f"After alignment · FU-space slice {slice_index}",
            use_container_width=True,
        )

    with reference:
        st.markdown("**FU CT**")
        st.image(
            fu_slice,
            caption=f"Reference · FU-space slice {slice_index}",
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Longitudinal CT Alignment",
        page_icon="🩻",
        layout="wide",
    )

    st.title("Longitudinal BL → FU CT Alignment")
    st.caption(f"**{APP_VERSION}**")
    st.caption(
        "ITKElastix rigid-only registration. "
        "The viewer shows Original BL, Registered BL, and FU CT side by side."
    )

    with st.sidebar:
        st.caption(APP_VERSION)
        st.header("Data")

        manifest_text = st.text_input(
            "Pair manifest",
            value=DEFAULT_PAIRS,
            help="Usually outputs/cohort_a_subset/cohort_a_subset_pairs.csv",
        )
        data_root_text = st.text_input(
            "Data root",
            value=(os.environ.get("DATA_ROOT") or os.environ.get("COHORT_B_ROOT") or os.environ.get("COHORT_A_ROOT", "")),
            help=(
                "Leave blank if manifest paths are self-contained or a data-root "
                "environment variable is already set."
            ),
        )
        output_root_text = st.text_input(
            "Registration output root",
            value=DEFAULT_OUTPUT_ROOT,
        )

    manifest_path = Path(manifest_text).expanduser()
    if not manifest_path.exists():
        st.error(f"Pair manifest not found: {manifest_path}")
        st.stop()

    try:
        manifest_modified_ns = manifest_path.stat().st_mtime_ns
        patient_ids = _patient_ids(
            str(manifest_path),
            manifest_modified_ns,
        )
    except Exception as exc:
        st.error(f"Could not read pair manifest: {exc}")
        st.stop()

    if not patient_ids:
        st.warning("No patient IDs were found in the pair manifest.")
        st.stop()

    with st.sidebar:
        patient_id = st.selectbox(
            "Patient",
            options=patient_ids,
        )

        st.header("Alignment")
        st.write("Conservative baseline: **Rigid only**")
        run_alignment = st.button(
            "Run / re-run alignment",
            type="primary",
            use_container_width=True,
        )

        st.header("Viewer")
        window_name = st.selectbox(
            "CT window",
            options=("Soft tissue", "Lung", "Bone", "Wide"),
            index=0,
        )

        show_masks = st.checkbox(
            "Show lesion masks",
            value=False,
            help=(
                "Overlay BL/FU lesion masks. The BL mask is transformed into "
                "FU space using the saved rigid transform and nearest-neighbour "
                "interpolation."
            ),
        )

    data_root = _normalise_optional_path(data_root_text)
    output_root = Path(output_root_text).expanduser()
    output_dir, registered_path, registered_mask_path = _registration_paths(
        output_root,
        patient_id,
    )

    try:
        pair = _load_pair_cached(
            str(manifest_path),
            patient_id,
            data_root,
            manifest_modified_ns,
        )
    except Exception as exc:
        st.error(f"Could not load CT pair for patient '{patient_id}': {exc}")
        st.stop()

    if pair.baseline.ct is None or pair.followup.ct is None:
        st.error("This patient does not have both baseline and follow-up CT loaded.")
        st.stop()

    bl_ct = pair.baseline.ct
    fu_ct = pair.followup.ct

    top1, top2, top3 = st.columns(3)
    top1.metric("Patient", patient_id)
    top2.metric("BL CT shape", " × ".join(map(str, bl_ct.data.shape)))
    top3.metric("FU CT shape", " × ".join(map(str, fu_ct.data.shape)))

    if run_alignment:
        try:
            with st.spinner(f"Registering {patient_id}: rigid only..."):
                register_patient_ct(
                    pair,
                    output_dir=output_dir,
                    stages=("rigid",),
                    number_of_resolutions=3,
                    maximum_iterations=256,
                    number_of_spatial_samples=4096,
                    log_to_console=False,
                    overwrite=True,
                )

            # Registered CT changed: invalidate both source-volume and display caches.
            _load_registered_ct_cached.clear()
            _load_registered_mask_cached.clear()
            st.session_state.pop("_alignment_display_cache", None)
            st.success(f"Alignment completed: {registered_path}")
        except RegistrationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected registration failure: {exc}")

    ready = _registration_is_ready(output_dir, registered_path)

    if not ready:
        st.info(
            "No completed rigid-only result is available for this patient yet. "
            "Click **Run / re-run alignment** in the sidebar."
        )
        st.write("BL CT:", bl_ct.metadata.path)
        st.write("FU CT:", fu_ct.metadata.path)
        st.stop()

    try:
        registered_modified_ns = registered_path.stat().st_mtime_ns
        registered_bl = _load_registered_ct_cached(
            str(registered_path),
            registered_modified_ns,
        )
    except Exception as exc:
        st.error(f"Could not load registered baseline CT: {exc}")
        st.stop()

    if not _geometry_matches(registered_bl, fu_ct):
        st.error(
            "Registered BL CT and FU CT do not share the same shape/affine. "
            "The aligned slice viewer is disabled because equal slice indices "
            "would not represent the same physical space."
        )
        st.stop()

    transform_path = output_dir / "TransformParameters.0.txt"

    try:
        mapped_z_by_fu_slice = _fu_to_bl_slice_map(
            bl_shape=tuple(int(v) for v in bl_ct.data.shape),
            bl_affine_flat=tuple(
                float(v)
                for v in np.asarray(
                    bl_ct.metadata.affine,
                    dtype=float,
                ).reshape(-1)
            ),
            fu_shape=tuple(int(v) for v in fu_ct.data.shape),
            fu_affine_flat=tuple(
                float(v)
                for v in np.asarray(
                    fu_ct.metadata.affine,
                    dtype=float,
                ).reshape(-1)
            ),
            transform_path=str(transform_path),
            transform_modified_ns=transform_path.stat().st_mtime_ns,
        )
    except Exception as exc:
        st.error(
            "Could not build the FU → Original-BL native-slice mapping from "
            f"the saved rigid transform: {exc}"
        )
        st.stop()

    bl_mask = pair.baseline.lesion_mask
    fu_mask = pair.followup.lesion_mask
    registered_bl_mask = None

    if show_masks:
        if bl_mask is None or fu_mask is None:
            st.warning(
                "Lesion-mask overlay requested, but this patient does not have "
                "both BL and FU lesion masks. Available masks will still be shown."
            )

        try:
            if bl_mask is not None:
                _validate_image_mask_geometry(
                    bl_ct,
                    bl_mask,
                    label="Baseline",
                )
            if fu_mask is not None:
                _validate_image_mask_geometry(
                    fu_ct,
                    fu_mask,
                    label="Follow-up",
                )

            if bl_mask is not None:
                with st.spinner(
                    "Preparing rigidly transformed BL lesion mask..."
                ):
                    _warp_baseline_mask_to_fu(
                        bl_mask,
                        transform_path=transform_path,
                        registered_mask_path=registered_mask_path,
                    )

                registered_mask_modified_ns = (
                    registered_mask_path.stat().st_mtime_ns
                )
                registered_bl_mask = _load_registered_mask_cached(
                    str(registered_mask_path),
                    registered_mask_modified_ns,
                )

                _validate_image_mask_geometry(
                    fu_ct,
                    registered_bl_mask,
                    label="Registered BL mask / FU",
                )

        except Exception as exc:
            st.error(f"Could not prepare lesion-mask overlay: {exc}")
            st.stop()

    try:
        (
            original_display,
            registered_display,
            fu_display,
            low_hu,
            high_hu,
        ) = _get_display_volumes(
            patient_id=patient_id,
            registered_path=registered_path,
            registered_modified_ns=registered_modified_ns,
            bl_ct=bl_ct,
            registered_bl=registered_bl,
            fu_ct=fu_ct,
            window_name=window_name,
        )
    except Exception as exc:
        st.error(f"Could not prepare CT display volumes: {exc}")
        st.stop()

    _render_slice_viewer(
        original_display,
        registered_display,
        fu_display,
        bl_ct,
        fu_ct,
        patient_id,
        low_hu,
        high_hu,
        mapped_z_by_fu_slice=mapped_z_by_fu_slice,
        show_masks=show_masks,
        bl_mask=bl_mask,
        registered_bl_mask=registered_bl_mask,
        fu_mask=fu_mask,
    )

    with st.expander("Registration details"):
        st.write("BL source:", str(bl_ct.metadata.path))
        st.write("FU source:", str(fu_ct.metadata.path))
        st.write("Registered BL:", str(registered_path))
        st.write("Output directory:", str(output_dir))
        st.write(
            "Rigid transform:",
            str(output_dir / "TransformParameters.0.txt"),
        )
        st.write(
            "Original-BL slice selection:",
            "FU slice centre mapped directly through the saved Elastix "
            "fixed(FU) → moving(BL) rigid transform, then rounded to the "
            "nearest native BL axial slice.",
        )
        if registered_mask_path.exists():
            st.write(
                "Registered BL lesion mask:",
                str(registered_mask_path),
            )
        st.write("Registered BL shape:", registered_bl.data.shape)
        st.write("FU shape:", fu_ct.data.shape)
        st.write("FU orientation:", fu_ct.metadata.orientation)
        st.write("FU spacing (mm):", fu_ct.metadata.spacing_mm)


if __name__ == "__main__":
    main()
