from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Allow:
#   streamlit run tools/align_longitudinal_patient_v2.py
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
APP_VERSION = "V2 · Fast 3-panel viewer"


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
def _load_ct_pair_cached(
    manifest_path: str,
    patient_id: str,
    data_root: str | None,
    manifest_modified_ns: int,
):
    """Load BL/FU CT once and keep the volumes in server memory."""
    del manifest_modified_ns
    return load_patient_pair(
        manifest_path,
        patient_id,
        data_root=data_root,
        modalities=("ct",),
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


def _normalise_optional_path(text: str) -> str | None:
    value = str(text).strip()
    return value if value else None


def _registration_paths(
    output_root: str | Path,
    patient_id: str,
) -> tuple[Path, Path]:
    output_dir = Path(output_root).expanduser() / str(patient_id)
    registered_path = output_dir / "registered_baseline_ct.nii.gz"
    return output_dir, registered_path


def _registration_is_ready(output_dir: Path, registered_path: Path) -> bool:
    """Rigid + affine output files required by the current stable pipeline."""
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


def _closest_native_bl_slice(
    bl_ct: LoadedVolume,
    target_world_xyz: np.ndarray,
) -> tuple[int, float, bool]:
    """
    Find the original BL axial slice plane closest to a FU-space world point.

    This is only an approximate 'before registration' comparison. Registered BL
    and FU share exact geometry, while Original BL remains in its native frame.

    Returns
    -------
    index:
        Nearest valid native BL slice index.
    continuous_index:
        Unrounded slice coordinate before clamping.
    outside_fov:
        True when the target lies beyond the native BL slice range.
    """
    affine = np.asarray(bl_ct.metadata.affine, dtype=float)
    shape = bl_ct.data.shape

    centre_voxel_0 = np.asarray(
        [
            (shape[0] - 1) / 2.0,
            (shape[1] - 1) / 2.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )
    centre_world_0 = (affine @ centre_voxel_0)[:3]

    row_axis = affine[:3, 0]
    col_axis = affine[:3, 1]
    slice_step = affine[:3, 2]

    normal = np.cross(row_axis, col_axis)
    normal_norm = float(np.linalg.norm(normal))

    target = np.asarray(target_world_xyz, dtype=float)

    if normal_norm > 1e-8:
        normal /= normal_norm
        denominator = float(np.dot(slice_step, normal))
    else:
        denominator = 0.0

    if abs(denominator) > 1e-8:
        continuous_index = float(
            np.dot(target - centre_world_0, normal) / denominator
        )
    else:
        # Fallback for an unusual/sheared affine: project onto the slice-step
        # centre line instead of the slice-plane normal.
        step_norm_sq = float(np.dot(slice_step, slice_step))
        if step_norm_sq <= 1e-12:
            continuous_index = 0.0
        else:
            continuous_index = float(
                np.dot(target - centre_world_0, slice_step) / step_norm_sq
            )

    last_index = shape[2] - 1
    outside_fov = continuous_index < -0.5 or continuous_index > last_index + 0.5

    index = int(np.clip(np.rint(continuous_index), 0, last_index))
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
) -> None:
    """
    Three-column Before / After / Reference viewer.

    Registered BL and FU use the same exact FU slice index. Original BL remains
    in its native geometry, so its displayed slice is selected by the closest
    native axial plane to the current FU world position.
    """
    st.divider()
    st.subheader("Alignment comparison")
    st.caption(
        "Original BL = before registration (approximate nearest native slice). "
        "Registered BL and FU = exact same FU-space slice."
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
        _closest_native_bl_slice(bl_ct, world_xyz)
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
            "This FU-space position falls outside the native BL slice range. "
            f"The Original BL panel is showing the nearest edge slice "
            f"({original_index})."
        )

    before, after, reference = st.columns(3)

    with before:
        st.markdown("**Original BL CT**")
        st.image(
            original_slice,
            caption=(
                f"Before alignment · native slice {original_index} "
                f"(estimated {original_float_index:.1f})"
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
        "ITKElastix rigid + affine registration. "
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

    data_root = _normalise_optional_path(data_root_text)
    output_root = Path(output_root_text).expanduser()
    output_dir, registered_path = _registration_paths(
        output_root,
        patient_id,
    )

    try:
        pair = _load_ct_pair_cached(
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
            with st.spinner(f"Registering {patient_id}: rigid → affine..."):
                register_patient_ct(
                    pair,
                    output_dir=output_dir,
                    stages=("rigid", "affine"),
                    number_of_resolutions=3,
                    maximum_iterations=256,
                    number_of_spatial_samples=4096,
                    log_to_console=False,
                    overwrite=True,
                )

            # Registered CT changed: invalidate both source-volume and display caches.
            _load_registered_ct_cached.clear()
            st.session_state.pop("_alignment_display_cache", None)
            st.success(f"Alignment completed: {registered_path}")
        except RegistrationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected registration failure: {exc}")

    ready = _registration_is_ready(output_dir, registered_path)

    if not ready:
        st.info(
            "No completed rigid + affine result is available for this patient yet. "
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
    )

    with st.expander("Registration details"):
        st.write("BL source:", str(bl_ct.metadata.path))
        st.write("FU source:", str(fu_ct.metadata.path))
        st.write("Registered BL:", str(registered_path))
        st.write("Output directory:", str(output_dir))
        st.write(
            "Transforms:",
            [
                str(output_dir / "TransformParameters.0.txt"),
                str(output_dir / "TransformParameters.1.txt"),
            ],
        )
        st.write("Registered BL shape:", registered_bl.data.shape)
        st.write("FU shape:", fu_ct.data.shape)
        st.write("FU orientation:", fu_ct.metadata.orientation)
        st.write("FU spacing (mm):", fu_ct.metadata.spacing_mm)


if __name__ == "__main__":
    main()
