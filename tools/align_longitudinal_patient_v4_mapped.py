from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

#allow streamlit to run from the project root without installing src
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
from src.correspondence_visualization import OUTCOME_STYLES, selected_lesion_mask, lesion_display_slice
from src.lesion_min_cost_flow import MatcherConfig  # noqa: E402
from src.matching_dashboard import (  # noqa: E402
    MatchImageFocus,
    MatchingDashboardError,
    load_patient_features,
    load_patient_matches,
    resolve_match_image_focus,
    run_patient_match,
    summarise_patient_matches,
)
from src.multiplanar_view import (  # noqa: E402
    PLANE_NAMES,
    anatomical_axis,
    display_pixel_spacing_mm,
    extract_display_slice,
    physical_aspect_resize,
    plane_info,
    resample_to_reference,
    slice_centre_world_mm,
)

DEFAULT_PAIRS = "outputs/cohort_a_subset/cohort_a_subset_pairs.csv"
DEFAULT_OUTPUT_ROOT = "outputs/registration"
DEFAULT_MATCH_RESULTS = "outputs/tracking_pipeline_11/lesion_matches.csv"
DEFAULT_ALIGNED_FEATURES = "outputs/tracking_pipeline_11/aligned_lesion_features.csv"
DEFAULT_COST_MATRIX_DIR = "outputs/tracking_pipeline_11/lesion_pair_cost_matrices"
DEFAULT_PAIR_COSTS = "outputs/tracking_pipeline_11/lesion_pair_costs.csv"
DEFAULT_EVALUATION_DETAILS = "outputs/tracking_evaluation_15/lesion_tracking_details.csv"
APP_VERSION = "V4.3 · Alignment + detailed lesion correspondence"

@st.cache_data(show_spinner=False)
def _patient_ids(manifest_path: str, modified_ns: int) -> tuple[str, ...]:
    #read patient ids from pair manifest.
    del modified_ns  #used only to invalidate the cache when manifest changes
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
    load_pet: bool,
):
    #cache the volumes needed for the selected dashboard mode
    del manifest_modified_ns
    modalities = ("ct", "pet", "lesion_mask") if load_pet else ("ct", "lesion_mask")
    return load_patient_pair(
        manifest_path,
        patient_id,
        data_root=data_root,
        modalities=modalities,
        allow_missing=True,
    )

@st.cache_resource(show_spinner=False)
def _load_registered_ct_cached(
    registered_path: str,
    modified_ns: int,
) -> LoadedVolume:
    #cache registered CT data across streamlit reruns
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
    #cache the rigidly transformed BL lesion mask
    del modified_ns
    return load_nifti_volume(
        registered_mask_path,
        role="registered baseline lesion mask",
        preserve_dtype=True,
    )

@st.cache_resource(show_spinner=False)
def _load_registered_pet_cached(
    registered_pet_path: str,
    modified_ns: int,
) -> LoadedVolume:
    #cache the rigidly transformed BL PET volume
    del modified_ns
    return load_nifti_volume(
        registered_pet_path,
        role="registered baseline PET",
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
    #apply the saved BL-to-FU transform with nearest-neighbour interpolation to preserve mask labels
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
    #preserve label semantics.
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

def _registered_pet_needs_update(
    registered_pet_path: Path,
    baseline_pet_path: Path,
    transform_path: Path,
) -> bool:
    if not registered_pet_path.exists():
        return True
    try:
        output_time = registered_pet_path.stat().st_mtime_ns
        return (
            baseline_pet_path.stat().st_mtime_ns > output_time
            or transform_path.stat().st_mtime_ns > output_time
        )
    except OSError:
        return True

def _warp_baseline_pet_to_fu(
    baseline_pet: LoadedVolume,
    *,
    transform_path: Path,
    registered_pet_path: Path,
) -> Path:
    #apply the saved BL-to-FU transform to PET with linear interpolation
    if not transform_path.exists():
        raise FileNotFoundError(f"Rigid transform not found: {transform_path}")
    registered_pet_path.parent.mkdir(parents=True, exist_ok=True)
    if not _registered_pet_needs_update(
        registered_pet_path,
        baseline_pet.metadata.path,
        transform_path,
    ):
        return registered_pet_path
    try:
        import itk
    except ImportError as exc:
        raise RuntimeError(
            "ITKElastix is required to transform baseline PET into FU space."
        ) from exc
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(str(transform_path))
    #use linear interpolation for continuous PET intensities
    parameter_object.SetParameter(
        0,
        "ResampleInterpolator",
        "FinalBSplineInterpolator",
    )
    parameter_object.SetParameter(
        0,
        "FinalBSplineInterpolationOrder",
        "1",
    )
    parameter_object.SetParameter(0, "DefaultPixelValue", "0")
    parameter_object.SetParameter(0, "ResultImagePixelType", "float")
    moving_pet = itk.imread(str(baseline_pet.metadata.path), itk.F)
    transformix = itk.TransformixFilter.New(moving_pet)
    transformix.SetTransformParameterObject(parameter_object)
    transformix.SetLogToConsole(False)
    if hasattr(transformix, "SetLogToFile"):
        transformix.SetLogToFile(False)
    transformix.UpdateLargestPossibleRegion()
    itk.imwrite(transformix.GetOutput(), str(registered_pet_path))
    return registered_pet_path

def _overlay_mask(
    grayscale: np.ndarray,
    mask: np.ndarray | None,
    *,
    alpha: float = 0.48,
    colour: tuple[int, int, int] = (255, 64, 32),
) -> np.ndarray:
    #overlay mask voxels on a CT slice and return uint8 RGB data for streamlit
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
    #keep the underlying CT visible through the mask colour
    overlay_colour = np.asarray(colour, dtype=np.float32)
    pixels = rgb[lesion].astype(np.float32)
    pixels = (1.0 - alpha) * pixels + alpha * overlay_colour
    rgb[lesion] = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
    #add a bright boundary so small lesions remain visible
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
    rgb[boundary] = np.asarray(
        [255, 220, 64] if colour == (255, 64, 32) else colour, dtype=np.uint8
    )
    return rgb

def _mask_slice(
    volume: np.ndarray,
    index: int,
    *,
    axis: int,
) -> np.ndarray:
    #extract and orient the mask slice in the image plane
    return extract_display_slice(
        np.asarray(volume) > 0,
        axis=axis,
        index=index,
    )

def _normalise_optional_path(text: str) -> str | None:
    value = str(text).strip()
    return value if value else None

def _normalise_optional_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _safe_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@st.cache_data(show_spinner=False)
def _load_csv_cached(path: str, modified_ns: int) -> pd.DataFrame:
    # Cache auxiliary CSVs and invalidate when the file changes.
    del modified_ns
    return pd.read_csv(path)


def _patient_auxiliary_rows(path: Path, patient_id: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = _load_csv_cached(str(path), path.stat().st_mtime_ns).copy()
    if "patient_id" not in frame.columns:
        return None
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip()
    return frame[frame["patient_id"] == str(patient_id)].copy()


def _event_row(
    frame: pd.DataFrame | None,
    *,
    bl_lesion_id: str | None,
    fu_lesion_id: str | None,
) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    if "bl_lesion_id" not in frame.columns or "fu_lesion_id" not in frame.columns:
        return None

    bl_values = frame["bl_lesion_id"].map(_normalise_optional_id)
    fu_values = frame["fu_lesion_id"].map(_normalise_optional_id)
    selected = frame[(bl_values == bl_lesion_id) & (fu_values == fu_lesion_id)]
    if selected.empty:
        return None
    return selected.iloc[0]


def _feature_row(
    features: pd.DataFrame,
    *,
    timepoint: str,
    lesion_id: str | None,
) -> pd.Series | None:
    if lesion_id is None or features.empty:
        return None
    required = {"timepoint", "lesion_id"}
    if not required.issubset(features.columns):
        return None
    selected = features[
        (features["timepoint"].astype(str).str.upper() == timepoint.upper())
        & (features["lesion_id"].astype(str) == lesion_id)
    ]
    if selected.empty:
        return None
    return selected.iloc[0]


def _format_metric_number(value: object, *, digits: int = 3, suffix: str = "") -> str:
    number = _safe_float(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}{suffix}"


def _render_matching_panel(
    *,
    patient_id: str,
    results_path: Path,
    feature_path: Path,
    matrix_dir: Path,
    pair_cost_path: Path,
    evaluation_details_path: Path | None,
    run_matching: bool,
    config: MatcherConfig,
) -> MatchImageFocus | None:
    # Run/load matching, then expose the evidence required by Jira P31-21.
    st.subheader("Lesion correspondence")
    st.caption(
        "Automatic BL ↔ FU outcomes for the selected patient. Select an event "
        "to inspect lesion IDs, matcher features, cost/distance, unmatched status, "
        "and ground-truth agreement when evaluation data is available."
    )

    matrix_path = matrix_dir / f"{patient_id}_cost_matrix.csv"

    if run_matching:
        try:
            with st.spinner(f"Running min-cost-flow matching for {patient_id}..."):
                run_patient_match(matrix_path, patient_id, results_path, config)
            st.success(f"Matching completed and saved to {results_path}")
        except (FileNotFoundError, MatchingDashboardError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected matching failure for patient '{patient_id}': {exc}")

    try:
        matches = load_patient_matches(results_path, patient_id)
    except FileNotFoundError as exc:
        st.info(
            f"{exc} Run matching from the sidebar after generating this "
            "patient's cost matrix."
        )
        return None
    except MatchingDashboardError as exc:
        st.warning(str(exc))
        return None

    if matches.empty:
        st.info("No lesion correspondence rows are available for this patient.")
        return None

    summary = summarise_patient_matches(matches, patient_id)
    counts = summary.outcome_counts

    metric_columns = st.columns(6)
    metric_columns[0].metric("BL lesions", summary.baseline_lesions)
    metric_columns[1].metric("FU lesions", summary.followup_lesions)
    for column, match_type in zip(
        metric_columns[2:],
        ("MATCHED", "MERGING", "DISAPPEARING", "NEW"),
    ):
        column.metric(match_type.title(), counts.get(match_type, 0))

    # Make unmatched events explicit rather than relying only on a blank endpoint.
    display_matches = matches.copy()

    def _interpretation(row: pd.Series) -> str:
        event_type = str(row.get("match_type", "")).upper()
        if event_type == "DISAPPEARING":
            return "Unmatched BL lesion (disappearing)"
        if event_type == "NEW":
            return "Unmatched FU lesion (new)"
        if event_type == "MERGING":
            return "BL lesion participates in merge"
        return "BL ↔ FU correspondence"

    display_matches["interpretation"] = display_matches.apply(_interpretation, axis=1)
    for id_column in ("bl_lesion_id", "fu_lesion_id"):
        if id_column in display_matches.columns:
            display_matches[id_column] = display_matches[id_column].map(
                lambda value: _normalise_optional_id(value) or "—"
            )

    preferred_columns = [
        "bl_lesion_id",
        "fu_lesion_id",
        "match_type",
        "interpretation",
        "assignment_role",
        "pair_cost",
        "event_cost",
        "match_cost",
    ]
    display_columns = [column for column in preferred_columns if column in display_matches]

    st.dataframe(
        display_matches[display_columns],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(f"Matching results: {results_path}")

    def _match_option_label(index: int) -> str:
        row = matches.iloc[index]
        bl_id = _normalise_optional_id(row.get("bl_lesion_id")) or "—"
        fu_id = _normalise_optional_id(row.get("fu_lesion_id")) or "—"
        return f"{row['match_type']} · {bl_id} → {fu_id}"

    selected_index = st.selectbox(
        "Select lesion correspondence / unmatched event",
        options=list(range(len(matches))),
        format_func=_match_option_label,
        key=f"selected_match_{patient_id}",
    )

    selected_match = matches.iloc[int(selected_index)]
    bl_id = _normalise_optional_id(selected_match.get("bl_lesion_id"))
    fu_id = _normalise_optional_id(selected_match.get("fu_lesion_id"))
    match_type = str(selected_match.get("match_type", "")).upper()

    # Load aligned features for selected-lesion details and image linking.
    try:
        features = load_patient_features(feature_path, patient_id)
    except (FileNotFoundError, MatchingDashboardError) as exc:
        st.warning(f"Matching results are available, but feature details are unavailable: {exc}")
        features = pd.DataFrame()

    bl_feature = _feature_row(features, timepoint="BL", lesion_id=bl_id)
    fu_feature = _feature_row(features, timepoint="FU", lesion_id=fu_id)

    # Pair-cost CSV contains the actual feature breakdown used to construct candidate edges.
    try:
        pair_costs = _patient_auxiliary_rows(pair_cost_path, patient_id)
    except Exception as exc:
        st.warning(f"Could not read pair-cost details: {exc}")
        pair_costs = None

    pair_cost_row = _event_row(
        pair_costs,
        bl_lesion_id=bl_id,
        fu_lesion_id=fu_id,
    )

    st.markdown("#### Selected match details")
    overview = st.columns(4)
    overview[0].metric("BL lesion ID", bl_id or "—")
    overview[1].metric("FU lesion ID", fu_id or "—")
    overview[2].metric("Outcome", match_type or "—")
    overview[3].metric(
        "Match cost",
        _format_metric_number(selected_match.get("match_cost"), digits=4),
    )

    if match_type == "DISAPPEARING":
        st.warning(f"{bl_id or 'BL lesion'} has no FU correspondence and is labelled DISAPPEARING.")
    elif match_type == "NEW":
        st.warning(f"{fu_id or 'FU lesion'} has no BL correspondence and is labelled NEW.")

    # Feature block: show exact pair-cost inputs when a BL↔FU candidate exists.
    st.markdown("#### Matcher features and cost")

    if pair_cost_row is not None:
        feature_metrics = st.columns(4)
        feature_metrics[0].metric(
            "Distance",
            _format_metric_number(pair_cost_row.get("distance_mm"), digits=2, suffix=" mm"),
        )
        feature_metrics[1].metric(
            "BL volume",
            _format_metric_number(pair_cost_row.get("bl_volume_ml"), digits=3, suffix=" mL"),
        )
        feature_metrics[2].metric(
            "FU volume",
            _format_metric_number(pair_cost_row.get("fu_volume_ml"), digits=3, suffix=" mL"),
        )
        size_fraction = _safe_float(pair_cost_row.get("size_difference_fraction"))
        feature_metrics[3].metric(
            "Size difference",
            "—" if size_fraction is None else f"{100.0 * size_fraction:.1f}%",
        )

        cost_fields = [
            "distance_cost",
            "size_cost",
            "pet_cost",
            "total_cost",
            "candidate_rank_for_bl",
            "pet_feature",
            "bl_pet_value",
            "fu_pet_value",
            "pet_difference_fraction",
            "pet_used",
        ]
        available_cost_fields = [field for field in cost_fields if field in pair_cost_row.index]
        if available_cost_fields:
            detail_table = pd.DataFrame(
                {
                    "Field": available_cost_fields,
                    "Value": [pair_cost_row.get(field) for field in available_cost_fields],
                }
            )
            with st.expander("Cost component details"):
                st.dataframe(detail_table, hide_index=True, use_container_width=True)

        st.caption(f"Matcher feature/cost source: {pair_cost_path}")
    else:
        # NEW/DISAPPEARING rows have no BL↔FU candidate edge. Show the available
        # single-lesion features instead and make distance explicitly N/A.
        feature_metrics = st.columns(4)
        feature_metrics[0].metric("Distance", "N/A")
        feature_metrics[1].metric(
            "BL volume",
            _format_metric_number(
                None if bl_feature is None else bl_feature.get("volume_ml"),
                digits=3,
                suffix=" mL",
            ),
        )
        feature_metrics[2].metric(
            "FU volume",
            _format_metric_number(
                None if fu_feature is None else fu_feature.get("volume_ml"),
                digits=3,
                suffix=" mL",
            ),
        )
        feature_metrics[3].metric("Size difference", "N/A")

        if bl_id is not None and fu_id is not None:
            st.info(
                "No matching row was found in the pair-cost detail CSV for this BL/FU pair. "
                "Check that the pair-cost CSV belongs to the same pipeline run as the match results."
            )
        else:
            st.caption(
                "Distance and pairwise size difference are not applicable to NEW/DISAPPEARING "
                "events because one endpoint is absent. The displayed match cost is the unmatched "
                "event penalty used by the flow model."
            )

    # Ground-truth agreement is read only from evaluation output. The dashboard does
    # not use GT to make or change the prediction.
    st.markdown("#### Ground-truth agreement")

    evaluation_rows = None
    if evaluation_details_path is not None:
        try:
            evaluation_rows = _patient_auxiliary_rows(evaluation_details_path, patient_id)
        except Exception as exc:
            st.warning(f"Could not read ground-truth evaluation details: {exc}")

    if evaluation_details_path is None or not evaluation_details_path.exists():
        st.info(
            "Ground-truth evaluation is unavailable for this view. Provide a "
            "lesion_tracking_details.csv path in the sidebar to enable it."
        )
    elif evaluation_rows is None or evaluation_rows.empty:
        st.info(
            "The evaluation details file exists, but it contains no scored rows "
            f"for patient {patient_id}."
        )
    else:
        evaluation_row = _event_row(
            evaluation_rows,
            bl_lesion_id=bl_id,
            fu_lesion_id=fu_id,
        )

        if evaluation_row is None:
            st.info(
                "No evaluation row matches this predicted event. Verify that the "
                "evaluation file was generated from the same lesion_matches.csv."
            )
        else:
            gt_status = str(evaluation_row.get("status", "")).upper()
            gt_type = _normalise_optional_id(evaluation_row.get("ground_truth_type")) or "—"
            predicted_type = _normalise_optional_id(evaluation_row.get("predicted_type")) or match_type or "—"
            topology_correct = bool(evaluation_row.get("topology_correct", False))

            if gt_status == "CORRECT" and topology_correct:
                st.success(
                    f"✓ CORRECT — the BL/FU endpoints and event type agree with ground truth "
                    f"({gt_type})."
                )
            elif gt_status == "CORRECT":
                st.warning(
                    "✓ Endpoint correspondence agrees with ground truth, but the event type differs: "
                    f"predicted {predicted_type}, ground truth {gt_type}."
                )
            elif gt_status == "INCORRECT":
                st.error(
                    "✗ INCORRECT — this predicted BL/FU event is not present in the ground-truth "
                    "link set."
                )
            else:
                st.info(f"Evaluation status: {gt_status or 'unknown'}")

            gt_columns = st.columns(3)
            gt_columns[0].metric("Evaluation", gt_status or "—")
            gt_columns[1].metric("Predicted type", predicted_type)
            gt_columns[2].metric("Ground-truth type", gt_type)

        missed = evaluation_rows[
            evaluation_rows.get("status", pd.Series(index=evaluation_rows.index, dtype=str))
            .astype(str)
            .str.upper()
            .eq("MISSED")
        ].copy()
        if not missed.empty:
            missed_columns = [
                column
                for column in (
                    "bl_lesion_id",
                    "fu_lesion_id",
                    "ground_truth_type",
                    "expert_lesion_id",
                )
                if column in missed.columns
            ]
            with st.expander(f"Missed ground-truth events for this patient ({len(missed)})"):
                st.caption(
                    "These events exist in ground truth but are absent from the predicted match set."
                )
                st.dataframe(
                    missed[missed_columns],
                    hide_index=True,
                    use_container_width=True,
                )

        st.caption(f"Evaluation source: {evaluation_details_path}")

    if run_matching and evaluation_details_path is not None and evaluation_details_path.exists():
        st.caption(
            "Note: re-running matching in the dashboard does not automatically recompute "
            "ground-truth evaluation. Re-run the evaluation pipeline before treating the "
            "displayed GT status as current."
        )

    # Preserve existing image-linking behaviour.
    if features.empty:
        return None

    try:
        focus = resolve_match_image_focus(selected_match, features)
    except MatchingDashboardError as exc:
        st.warning(f"Match details are available, but image linking is unavailable: {exc}")
        return None

    st.caption(f"Spatial locations loaded from: {feature_path}")
    return focus


def _centroid_marker_slice(
    shape: tuple[int, int, int],
    centroid_voxel: tuple[float, float, float],
    *,
    axis: int,
    radius: int = 5,
) -> tuple[np.ndarray, int]:
    #draw a 2-D marker at the lesion centroid
    point = np.asarray(centroid_voxel, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"Invalid lesion voxel centroid: {centroid_voxel!r}")
    if any(value < -0.5 or value > size - 0.5 for value, size in zip(point, shape)):
        raise ValueError(
            f"Lesion centroid {centroid_voxel!r} is outside image shape {shape}."
        )
    slice_index = int(np.clip(np.rint(point[axis]), 0, shape[axis] - 1))
    remaining_axes = [value for value in range(3) if value != axis]
    raw_shape = (shape[remaining_axes[0]], shape[remaining_axes[1]])
    raw_marker = np.zeros(raw_shape, dtype=bool)
    centre_row = int(np.clip(np.rint(point[remaining_axes[0]]), 0, raw_shape[0] - 1))
    centre_col = int(np.clip(np.rint(point[remaining_axes[1]]), 0, raw_shape[1] - 1))
    rows, columns = np.ogrid[: raw_shape[0], : raw_shape[1]]
    disk = (rows - centre_row) ** 2 + (columns - centre_col) ** 2 <= radius**2
    raw_marker[disk] = True
    return np.ascontiguousarray(np.flipud(np.rot90(raw_marker))), slice_index

def _render_lesion_focus(
    focus: MatchImageFocus,
    *,
    registered_display: np.ndarray,
    fu_display: np.ndarray,
    fu_ct: LoadedVolume,
    modality: str,
    plane_name: str,
    baseline_mask=None,
    followup_mask=None,
) -> None:
    #show each selected mask on a slice that contains the lesion
    st.divider()
    st.subheader("Selected correspondence on aligned scans")
    status, colour = OUTCOME_STYLES.get(focus.match_type, (focus.match_type, (0, 220, 255)))
    st.markdown(f"**{status}**")
    st.caption("Cyan: automatic match · Amber: appearing · Pink: disappearing. "
               "Both panels highlight the selected segmentation on its largest cross-section.")
    info = plane_info(fu_ct, plane_name)
    row_mm, col_mm = display_pixel_spacing_mm(fu_ct, plane_name)

    def render_location(location, display, title: str, mask) -> None:
        if location is None:
            st.info(f"No {title} lesion for this {focus.match_type} outcome.")
            return
        try:
            selected = selected_lesion_mask(location, fu_ct, mask)
            marker, slice_index = lesion_display_slice(selected, info.voxel_axis)
            image = _uint8_slice(display, slice_index, axis=info.voxel_axis)
            image = _overlay_mask(image, marker, alpha=0.60, colour=colour)
            image = physical_aspect_resize(
                image,
                row_spacing_mm=row_mm,
                column_spacing_mm=col_mm,
            )
        except Exception as exc:
            st.error(f"Could not locate {location.lesion_id} on the scan: {exc}")
            return
        st.markdown(f"**{title}: {location.lesion_id}**")
        st.image(
            image,
            caption=(
                f"{plane_name} slice {slice_index + 1}/{info.slice_count} · "
                f"centroid voxel ({location.centroid_voxel[0]:.1f}, "
                f"{location.centroid_voxel[1]:.1f}, "
                f"{location.centroid_voxel[2]:.1f})"
            ),
            use_container_width=True,
        )

    bl_column, fu_column = st.columns(2)
    with bl_column:
        render_location(focus.baseline, registered_display, f"Registered BL {modality}", baseline_mask)
    with fu_column:
        render_location(focus.followup, fu_display, f"FU {modality}", followup_mask)

def _registration_paths(
    output_root: str | Path,
    patient_id: str,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_root).expanduser() / str(patient_id)
    registered_path = output_dir / "registered_baseline_ct.nii.gz"
    registered_mask_path = output_dir / "registered_baseline_lesion_mask.nii.gz"
    return output_dir, registered_path, registered_mask_path

def _registration_is_ready(output_dir: Path, registered_path: Path) -> bool:
    #return the rigid-registration output paths used by the pipeline
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
    #window CT data once as uint8 so slider changes only extract and rotate slices
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D CT volume, got {data.shape}.")
    if high_hu <= low_hu:
        raise ValueError("CT window upper bound must be greater than lower bound.")

    #work in float32 to reduce temporary memory compared with float64
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

def _prepare_pet_display_volume(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    #normalise PET intensities by percentile once for slice viewing
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D PET volume, got {data.shape}.")
    image = np.asarray(data, dtype=np.float32).copy()
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("PET volume contains no finite intensity values.")
    low, high = np.percentile(finite, (1.0, 99.5))
    low = float(low)
    high = float(high)
    if high <= low:
        high = low + 1.0
    np.nan_to_num(image, copy=False, nan=low, posinf=high, neginf=low)
    np.clip(image, low, high, out=image)
    image -= low
    image *= 255.0 / (high - low)
    return np.rint(image).astype(np.uint8), low, high

def _get_pet_display_volumes(
    *,
    patient_id: str,
    original_bl_pet: LoadedVolume,
    registered_bl_pet: LoadedVolume,
    followup_pet: LoadedVolume,
    registered_pet_path: Path,
    registered_pet_modified_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    signature = (
        "PET",
        str(patient_id),
        str(original_bl_pet.metadata.path),
        _file_modified_ns(original_bl_pet.metadata.path),
        str(registered_pet_path.resolve()),
        int(registered_pet_modified_ns),
        str(followup_pet.metadata.path),
        _file_modified_ns(followup_pet.metadata.path),
    )
    cache = st.session_state.get("_alignment_display_cache")
    if cache is None or cache.get("signature") != signature:
        with st.spinner("Preparing PET display volumes..."):
            original_display, original_low, original_high = _prepare_pet_display_volume(
                original_bl_pet.data
            )
            registered_display, registered_low, registered_high = _prepare_pet_display_volume(
                registered_bl_pet.data
            )
            fu_display, fu_low, fu_high = _prepare_pet_display_volume(
                followup_pet.data
            )
        #normalise PET panels separately to preserve contrast; the shared caption range is only a reference
        st.session_state["_alignment_display_cache"] = {
            "signature": signature,
            "original": original_display,
            "registered": registered_display,
            "followup": fu_display,
            "low_hu": min(original_low, registered_low, fu_low),
            "high_hu": max(original_high, registered_high, fu_high),
        }
        cache = st.session_state["_alignment_display_cache"]
    return (
        cache["original"],
        cache["registered"],
        cache["followup"],
        float(cache["low_hu"]),
        float(cache["high_hu"]),
    )

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
    #identify cached display volumes; keep only the current patient and window to limit memory use
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
    #cache windowed uint8 BL, registered BL and FU volumes per patient and window for slice selection
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

def _uint8_slice(
    volume: np.ndarray,
    index: int,
    *,
    axis: int,
) -> np.ndarray:
    #extract a normalised slice in the selected anatomical plane
    return extract_display_slice(volume, axis=axis, index=index)

def _ras_to_lps(point_xyz: np.ndarray) -> np.ndarray:
    #convert NIfTI RAS coordinates to ITK/Elastix LPS coordinates
    point = np.asarray(point_xyz, dtype=float)
    return np.asarray([-point[0], -point[1], point[2]], dtype=float)

def _lps_to_ras(point_xyz: np.ndarray) -> np.ndarray:
    #convert ITK/Elastix LPS coordinates to NIfTI RAS coordinates
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
    #rebuild the Elastix Euler transform from fixed FU to moving native BL, no inversion is needed
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
    bl_orientation: tuple[str, str, str],
    fu_shape: tuple[int, int, int],
    fu_affine_flat: tuple[float, ...],
    fu_orientation: tuple[str, str, str],
    plane_name: str,
    transform_path: str,
    transform_modified_ns: int,
) -> tuple[float, ...]:
    #map each FU plane centre to the nearest native BL plane of the same orientation
    #registered BL and FU share the FU grid and slice index
    del transform_modified_ns  #cache invalidation only
    bl_affine = np.asarray(bl_affine_flat, dtype=float).reshape(4, 4)
    fu_affine = np.asarray(fu_affine_flat, dtype=float).reshape(4, 4)
    bl_axis = anatomical_axis(bl_orientation, plane_name)
    fu_axis = anatomical_axis(fu_orientation, plane_name)
    try:
        bl_affine_inv = np.linalg.inv(bl_affine)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Original BL affine is not invertible.") from exc
    rigid = _build_elastix_euler_transform(transform_path)
    mapped_axis: list[float] = []
    centre = np.asarray([(size - 1) / 2.0 for size in fu_shape] + [1.0], dtype=float)
    for slice_index in range(fu_shape[fu_axis]):
        fu_voxel = centre.copy()
        fu_voxel[fu_axis] = float(slice_index)
        #nibabel gives RAS physical coordinates
        fu_world_ras = (fu_affine @ fu_voxel)[:3]
        fu_world_lps = _ras_to_lps(fu_world_ras)
        #Elastix transform direction is fixed(FU) -> moving(BL)
        bl_world_lps = np.asarray(
            rigid.TransformPoint(tuple(float(v) for v in fu_world_lps)),
            dtype=float,
        )
        bl_world_ras = _lps_to_ras(bl_world_lps)
        bl_world_h = np.asarray(
            [bl_world_ras[0], bl_world_ras[1], bl_world_ras[2], 1.0],
            dtype=float,
        )
        bl_voxel = bl_affine_inv @ bl_world_h
        mapped_axis.append(float(bl_voxel[bl_axis]))
    return tuple(mapped_axis)

def _mapped_native_bl_slice(
    mapped_z_by_fu_slice: tuple[float, ...],
    *,
    fu_slice_index: int,
    bl_slice_count: int,
) -> tuple[int, float, bool]:
    #find the nearest native BL axial slice using the saved transform, not raw-coordinate proximity
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
    display_low: float,
    display_high: float,
    *,
    modality: str,
    plane_name: str,
    mapped_axis_by_fu_slice: tuple[float, ...],
    show_masks: bool,
    bl_mask: LoadedVolume | None,
    registered_bl_mask: LoadedVolume | None,
    fu_mask: LoadedVolume | None,
) -> None:
    #show original BL, registered BL and FU in three columns
    modality_name = str(modality).upper()
    fu_info = plane_info(fu_ct, plane_name)
    bl_info = plane_info(bl_ct, plane_name)
    st.divider()
    st.subheader("Multiplanar alignment comparison")
    st.caption(
        f"{plane_name} view · Original BL = nearest native {plane_name.lower()} "
        "slice located using the saved rigid FU→BL point mapping. Registered "
        "BL and FU = the exact same FU-space anatomical plane. "
        + ("Lesion masks are ON." if show_masks else "Lesion masks are OFF.")
    )
    slice_count = fu_info.slice_count
    state_key = f"slice_{patient_id}_{plane_name.lower()}"
    if state_key not in st.session_state:
        st.session_state[state_key] = slice_count // 2
    elif not 0 <= int(st.session_state[state_key]) < slice_count:
        st.session_state[state_key] = slice_count // 2
    slice_index = st.slider(
        f"FU-space {plane_name.lower()} slice",
        min_value=0,
        max_value=slice_count - 1,
        step=1,
        key=state_key,
    )
    world_xyz = slice_centre_world_mm(
        fu_ct,
        plane=plane_name,
        index=slice_index,
    )
    original_index, original_float_index, outside_fov = _mapped_native_bl_slice(
        mapped_axis_by_fu_slice,
        fu_slice_index=slice_index,
        bl_slice_count=bl_info.slice_count,
    )
    try:
        original_slice = _uint8_slice(
            original_display,
            original_index,
            axis=bl_info.voxel_axis,
        )
        registered_slice = _uint8_slice(
            registered_display,
            slice_index,
            axis=fu_info.voxel_axis,
        )
        fu_slice = _uint8_slice(
            fu_display,
            slice_index,
            axis=fu_info.voxel_axis,
        )
        original_mask_slice = None
        registered_mask_slice = None
        fu_mask_slice = None
        if show_masks:
            if bl_mask is not None:
                original_mask_slice = _mask_slice(
                    bl_mask.data,
                    original_index,
                    axis=bl_info.voxel_axis,
                )
            if registered_bl_mask is not None:
                registered_mask_slice = _mask_slice(
                    registered_bl_mask.data,
                    slice_index,
                    axis=fu_info.voxel_axis,
                )
            if fu_mask is not None:
                fu_mask_slice = _mask_slice(
                    fu_mask.data,
                    slice_index,
                    axis=fu_info.voxel_axis,
                )
        original_slice = _overlay_mask(original_slice, original_mask_slice)
        registered_slice = _overlay_mask(registered_slice, registered_mask_slice)
        fu_slice = _overlay_mask(fu_slice, fu_mask_slice)
        #preserve image proportions with native BL spacing for original BL and FU spacing for both aligned panels
        bl_row_mm, bl_col_mm = display_pixel_spacing_mm(bl_ct, plane_name)
        fu_row_mm, fu_col_mm = display_pixel_spacing_mm(fu_ct, plane_name)
        original_slice = physical_aspect_resize(
            original_slice,
            row_spacing_mm=bl_row_mm,
            column_spacing_mm=bl_col_mm,
        )
        registered_slice = physical_aspect_resize(
            registered_slice,
            row_spacing_mm=fu_row_mm,
            column_spacing_mm=fu_col_mm,
        )
        fu_slice = physical_aspect_resize(
            fu_slice,
            row_spacing_mm=fu_row_mm,
            column_spacing_mm=fu_col_mm,
        )
    except Exception as exc:
        st.error(f"Could not render current {plane_name.lower()} slices: {exc}")
        return
    if modality_name == "CT":
        intensity_text = f"window [{display_low:.0f}, {display_high:.0f}] HU"
    else:
        intensity_text = (
            f"PET display percentile range ≈ [{display_low:.3g}, {display_high:.3g}]"
        )
    st.caption(
        f"FU-space {plane_name.lower()} slice {slice_index + 1}/{slice_count} · "
        f"world centre ≈ ({world_xyz[0]:.1f}, {world_xyz[1]:.1f}, "
        f"{world_xyz[2]:.1f}) mm · {intensity_text} · "
        "display aspect = physical voxel spacing"
    )
    if outside_fov:
        st.warning(
            f"The rigid-mapped FU position falls outside the native BL "
            f"{plane_name.lower()} range. The Original BL panel is showing "
            f"the nearest edge slice ({original_index})."
        )
    before, after, reference = st.columns(3)
    with before:
        st.markdown(f"**Original BL {modality_name}**")
        st.image(
            original_slice,
            caption=(
                f"Native-BL grid · rigid-mapped {plane_name.lower()} slice "
                f"{original_index} (mapped index={original_float_index:.1f})"
            ),
            use_container_width=True,
        )
    with after:
        st.markdown(f"**Registered BL {modality_name}**")
        st.image(
            registered_slice,
            caption=(
                f"After alignment · FU-space {plane_name.lower()} "
                f"slice {slice_index}"
            ),
            use_container_width=True,
        )
    with reference:
        st.markdown(f"**FU {modality_name}**")
        st.image(
            fu_slice,
            caption=(
                f"Reference · FU-space {plane_name.lower()} slice {slice_index}"
            ),
            use_container_width=True,
        )

def main() -> None:
    st.set_page_config(
        page_title="Longitudinal Multiplanar Viewer",
        page_icon="🩻",
        layout="wide",
    )
    st.title("Longitudinal BL → FU Multiplanar Viewer")
    st.caption(f"**{APP_VERSION}**")
    st.caption(
        "ITKElastix rigid-only registration with linked Axial / Coronal / "
        "Sagittal CT/PET viewing in FU space."
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
            value=(
                os.environ.get("DATA_ROOT")
                or os.environ.get("COHORT_B_ROOT")
                or os.environ.get("COHORT_A_ROOT", "")
            ),
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
        patient_ids = _patient_ids(str(manifest_path), manifest_modified_ns)
    except Exception as exc:
        st.error(f"Could not read pair manifest: {exc}")
        st.stop()
    if not patient_ids:
        st.warning("No patient IDs were found in the pair manifest.")
        st.stop()
    with st.sidebar:
        patient_id = st.selectbox("Patient", options=patient_ids)
        st.header("Alignment")
        st.write("Conservative baseline: **Rigid only**")
        run_alignment = st.button(
            "Run / re-run alignment",
            type="primary",
            use_container_width=True,
        )
        st.header("Viewer")
        display_modality = st.selectbox(
            "Modality",
            options=("CT", "PET"),
            index=0,
            help=(
                "PET is shown on the corresponding CT grids. Baseline PET is "
                "also transformed into FU space using the saved rigid transform."
            ),
        )
        plane_name = st.selectbox(
            "View plane",
            options=PLANE_NAMES,
            index=0,
            help=(
                "Axial = head-to-foot slices, Coronal = front/back slices, "
                "Sagittal = side-to-side slices."
            ),
        )
        if display_modality == "CT":
            window_name = st.selectbox(
                "CT window",
                options=("Soft tissue", "Lung", "Bone", "Wide"),
                index=0,
            )
        else:
            window_name = "Soft tissue"  # unused in PET mode
            st.caption("PET display uses robust percentile normalisation.")
        show_masks = st.checkbox(
            "Show lesion masks",
            value=False,
            help=(
                "Overlay BL/FU lesion masks. The BL mask is transformed into "
                "FU space using the saved rigid transform and nearest-neighbour "
                "interpolation."
            ),
        )
        st.header("Lesion matching")
        match_results_text = st.text_input(
            "Matching results CSV",
            value=DEFAULT_MATCH_RESULTS,
            help=(
                "A combined lesion_matches.csv. Results for the selected patient "
                "are loaded automatically."
            ),
        )
        aligned_features_text = st.text_input(
            "Aligned lesion features CSV",
            value=DEFAULT_ALIGNED_FEATURES,
            help=(
                "Provides FU-grid BL/FU lesion centroids used to locate a "
                "selected correspondence on the aligned scans."
            ),
        )
        cost_matrix_dir_text = st.text_input(
            "Cost matrix directory",
            value=DEFAULT_COST_MATRIX_DIR,
            help="Contains <patient_id>_cost_matrix.csv files.",
        )
        pair_costs_text = st.text_input(
            "Pair-cost details CSV",
            value=DEFAULT_PAIR_COSTS,
            help=(
                "Optional long-form lesion_pair_costs.csv used to show distance, "
                "volume, size difference, PET features, and cost components."
            ),
        )
        evaluation_details_text = st.text_input(
            "Ground-truth evaluation details CSV",
            value=DEFAULT_EVALUATION_DETAILS,
            help=(
                "Optional lesion_tracking_details.csv. When available, the selected "
                "prediction is labelled CORRECT/INCORRECT and missed GT events are shown."
            ),
        )
        with st.expander("Matcher settings"):
            disappearing_penalty = st.number_input(
                "Disappearing penalty", min_value=0.0, value=1.0, step=0.1
            )
            new_lesion_penalty = st.number_input(
                "New lesion penalty", min_value=0.0, value=1.0, step=0.1
            )
            merge_penalty = st.number_input(
                "Merge penalty", min_value=0.0, value=0.2, step=0.1
            )
            max_bl_per_fu = st.number_input(
                "Maximum BL lesions per FU lesion",
                min_value=1,
                value=3,
                step=1,
            )
        run_matching = st.button(
            "Run / re-run lesion matching",
            use_container_width=True,
            help="Runs NetworkX min-cost-flow for the selected patient only.",
        )
    data_root = _normalise_optional_path(data_root_text)
    output_root = Path(output_root_text).expanduser()
    output_dir, registered_path, registered_mask_path = _registration_paths(
        output_root,
        patient_id,
    )
    registered_pet_path = output_dir / "registered_baseline_pet.nii.gz"
    try:
        pair = _load_pair_cached(
            str(manifest_path),
            patient_id,
            data_root,
            manifest_modified_ns,
            display_modality == "PET",
        )
    except Exception as exc:
        st.error(f"Could not load patient '{patient_id}': {exc}")
        st.stop()
    if pair.baseline.ct is None or pair.followup.ct is None:
        st.error("This patient does not have both baseline and follow-up CT loaded.")
        st.stop()
    bl_ct = pair.baseline.ct
    fu_ct = pair.followup.ct
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Patient", patient_id)
    top2.metric("View", plane_name)
    top3.metric("BL CT shape", " × ".join(map(str, bl_ct.data.shape)))
    top4.metric("FU CT shape", " × ".join(map(str, fu_ct.data.shape)))
    match_image_focus = None
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
            _load_registered_ct_cached.clear()
            _load_registered_mask_cached.clear()
            _load_registered_pet_cached.clear()
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
            "The multiplanar viewer is disabled because equal plane indices "
            "would not represent the same physical space."
        )
        st.stop()
    transform_path = output_dir / "TransformParameters.0.txt"
    #build the native-BL mapping for the currently selected anatomical plane
    try:
        mapped_axis_by_fu_slice = _fu_to_bl_slice_map(
            bl_shape=tuple(int(v) for v in bl_ct.data.shape),
            bl_affine_flat=tuple(
                float(v)
                for v in np.asarray(bl_ct.metadata.affine, dtype=float).reshape(-1)
            ),
            bl_orientation=tuple(str(v) for v in bl_ct.metadata.orientation),
            fu_shape=tuple(int(v) for v in fu_ct.data.shape),
            fu_affine_flat=tuple(
                float(v)
                for v in np.asarray(fu_ct.metadata.affine, dtype=float).reshape(-1)
            ),
            fu_orientation=tuple(str(v) for v in fu_ct.metadata.orientation),
            plane_name=plane_name,
            transform_path=str(transform_path),
            transform_modified_ns=transform_path.stat().st_mtime_ns,
        )
    except Exception as exc:
        st.error(
            f"Could not build the FU → Original-BL {plane_name.lower()} mapping "
            f"from the saved rigid transform: {exc}"
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
                _validate_image_mask_geometry(bl_ct, bl_mask, label="Baseline")
            if fu_mask is not None:
                _validate_image_mask_geometry(fu_ct, fu_mask, label="Follow-up")
            if bl_mask is not None:
                with st.spinner("Preparing rigidly transformed BL lesion mask..."):
                    _warp_baseline_mask_to_fu(
                        bl_mask,
                        transform_path=transform_path,
                        registered_mask_path=registered_mask_path,
                    )
                registered_bl_mask = _load_registered_mask_cached(
                    str(registered_mask_path),
                    registered_mask_path.stat().st_mtime_ns,
                )
                _validate_image_mask_geometry(
                    fu_ct,
                    registered_bl_mask,
                    label="Registered BL mask / FU",
                )
        except Exception as exc:
            st.error(f"Could not prepare lesion-mask overlay: {exc}")
            st.stop()
    #prepare the selected image modality on CT-defined BL/FU grids
    if display_modality == "CT":
        try:
            (
                original_display,
                registered_display,
                fu_display,
                display_low,
                display_high,
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
    else:
        bl_pet = pair.baseline.pet
        fu_pet = pair.followup.pet
        if bl_pet is None or fu_pet is None:
            st.error(
                "PET view requires both baseline and follow-up PET paths for this "
                "patient. Switch back to CT or choose a patient with PET data."
            )
            st.stop()

        try:
            with st.spinner("Preparing PET on CT-aligned grids..."):
                #assume PET and CT are already co-registered; resampling only changes the voxel grid
                original_bl_pet = resample_to_reference(
                    bl_pet,
                    bl_ct,
                    order=1,
                    role="baseline PET on BL CT grid",
                )
                followup_pet = resample_to_reference(
                    fu_pet,
                    fu_ct,
                    order=1,
                    role="follow-up PET on FU CT grid",
                )

                _warp_baseline_pet_to_fu(
                    bl_pet,
                    transform_path=transform_path,
                    registered_pet_path=registered_pet_path,
                )
                registered_bl_pet = _load_registered_pet_cached(
                    str(registered_pet_path),
                    registered_pet_path.stat().st_mtime_ns,
                )
            if not _geometry_matches(registered_bl_pet, fu_ct):
                st.error(
                    "Registered BL PET does not share the FU CT output grid. "
                    "PET multiplanar comparison is disabled for this patient."
                )
                st.stop()
            if not _geometry_matches(followup_pet, fu_ct):
                st.error(
                    "FU PET could not be resampled onto the FU CT grid."
                )
                st.stop()
            if not _geometry_matches(original_bl_pet, bl_ct):
                st.error(
                    "BL PET could not be resampled onto the native BL CT grid."
                )
                st.stop()

            (
                original_display,
                registered_display,
                fu_display,
                display_low,
                display_high,
            ) = _get_pet_display_volumes(
                patient_id=patient_id,
                original_bl_pet=original_bl_pet,
                registered_bl_pet=registered_bl_pet,
                followup_pet=followup_pet,
                registered_pet_path=registered_pet_path,
                registered_pet_modified_ns=registered_pet_path.stat().st_mtime_ns,
            )
        except Exception as exc:
            st.error(f"Could not prepare PET display volumes: {exc}")
            st.stop()
    _render_slice_viewer(
        original_display,
        registered_display,
        fu_display,
        bl_ct,
        fu_ct,
        patient_id,
        display_low,
        display_high,
        modality=display_modality,
        plane_name=plane_name,
        mapped_axis_by_fu_slice=mapped_axis_by_fu_slice,
        show_masks=show_masks,
        bl_mask=bl_mask,
        registered_bl_mask=registered_bl_mask,
        fu_mask=fu_mask,
    )

    # Matching is intentionally rendered after the complete alignment viewer.
    st.divider()
    st.header("Lesion Matching")

    evaluation_details_path = None
    if str(evaluation_details_text).strip():
        evaluation_details_path = Path(evaluation_details_text).expanduser()

    match_image_focus = _render_matching_panel(
        patient_id=patient_id,
        results_path=Path(match_results_text).expanduser(),
        feature_path=Path(aligned_features_text).expanduser(),
        matrix_dir=Path(cost_matrix_dir_text).expanduser(),
        pair_cost_path=Path(pair_costs_text).expanduser(),
        evaluation_details_path=evaluation_details_path,
        run_matching=run_matching,
        config=MatcherConfig(
            disappearing_penalty=float(disappearing_penalty),
            new_lesion_penalty=float(new_lesion_penalty),
            merge_penalty=float(merge_penalty),
            max_bl_per_fu=int(max_bl_per_fu),
        ),
    )

    if match_image_focus is not None:
        # The selected correspondence image needs the registered BL mask even if
        # the alignment-view mask checkbox is OFF, so prepare it lazily here.
        if registered_bl_mask is None and bl_mask is not None:
            try:
                _validate_image_mask_geometry(bl_ct, bl_mask, label="Baseline")
                with st.spinner(
                    "Preparing registered BL lesion mask for matching visualisation..."
                ):
                    _warp_baseline_mask_to_fu(
                        bl_mask,
                        transform_path=transform_path,
                        registered_mask_path=registered_mask_path,
                    )
                registered_bl_mask = _load_registered_mask_cached(
                    str(registered_mask_path),
                    registered_mask_path.stat().st_mtime_ns,
                )
                _validate_image_mask_geometry(
                    fu_ct,
                    registered_bl_mask,
                    label="Registered BL mask / FU",
                )
            except Exception as exc:
                st.warning(
                    "Could not prepare the registered BL lesion mask for matching "
                    f"visualisation: {exc}"
                )

        _render_lesion_focus(
            match_image_focus,
            registered_display=_prepare_display_volume(
                registered_bl.data, low_hu=-160, high_hu=240
            ),
            fu_display=_prepare_display_volume(
                fu_ct.data, low_hu=-160, high_hu=240
            ),
            fu_ct=fu_ct,
            modality="CT",
            plane_name=plane_name,
            baseline_mask=registered_bl_mask,
            followup_mask=fu_mask,
        )
    with st.expander("Registration / viewer details"):
        st.write("BL CT source:", str(bl_ct.metadata.path))
        st.write("FU CT source:", str(fu_ct.metadata.path))
        st.write("Registered BL CT:", str(registered_path))
        st.write("Output directory:", str(output_dir))
        st.write("Rigid transform:", str(transform_path))
        st.write("Selected modality:", display_modality)
        st.write("Selected anatomical plane:", plane_name)
        st.write(
            "Original-BL plane selection:",
            "The centre of the selected FU anatomical plane is mapped through "
            "the saved Elastix fixed(FU) → moving(BL) rigid transform and then "
            "rounded to the nearest native BL plane of the same orientation.",
        )
        if registered_mask_path.exists():
            st.write("Registered BL lesion mask:", str(registered_mask_path))
        if registered_pet_path.exists():
            st.write("Registered BL PET:", str(registered_pet_path))
        st.write("Registered BL shape:", registered_bl.data.shape)
        st.write("FU shape:", fu_ct.data.shape)
        st.write("BL orientation:", bl_ct.metadata.orientation)
        st.write("FU orientation:", fu_ct.metadata.orientation)
        st.write("FU spacing (mm):", fu_ct.metadata.spacing_mm)

if __name__ == "__main__":
    main()
