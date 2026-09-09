from __future__ import annotations

"""
Evaluate longitudinal lesion-tracking predictions against Cohort A expert CSVs.

Primary metric:
    tracking_accuracy = correct predicted links / number of ground-truth links

The script also reports precision / recall / F1 and a stricter topology-aware
score.  Ground-truth expert lesion IDs are NOT assumed to be the same as the
temporary connected-component IDs used by the matcher. Instead, expert cog_bl
and cog_fu voxel coordinates are mapped to native-timepoint component centroids.
Aligned/FU-space centroids remain reserved for the matching algorithm itself.

Example (single patient)
------------------------
python tools/evaluate_lesion_tracking_accuracy.py ^
  --ground-truth "data/0a09c8844b.csv" ^
  --predictions outputs/lesion_matches.csv ^
  --features outputs/aligned_lesion_features.csv ^
  --patient-id 0a09c8844b ^
  --out-details outputs/lesion_tracking_details.csv ^
  --out-summary outputs/lesion_tracking_summary.csv

Example (multiple patients)
---------------------------
python tools/evaluate_lesion_tracking_accuracy.py ^
  --ground-truth data/gt/0a09c8844b.csv data/gt/0aa1883c64.csv ^
  --predictions outputs/lesion_matches.csv ^
  --features outputs/aligned_features/0a09c8844b.csv outputs/aligned_features/0aa1883c64.csv
"""

import argparse
from pathlib import Path
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd


class TrackingEvaluationError(ValueError):
    """Raised when tracking predictions cannot be evaluated safely."""


GT_TYPE_MAP = {
    "UNCHANGED": "MATCHED",
    "MATCHED": "MATCHED",
    "MERGING": "MERGING",
    "DISAPPEARING": "DISAPPEARING",
    "NEWLYAPPEARING": "NEW",
    "NEW": "NEW",
}

PRED_TYPE_MAP = {
    "UNCHANGED": "MATCHED",
    "MATCHED": "MATCHED",
    "MERGING": "MERGING",
    "DISAPPEARING": "DISAPPEARING",
    "NEWLYAPPEARING": "NEW",
    "NEW": "NEW",
}


def _read_csv(path: str | Path, description: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} CSV does not exist: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise TrackingEvaluationError(
            f"Could not read {description} CSV '{path}': {exc}"
        ) from exc


def _normalise_optional_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _parse_bool(value: object, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", ""}:
        return False
    return default


def _parse_xyz(value: object) -> np.ndarray | None:
    """Parse Cohort A coordinates stored as 'x y z' text."""
    if pd.isna(value):
        return None

    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        parts = re.split(r"[\s,;]+", text)
        try:
            arr = np.asarray([float(x) for x in parts if x != ""], dtype=float)
        except ValueError:
            return None

    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return None
    return arr


def _infer_patient_id_from_filename(path: str | Path) -> str:
    stem = Path(path).stem

    # Handles names such as:
    #   0a09c8844b.csv
    #   0a09c8844b(2).csv
    #   0a09c8844b_reference.csv
    match = re.match(r"([A-Za-z0-9]+)", stem)
    if not match:
        raise TrackingEvaluationError(
            f"Could not infer patient ID from ground-truth filename: {path}"
        )
    return match.group(1)



def _normalise_image_id(value: object) -> int | None:
    """
    Normalise Cohort A img_id_bl/img_id_fu values.

    Handles values such as 0, 0.0, "0", "00", "1", and blank/NaN.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    try:
        numeric = float(text)
    except ValueError as exc:
        raise TrackingEvaluationError(
            f"Invalid Cohort A image ID value: {value!r}"
        ) from exc

    if not math.isfinite(numeric) or not numeric.is_integer():
        raise TrackingEvaluationError(
            f"Cohort A image ID must be an integer; got {value!r}."
        )

    return int(numeric)


def filter_ground_truth_to_scan_pair(
    ground_truth: pd.DataFrame,
    *,
    bl_image_id: int,
    fu_image_id: int,
) -> pd.DataFrame:
    """
    Keep only expert rows belonging to the selected BL/FU image pair.

    Cohort A reference CSVs may contain lesion correspondences from multiple
    image pairs for the same patient (for example img_id_bl/img_id_fu 0 and 1).
    The matcher, however, operates on one manifest-selected pair such as
    BL_00 -> FU_00. Therefore ground truth MUST be filtered before centroid
    mapping.

    Rule:
      - when img_id_bl is present, it must equal bl_image_id;
      - when img_id_fu is present, it must equal fu_image_id;
      - a missing BL image ID is only allowed for a NEW lesion;
      - a missing FU image ID is only allowed for a DISAPPEARING lesion.

    This preserves pair identity while remaining safe for endpoint-missing rows.
    """
    required = {"topology_class", "img_id_bl", "img_id_fu"}
    missing = sorted(required - set(ground_truth.columns))
    if missing:
        raise TrackingEvaluationError(
            "Ground-truth CSV cannot be filtered to the selected scan pair "
            "because it is missing: " + ", ".join(missing)
        )

    selected_rows: list[int] = []

    for index, row in ground_truth.iterrows():
        raw_type = str(row["topology_class"]).strip().upper()
        normalised_type = GT_TYPE_MAP.get(raw_type)
        if normalised_type is None:
            raise TrackingEvaluationError(
                f"Unsupported ground-truth topology_class={raw_type!r}."
            )

        row_bl = _normalise_image_id(row.get("img_id_bl"))
        row_fu = _normalise_image_id(row.get("img_id_fu"))

        if row_bl is None:
            bl_ok = normalised_type == "NEW"
        else:
            bl_ok = row_bl == int(bl_image_id)

        if row_fu is None:
            fu_ok = normalised_type == "DISAPPEARING"
        else:
            fu_ok = row_fu == int(fu_image_id)

        if bl_ok and fu_ok:
            selected_rows.append(index)

    result = ground_truth.loc[selected_rows].copy()

    if result.empty:
        patient_text = ""
        if "patient_id" in ground_truth.columns:
            patient_ids = ground_truth["patient_id"].dropna().astype(str).unique()
            if len(patient_ids) == 1:
                patient_text = f" for patient {patient_ids[0]}"

        raise TrackingEvaluationError(
            f"No ground-truth rows remain{patient_text} after filtering to "
            f"BL image {bl_image_id} / FU image {fu_image_id}."
        )

    result["_selected_bl_image_id"] = int(bl_image_id)
    result["_selected_fu_image_id"] = int(fu_image_id)

    return result.reset_index(drop=True)


def load_ground_truths(
    paths: Iterable[str | Path],
    *,
    patient_id_override: str | None = None,
    include_non_challenge: bool = False,
    include_unclear: bool = False,
) -> pd.DataFrame:
    paths = list(paths)
    if not paths:
        raise TrackingEvaluationError("At least one ground-truth CSV is required.")

    if patient_id_override is not None and len(paths) != 1:
        raise TrackingEvaluationError(
            "--patient-id can only be used with exactly one ground-truth CSV."
        )

    frames: list[pd.DataFrame] = []

    for raw_path in paths:
        frame = _read_csv(raw_path, "ground-truth")

        required = {"lesion_id", "topology_class"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise TrackingEvaluationError(
                f"Ground-truth CSV '{raw_path}' is missing: {', '.join(missing)}"
            )

        if "patient_id" not in frame.columns:
            patient_id = (
                patient_id_override
                if patient_id_override is not None
                else _infer_patient_id_from_filename(raw_path)
            )
            frame["patient_id"] = str(patient_id)
        else:
            frame["patient_id"] = frame["patient_id"].astype(str).str.strip()

        if not include_non_challenge and "use_for_challenge" in frame.columns:
            keep = frame["use_for_challenge"].map(lambda x: _parse_bool(x, default=True))
            frame = frame[keep].copy()

        if not include_unclear and "linking_unclear" in frame.columns:
            unclear = frame["linking_unclear"].map(
                lambda x: _parse_bool(x, default=False)
            )
            frame = frame[~unclear].copy()

        frame["_gt_source_path"] = str(raw_path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True, sort=False)
    result["patient_id"] = result["patient_id"].astype(str).str.strip()
    result["topology_class"] = (
        result["topology_class"].astype(str).str.strip().str.upper()
    )
    return result


def load_aligned_features(paths: Iterable[str | Path]) -> pd.DataFrame:
    paths = list(paths)
    if not paths:
        raise TrackingEvaluationError(
            "At least one aligned lesion-feature CSV is required."
        )

    frames = [_read_csv(path, "aligned feature") for path in paths]
    features = pd.concat(frames, ignore_index=True, sort=False)

    if "lesion_id" not in features.columns:
        if "temporary_lesion_id" in features.columns:
            features = features.rename(columns={"temporary_lesion_id": "lesion_id"})
        else:
            raise TrackingEvaluationError(
                "Aligned features require lesion_id or temporary_lesion_id."
            )

    required = {
        "patient_id",
        "timepoint",
        "lesion_id",
        "centroid_x_vox",
        "centroid_y_vox",
        "centroid_z_vox",
        "native_centroid_x_vox",
        "native_centroid_y_vox",
        "native_centroid_z_vox",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise TrackingEvaluationError(
            "Aligned feature CSV is missing: " + ", ".join(missing)
        )

    features = features.copy()
    features["patient_id"] = features["patient_id"].astype(str).str.strip()
    features["timepoint"] = features["timepoint"].astype(str).str.strip().str.upper()
    features["lesion_id"] = features["lesion_id"].astype(str).str.strip()

    unknown_tp = sorted(set(features["timepoint"]) - {"BL", "FU"})
    if unknown_tp:
        raise TrackingEvaluationError(
            "Feature timepoint must contain only BL/FU; found: "
            + ", ".join(unknown_tp)
        )

    for column in (
        "centroid_x_vox",
        "centroid_y_vox",
        "centroid_z_vox",
        "native_centroid_x_vox",
        "native_centroid_y_vox",
        "native_centroid_z_vox",
    ):
        features[column] = pd.to_numeric(features[column], errors="coerce")

    aligned_coords = features[
        ["centroid_x_vox", "centroid_y_vox", "centroid_z_vox"]
    ].to_numpy(dtype=float)
    native_coords = features[
        [
            "native_centroid_x_vox",
            "native_centroid_y_vox",
            "native_centroid_z_vox",
        ]
    ].to_numpy(dtype=float)

    if not np.all(np.isfinite(aligned_coords)):
        raise TrackingEvaluationError(
            "Aligned feature voxel centroids must contain finite x/y/z values."
        )
    if not np.all(np.isfinite(native_coords)):
        raise TrackingEvaluationError(
            "Native feature voxel centroids must contain finite x/y/z values. "
            "Regenerate features with the identity-preserving batch extractor."
        )

    duplicate = features.duplicated(
        ["patient_id", "timepoint", "lesion_id"], keep=False
    )
    if duplicate.any():
        example = features.loc[
            duplicate, ["patient_id", "timepoint", "lesion_id"]
        ].iloc[0]
        raise TrackingEvaluationError(
            "Duplicate feature lesion ID found: "
            f"{example['patient_id']} {example['timepoint']} "
            f"{example['lesion_id']}"
        )

    return features


def load_predictions(path: str | Path) -> pd.DataFrame:
    predictions = _read_csv(path, "prediction")

    required = {"patient_id", "bl_lesion_id", "fu_lesion_id", "match_type"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise TrackingEvaluationError(
            "Prediction CSV is missing: " + ", ".join(missing)
        )

    predictions = predictions.copy()
    predictions["patient_id"] = predictions["patient_id"].astype(str).str.strip()
    predictions["match_type"] = (
        predictions["match_type"].astype(str).str.strip().str.upper()
    )
    predictions["normalised_type"] = predictions["match_type"].map(PRED_TYPE_MAP)

    unsupported = predictions["normalised_type"].isna()
    if unsupported.any():
        values = sorted(predictions.loc[unsupported, "match_type"].unique())
        raise TrackingEvaluationError(
            "Unsupported prediction match_type value(s): " + ", ".join(values)
        )

    predictions["bl_lesion_id"] = predictions["bl_lesion_id"].map(
        _normalise_optional_id
    )
    predictions["fu_lesion_id"] = predictions["fu_lesion_id"].map(
        _normalise_optional_id
    )

    return predictions


def _nearest_feature(
    coord: np.ndarray,
    feature_rows: pd.DataFrame,
    *,
    max_distance_vox: float,
    coordinate_columns: tuple[str, str, str],
    coordinate_label: str,
) -> tuple[str, float]:
    """Map an expert CoG to the nearest temporary lesion in one voxel space."""
    if feature_rows.empty:
        raise TrackingEvaluationError(
            f"No lesion features are available for {coordinate_label} mapping."
        )

    xyz = feature_rows[list(coordinate_columns)].to_numpy(dtype=float)

    distances = np.linalg.norm(xyz - coord[None, :], axis=1)
    best_pos = int(np.argmin(distances))
    best_distance = float(distances[best_pos])

    if best_distance > max_distance_vox:
        raise TrackingEvaluationError(
            f"Nearest temporary lesion in {coordinate_label} is "
            f"{best_distance:.2f} voxels away, exceeding "
            f"--max-map-distance-vox={max_distance_vox:.2f}."
        )

    lesion_id = str(feature_rows.iloc[best_pos]["lesion_id"])
    return lesion_id, best_distance

def _bl_suffix_fallback(
    patient_id: str,
    expert_lesion_id: object,
    bl_features: pd.DataFrame,
) -> str | None:
    """Last-resort BL mapping when cog_propagated is unavailable.

    This is deliberately used only for BL lesions.  FU temporary component
    numbers must NOT be assumed to equal expert lesion IDs, especially in merge
    cases where several expert BL lesions become one connected FU component.
    """
    try:
        numeric_id = int(float(expert_lesion_id))
    except (TypeError, ValueError):
        return None

    suffix = f"_BL_L{numeric_id:03d}"
    candidates = [
        str(value)
        for value in bl_features["lesion_id"]
        if str(value).endswith(suffix)
    ]

    if len(candidates) == 1:
        return candidates[0]
    return None


def map_ground_truth_to_temporary_ids(
    ground_truth: pd.DataFrame,
    features: pd.DataFrame,
    *,
    max_distance_vox: float = 30.0,
) -> pd.DataFrame:
    """
    Map expert GT rows onto the temporary IDs used by the matcher.

    Identity mapping intentionally uses native scan coordinates:

        GT cog_bl  -> BL native_centroid_*_vox
        GT cog_fu  -> FU native_centroid_*_vox

    It does NOT use cog_propagated to identify BL lesions.  The latter may have
    been produced by a different registration pipeline and should not determine
    lesion identity for this evaluator.
    """
    rows: list[dict[str, object]] = []

    native_columns = (
        "native_centroid_x_vox",
        "native_centroid_y_vox",
        "native_centroid_z_vox",
    )

    for patient_id, patient_gt in ground_truth.groupby("patient_id", sort=True):
        patient_features = features[features["patient_id"] == patient_id]
        bl_features = patient_features[patient_features["timepoint"] == "BL"]
        fu_features = patient_features[patient_features["timepoint"] == "FU"]

        if patient_features.empty:
            raise TrackingEvaluationError(
                f"No aligned features found for patient {patient_id}."
            )

        # A MERGING source row may omit its own cog_fu. In that case, reuse the
        # target expert lesion's FU centroid indicated by merged_into.
        target_fu_coords: dict[str, np.ndarray] = {}
        for _, gt_row in patient_gt.iterrows():
            coord = _parse_xyz(gt_row.get("cog_fu"))
            lesion_key = _normalise_optional_id(gt_row.get("lesion_id"))
            if coord is not None and lesion_key is not None:
                target_fu_coords[lesion_key] = coord

        used_bl_ids: dict[str, object] = {}

        for source_index, gt_row in patient_gt.iterrows():
            raw_type = str(gt_row["topology_class"]).strip().upper()
            normalised_type = GT_TYPE_MAP.get(raw_type)
            if normalised_type is None:
                raise TrackingEvaluationError(
                    f"Unsupported ground-truth topology_class={raw_type!r} "
                    f"for patient {patient_id}."
                )

            expert_lesion_id = gt_row.get("lesion_id")

            bl_id: str | None = None
            fu_id: str | None = None
            bl_map_distance: float | None = None
            fu_map_distance: float | None = None
            bl_map_method: str | None = None
            fu_map_method: str | None = None

            # ----------------------------------------------------------
            # BL endpoint: identify from the ORIGINAL BL scan.
            # ----------------------------------------------------------
            if normalised_type != "NEW":
                bl_coord = _parse_xyz(gt_row.get("cog_bl"))

                if bl_coord is not None:
                    bl_id, bl_map_distance = _nearest_feature(
                        bl_coord,
                        bl_features,
                        max_distance_vox=max_distance_vox,
                        coordinate_columns=native_columns,
                        coordinate_label="native BL voxel space",
                    )
                    bl_map_method = "cog_bl->native_BL_centroid"
                else:
                    # Last-resort fallback only when the expert BL coordinate
                    # itself is absent and the temporary ID suffix is unique.
                    bl_id = _bl_suffix_fallback(
                        patient_id,
                        expert_lesion_id,
                        bl_features,
                    )
                    if bl_id is None:
                        raise TrackingEvaluationError(
                            f"Cannot map BL expert lesion {expert_lesion_id!r} "
                            f"for patient {patient_id}: cog_bl is missing and "
                            "no safe BL suffix fallback was found."
                        )
                    bl_map_method = "BL_ID_SUFFIX_FALLBACK"

                # One expert BL lesion row must map to one unique BL temporary
                # component. FU duplicates are allowed for true merge events.
                if bl_id in used_bl_ids:
                    raise TrackingEvaluationError(
                        f"Two GT rows for patient {patient_id} mapped to the same "
                        f"native BL temporary lesion {bl_id}. Previous expert "
                        f"lesion={used_bl_ids[bl_id]!r}, "
                        f"current={expert_lesion_id!r}. "
                        "This suggests the native component extraction and the "
                        "expert lesion identities do not agree."
                    )
                used_bl_ids[bl_id] = expert_lesion_id

            # ----------------------------------------------------------
            # FU endpoint: identify directly from the native FU scan.
            # ----------------------------------------------------------
            if normalised_type != "DISAPPEARING":
                fu_coord = _parse_xyz(gt_row.get("cog_fu"))

                if fu_coord is None and raw_type == "MERGING":
                    merged_into = _normalise_optional_id(
                        gt_row.get("merged_into")
                    )
                    if merged_into is not None:
                        fu_coord = target_fu_coords.get(merged_into)

                if fu_coord is None:
                    raise TrackingEvaluationError(
                        f"Cannot map FU endpoint for expert lesion "
                        f"{expert_lesion_id!r} of patient {patient_id}: "
                        "cog_fu is missing."
                    )

                fu_id, fu_map_distance = _nearest_feature(
                    fu_coord,
                    fu_features,
                    max_distance_vox=max_distance_vox,
                    coordinate_columns=native_columns,
                    coordinate_label="native FU voxel space",
                )
                fu_map_method = "cog_fu->native_FU_centroid"

            rows.append(
                {
                    "patient_id": str(patient_id),
                    "expert_lesion_id": expert_lesion_id,
                    "ground_truth_topology": raw_type,
                    "normalised_type": normalised_type,
                    "gt_img_id_bl": _normalise_image_id(gt_row.get("img_id_bl")),
                    "gt_img_id_fu": _normalise_image_id(gt_row.get("img_id_fu")),
                    "selected_bl_image_id": (
                        _normalise_image_id(
                            gt_row.get("_selected_bl_image_id")
                        )
                    ),
                    "selected_fu_image_id": (
                        _normalise_image_id(
                            gt_row.get("_selected_fu_image_id")
                        )
                    ),
                    "bl_lesion_id": bl_id,
                    "fu_lesion_id": fu_id,
                    "bl_map_distance_vox": bl_map_distance,
                    "fu_map_distance_vox": fu_map_distance,
                    "bl_map_method": bl_map_method,
                    "fu_map_method": fu_map_method,
                    "source_row_index": int(source_index),
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "expert_lesion_id",
                "ground_truth_topology",
                "normalised_type",
                "gt_img_id_bl",
                "gt_img_id_fu",
                "selected_bl_image_id",
                "selected_fu_image_id",
                "bl_lesion_id",
                "fu_lesion_id",
                "bl_map_distance_vox",
                "fu_map_distance_vox",
                "bl_map_method",
                "fu_map_method",
                "source_row_index",
            ]
        )

    duplicate = result.duplicated(
        ["patient_id", "bl_lesion_id", "fu_lesion_id", "normalised_type"],
        keep=False,
    )
    if duplicate.any():
        example = result.loc[
            duplicate,
            [
                "patient_id",
                "expert_lesion_id",
                "bl_lesion_id",
                "fu_lesion_id",
                "normalised_type",
            ],
        ].iloc[0]
        raise TrackingEvaluationError(
            "Duplicate canonical ground-truth event after native-space mapping: "
            f"{dict(example)}"
        )

    return result

def _link_key(row: pd.Series | object) -> tuple[str, str | None, str | None]:
    return (
        str(row.patient_id),
        _normalise_optional_id(row.bl_lesion_id),
        _normalise_optional_id(row.fu_lesion_id),
    )


def _strict_key(
    row: pd.Series | object,
) -> tuple[str, str | None, str | None, str]:
    return (
        str(row.patient_id),
        _normalise_optional_id(row.bl_lesion_id),
        _normalise_optional_id(row.fu_lesion_id),
        str(row.normalised_type),
    )


def _safe_div(num: int | float, den: int | float) -> float:
    return float(num / den) if den else 0.0


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    jaccard = _safe_div(tp, tp + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard_accuracy": jaccard,
    }


def evaluate_tracking(
    mapped_ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = mapped_ground_truth.copy()
    pred = predictions.copy()

    gt_patients = set(gt["patient_id"].astype(str))
    pred = pred[pred["patient_id"].astype(str).isin(gt_patients)].copy()

    gt_link_rows = {_link_key(row): row for row in gt.itertuples(index=False)}
    pred_link_rows = {_link_key(row): row for row in pred.itertuples(index=False)}

    gt_links = set(gt_link_rows)
    pred_links = set(pred_link_rows)

    correct_links = gt_links & pred_links
    incorrect_links = pred_links - gt_links
    missed_links = gt_links - pred_links

    gt_strict = {_strict_key(row) for row in gt.itertuples(index=False)}
    pred_strict = {_strict_key(row) for row in pred.itertuples(index=False)}

    strict_tp = len(gt_strict & pred_strict)
    strict_fp = len(pred_strict - gt_strict)
    strict_fn = len(gt_strict - pred_strict)

    link_metrics = _metrics(
        len(correct_links),
        len(incorrect_links),
        len(missed_links),
    )
    strict_metrics = _metrics(strict_tp, strict_fp, strict_fn)

    # "Basic tracking accuracy" for the Jira story:
    # proportion of expert-confirmed correspondences recovered correctly.
    tracking_accuracy = _safe_div(len(correct_links), len(gt_links))
    topology_accuracy = _safe_div(strict_tp, len(gt_strict))

    summary_rows: list[dict[str, object]] = []

    def add_summary(
        patient_id: str,
        patient_gt: pd.DataFrame,
        patient_pred: pd.DataFrame,
    ) -> None:
        p_gt_links = {_link_key(r) for r in patient_gt.itertuples(index=False)}
        p_pred_links = {_link_key(r) for r in patient_pred.itertuples(index=False)}

        p_tp = len(p_gt_links & p_pred_links)
        p_fp = len(p_pred_links - p_gt_links)
        p_fn = len(p_gt_links - p_pred_links)
        p_link_metrics = _metrics(p_tp, p_fp, p_fn)

        p_gt_strict = {_strict_key(r) for r in patient_gt.itertuples(index=False)}
        p_pred_strict = {_strict_key(r) for r in patient_pred.itertuples(index=False)}
        p_stp = len(p_gt_strict & p_pred_strict)
        p_sfp = len(p_pred_strict - p_gt_strict)
        p_sfn = len(p_gt_strict - p_pred_strict)
        p_strict_metrics = _metrics(p_stp, p_sfp, p_sfn)

        summary_rows.append(
            {
                "patient_id": patient_id,
                "ground_truth_links": len(p_gt_links),
                "predicted_links": len(p_pred_links),
                "correct_matches": p_tp,
                "incorrect_matches": p_fp,
                "missed_matches": p_fn,
                "tracking_accuracy": _safe_div(p_tp, len(p_gt_links)),
                "precision": p_link_metrics["precision"],
                "recall": p_link_metrics["recall"],
                "f1": p_link_metrics["f1"],
                "strict_link_accuracy": p_link_metrics["jaccard_accuracy"],
                "topology_correct": p_stp,
                "topology_accuracy": _safe_div(p_stp, len(p_gt_strict)),
                "topology_precision": p_strict_metrics["precision"],
                "topology_recall": p_strict_metrics["recall"],
                "topology_f1": p_strict_metrics["f1"],
            }
        )

    for patient_id in sorted(gt_patients):
        add_summary(
            patient_id,
            gt[gt["patient_id"] == patient_id],
            pred[pred["patient_id"] == patient_id],
        )

    summary_rows.append(
        {
            "patient_id": "ALL",
            "ground_truth_links": len(gt_links),
            "predicted_links": len(pred_links),
            "correct_matches": len(correct_links),
            "incorrect_matches": len(incorrect_links),
            "missed_matches": len(missed_links),
            "tracking_accuracy": tracking_accuracy,
            "precision": link_metrics["precision"],
            "recall": link_metrics["recall"],
            "f1": link_metrics["f1"],
            "strict_link_accuracy": link_metrics["jaccard_accuracy"],
            "topology_correct": strict_tp,
            "topology_accuracy": topology_accuracy,
            "topology_precision": strict_metrics["precision"],
            "topology_recall": strict_metrics["recall"],
            "topology_f1": strict_metrics["f1"],
        }
    )

    summary = pd.DataFrame(summary_rows)

    detail_rows: list[dict[str, object]] = []

    for key in sorted(correct_links, key=str):
        gt_row = gt_link_rows[key]
        pred_row = pred_link_rows[key]
        topology_correct = str(gt_row.normalised_type) == str(pred_row.normalised_type)
        detail_rows.append(
            {
                "status": "CORRECT",
                "patient_id": key[0],
                "bl_lesion_id": key[1],
                "fu_lesion_id": key[2],
                "ground_truth_type": gt_row.normalised_type,
                "predicted_type": pred_row.normalised_type,
                "topology_correct": topology_correct,
                "expert_lesion_id": gt_row.expert_lesion_id,
                "bl_map_distance_vox": gt_row.bl_map_distance_vox,
                "fu_map_distance_vox": gt_row.fu_map_distance_vox,
            }
        )

    for key in sorted(incorrect_links, key=str):
        pred_row = pred_link_rows[key]
        detail_rows.append(
            {
                "status": "INCORRECT",
                "patient_id": key[0],
                "bl_lesion_id": key[1],
                "fu_lesion_id": key[2],
                "ground_truth_type": None,
                "predicted_type": pred_row.normalised_type,
                "topology_correct": False,
                "expert_lesion_id": None,
                "bl_map_distance_vox": None,
                "fu_map_distance_vox": None,
            }
        )

    for key in sorted(missed_links, key=str):
        gt_row = gt_link_rows[key]
        detail_rows.append(
            {
                "status": "MISSED",
                "patient_id": key[0],
                "bl_lesion_id": key[1],
                "fu_lesion_id": key[2],
                "ground_truth_type": gt_row.normalised_type,
                "predicted_type": None,
                "topology_correct": False,
                "expert_lesion_id": gt_row.expert_lesion_id,
                "bl_map_distance_vox": gt_row.bl_map_distance_vox,
                "fu_map_distance_vox": gt_row.fu_map_distance_vox,
            }
        )

    details = pd.DataFrame(detail_rows)

    return summary, details


def _write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate min-cost-flow lesion tracking against Cohort A expert "
            "correspondence CSVs."
        )
    )
    parser.add_argument(
        "--ground-truth",
        nargs="+",
        required=True,
        help="One or more Cohort A expert correspondence CSV files.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Matcher output CSV, normally outputs/lesion_matches.csv.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help=(
            "One or more aligned lesion-feature CSVs used to map expert "
            "coordinates to temporary matcher lesion IDs."
        ),
    )
    parser.add_argument(
        "--patient-id",
        help=(
            "Patient ID override for a single ground-truth CSV. Useful when the "
            "ground-truth filename is not exactly the patient ID."
        ),
    )
    parser.add_argument(
        "--bl-image-id",
        type=int,
        help=(
            "Optional Cohort A baseline image index. When supplied together "
            "with --fu-image-id, ground truth is filtered to that scan pair "
            "before lesion mapping."
        ),
    )
    parser.add_argument(
        "--fu-image-id",
        type=int,
        help=(
            "Optional Cohort A follow-up image index. Must be supplied together "
            "with --bl-image-id."
        ),
    )
    parser.add_argument(
        "--max-map-distance-vox",
        type=float,
        default=30.0,
        help=(
            "Maximum native voxel-index distance when mapping Cohort A "
            "cog_bl/cog_fu to a temporary lesion identity. Default: 30 voxels."
        ),
    )
    parser.add_argument(
        "--include-non-challenge",
        action="store_true",
        help="Do not filter out use_for_challenge=False rows.",
    )
    parser.add_argument(
        "--include-unclear",
        action="store_true",
        help="Include linking_unclear=True rows.",
    )
    parser.add_argument(
        "--out-summary",
        default="outputs/lesion_tracking_summary.csv",
        help="Per-patient and overall metric CSV.",
    )
    parser.add_argument(
        "--out-details",
        default="outputs/lesion_tracking_details.csv",
        help="Detailed correct / incorrect / missed result CSV.",
    )
    parser.add_argument(
        "--out-mapping",
        default="outputs/lesion_tracking_gt_mapping.csv",
        help="Ground-truth expert-to-temporary-ID mapping CSV for audit/debugging.",
    )
    args = parser.parse_args()

    if not math.isfinite(args.max_map_distance_vox) or args.max_map_distance_vox <= 0:
        parser.error("--max-map-distance-vox must be a finite value > 0.")

    if (args.bl_image_id is None) != (args.fu_image_id is None):
        parser.error("--bl-image-id and --fu-image-id must be supplied together.")

    ground_truth = load_ground_truths(
        args.ground_truth,
        patient_id_override=args.patient_id,
        include_non_challenge=args.include_non_challenge,
        include_unclear=args.include_unclear,
    )

    if args.bl_image_id is not None:
        ground_truth = filter_ground_truth_to_scan_pair(
            ground_truth,
            bl_image_id=args.bl_image_id,
            fu_image_id=args.fu_image_id,
        )

    features = load_aligned_features(args.features)
    predictions = load_predictions(args.predictions)

    mapped_gt = map_ground_truth_to_temporary_ids(
        ground_truth,
        features,
        max_distance_vox=args.max_map_distance_vox,
    )

    summary, details = evaluate_tracking(mapped_gt, predictions)

    summary_path = _write_csv(summary, args.out_summary)
    details_path = _write_csv(details, args.out_details)
    mapping_path = _write_csv(mapped_gt, args.out_mapping)

    overall = summary[summary["patient_id"] == "ALL"].iloc[0]

    print("Lesion tracking evaluation")
    print("--------------------------")
    print("GT identity mapping : cog_bl/cog_fu -> native lesion centroids")
    if args.bl_image_id is not None:
        print(
            f"GT scan-pair filter : BL_{args.bl_image_id:02d} -> "
            f"FU_{args.fu_image_id:02d}"
        )
    print(f"Ground-truth links : {int(overall['ground_truth_links'])}")
    print(f"Predicted links    : {int(overall['predicted_links'])}")
    print(f"Correct matches    : {int(overall['correct_matches'])}")
    print(f"Incorrect matches  : {int(overall['incorrect_matches'])}")
    print(f"Missed matches     : {int(overall['missed_matches'])}")
    print()
    print(f"Tracking accuracy  : {overall['tracking_accuracy']:.3f}")
    print(f"Precision          : {overall['precision']:.3f}")
    print(f"Recall             : {overall['recall']:.3f}")
    print(f"F1                 : {overall['f1']:.3f}")
    print(f"Strict link score  : {overall['strict_link_accuracy']:.3f}")
    print()
    print(f"Topology accuracy  : {overall['topology_accuracy']:.3f}")
    print(f"Topology F1        : {overall['topology_f1']:.3f}")
    print()
    print(f"Summary CSV        : {summary_path}")
    print(f"Details CSV        : {details_path}")
    print(f"GT mapping CSV     : {mapping_path}")


if __name__ == "__main__":
    main()
