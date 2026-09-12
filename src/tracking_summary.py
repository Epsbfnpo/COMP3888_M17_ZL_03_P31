from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math

import pandas as pd


class TrackingSummaryError(ValueError):
    """Raised when tracking-summary inputs are missing or inconsistent."""


REQUIRED_MATCH_COLUMNS = {
    "patient_id",
    "bl_lesion_id",
    "fu_lesion_id",
    "match_type",
}

#evaluation output. Only a subset is strictly required so the summary
#remains compatible with both the single-run and batch evaluation tools.
REQUIRED_EVALUATION_COLUMNS = {
    "patient_id",
    "correct_matches",
    "incorrect_matches",
    "tracking_accuracy",
}

SUMMARY_COLUMNS = [
    "patient_id",
    "bl_lesions",
    "fu_lesions",
    "matched_lesions",
    "matched_fu_lesions",
    "matched_links",
    "unmatched_lesions",
    "appearing_lesions",
    "disappearing_lesions",
    "merging_links",
    "merged_fu_lesions",
    "ground_truth_available",
    "ground_truth_links",
    "correct_matches",
    "incorrect_matches",
    "missed_matches",
    "tracking_accuracy",
    "precision",
    "recall",
    "f1",
    "topology_accuracy",
    "evaluation_status",
    "evaluation_error",
]


def _normalise_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _as_optional_int(value: object) -> int | pd._libs.missing.NAType:
    if value is None or pd.isna(value):
        return pd.NA
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return pd.NA


def _as_optional_float(value: object) -> float:
    if value is None or pd.isna(value):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def load_tracking_matches(path: str | Path) -> pd.DataFrame:
    #Load the matcher output while preserving patient/lesion IDs as strings
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Matching results CSV was not found: {path}")
    try:
        frame = pd.read_csv(
            path,
            dtype={
                "patient_id": str,
                "bl_lesion_id": str,
                "fu_lesion_id": str,
                "match_type": str,
            },
        )
    except Exception as exc:
        raise TrackingSummaryError(
            f"Could not read matching results CSV '{path}': {exc}"
        ) from exc

    missing = sorted(REQUIRED_MATCH_COLUMNS - set(frame.columns))
    if missing:
        raise TrackingSummaryError(
            "Matching results CSV is missing required column(s): "
            + ", ".join(missing)
        )

    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].fillna("").astype(str).str.strip()
    frame["match_type"] = (
        frame["match_type"].fillna("").astype(str).str.strip().str.upper()
    )
    if (frame["patient_id"] == "").any():
        raise TrackingSummaryError("Matching results contain a blank patient_id.")
    if (frame["match_type"] == "").any():
        raise TrackingSummaryError("Matching results contain a blank match_type.")
    return frame


def load_evaluation_summary(path: str | Path) -> pd.DataFrame:
    #Load the per-patient evaluation summary
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Evaluation summary CSV was not found: {path}")
    try:
        frame = pd.read_csv(path, dtype={"patient_id": str})
    except Exception as exc:
        raise TrackingSummaryError(
            f"Could not read evaluation summary CSV '{path}': {exc}"
        ) from exc

    missing = sorted(REQUIRED_EVALUATION_COLUMNS - set(frame.columns))
    if missing:
        raise TrackingSummaryError(
            "Evaluation summary CSV is missing required column(s): "
            + ", ".join(missing)
        )

    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].fillna("").astype(str).str.strip()
    if (frame["patient_id"] == "").any():
        raise TrackingSummaryError("Evaluation summary contains a blank patient_id.")
    if frame["patient_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["patient_id"].duplicated(), "patient_id"].unique())
        raise TrackingSummaryError(
            "Evaluation summary contains duplicate patient rows: "
            + ", ".join(duplicates)
        )
    return frame


def _patient_structural_summary(patient_matches: pd.DataFrame, patient_id: str) -> dict[str, object]:
    if patient_matches.empty:
        raise TrackingSummaryError(
            f"No matching results were found for patient '{patient_id}'."
        )

    bl_ids = patient_matches["bl_lesion_id"].map(_normalise_id)
    fu_ids = patient_matches["fu_lesion_id"].map(_normalise_id)
    match_types = patient_matches["match_type"].astype(str).str.strip().str.upper()

    has_bl = bl_ids.notna()
    has_fu = fu_ids.notna()
    has_both = has_bl & has_fu

    disappearing_mask = match_types.eq("DISAPPEARING") & has_bl
    appearing_mask = match_types.eq("NEW") & has_fu
    merging_mask = match_types.eq("MERGING") & has_both

    bl_unique = set(bl_ids[has_bl].astype(str))
    fu_unique = set(fu_ids[has_fu].astype(str))
    matched_bl_unique = set(bl_ids[has_both].astype(str))
    matched_fu_unique = set(fu_ids[has_both].astype(str))
    disappearing_unique = set(bl_ids[disappearing_mask].astype(str))
    appearing_unique = set(fu_ids[appearing_mask].astype(str))
    merged_fu_unique = set(fu_ids[merging_mask].astype(str))

    #Matcher design gives each BL lesion one final outcome. Counting unique BL
    #lesions with a FU endpoint therefore gives an intuitive "matched lesions"
    #count without double-counting a FU lesion in a many-to-one merge.
    return {
        "patient_id": str(patient_id),
        "bl_lesions": len(bl_unique),
        "fu_lesions": len(fu_unique),
        "matched_lesions": len(matched_bl_unique),
        "matched_fu_lesions": len(matched_fu_unique),
        "matched_links": int(has_both.sum()),
        "unmatched_lesions": len(disappearing_unique) + len(appearing_unique),
        "appearing_lesions": len(appearing_unique),
        "disappearing_lesions": len(disappearing_unique),
        "merging_links": int(merging_mask.sum()),
        "merged_fu_lesions": len(merged_fu_unique),
    }


def _blank_evaluation_fields() -> dict[str, object]:
    return {
        "ground_truth_available": False,
        "ground_truth_links": pd.NA,
        "correct_matches": pd.NA,
        "incorrect_matches": pd.NA,
        "missed_matches": pd.NA,
        "tracking_accuracy": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "f1": math.nan,
        "topology_accuracy": math.nan,
        "evaluation_status": "NOT_AVAILABLE",
        "evaluation_error": "",
    }


def _evaluation_fields(row: pd.Series | None) -> dict[str, object]:
    if row is None:
        return _blank_evaluation_fields()

    status = str(row.get("evaluation_status", "OK")).strip() or "OK"
    error = "" if pd.isna(row.get("error", "")) else str(row.get("error", "")).strip()
    tracking_accuracy = _as_optional_float(row.get("tracking_accuracy"))
    available = not status.upper().startswith("FAILED") and math.isfinite(tracking_accuracy)

    return {
        "ground_truth_available": bool(available),
        "ground_truth_links": _as_optional_int(row.get("ground_truth_links")),
        "correct_matches": _as_optional_int(row.get("correct_matches")),
        "incorrect_matches": _as_optional_int(row.get("incorrect_matches")),
        "missed_matches": _as_optional_int(row.get("missed_matches")),
        "tracking_accuracy": tracking_accuracy,
        "precision": _as_optional_float(row.get("precision")),
        "recall": _as_optional_float(row.get("recall")),
        "f1": _as_optional_float(row.get("f1")),
        "topology_accuracy": _as_optional_float(row.get("topology_accuracy")),
        "evaluation_status": status,
        "evaluation_error": error,
    }


def _evaluation_row_for_patient(
    evaluation_summary: pd.DataFrame | None,
    patient_id: str,
) -> pd.Series | None:
    if evaluation_summary is None or evaluation_summary.empty:
        return None
    rows = evaluation_summary.loc[
        evaluation_summary["patient_id"].astype(str) == str(patient_id)
    ]
    if rows.empty:
        return None
    if len(rows) > 1:
        raise TrackingSummaryError(
            f"Evaluation summary contains multiple rows for patient '{patient_id}'."
        )
    return rows.iloc[0]


def summarise_patient_tracking(
    matches: pd.DataFrame,
    patient_id: str,
    *,
    evaluation_summary: pd.DataFrame | None = None,
) -> dict[str, object]:
    #Return one summary row for a selected patient.
    missing = sorted(REQUIRED_MATCH_COLUMNS - set(matches.columns))
    if missing:
        raise TrackingSummaryError(
            "Matching results are missing required column(s): " + ", ".join(missing)
        )

    selected_id = str(patient_id).strip()
    if not selected_id:
        raise TrackingSummaryError("patient_id must not be blank.")

    patient_matches = matches.loc[
        matches["patient_id"].astype(str).str.strip() == selected_id
    ].copy()
    structural = _patient_structural_summary(patient_matches, selected_id)
    evaluation = _evaluation_fields(
        _evaluation_row_for_patient(evaluation_summary, selected_id)
    )
    return {**structural, **evaluation}


def _overall_evaluation_fields(
    evaluation_summary: pd.DataFrame | None,
    patient_ids: Iterable[str],
) -> dict[str, object]:
    if evaluation_summary is None or evaluation_summary.empty:
        return _blank_evaluation_fields()

    overall_rows = evaluation_summary.loc[
        evaluation_summary["patient_id"].astype(str).str.upper() == "ALL"
    ]
    if len(overall_rows) == 1:
        return _evaluation_fields(overall_rows.iloc[0])
    if len(overall_rows) > 1:
        raise TrackingSummaryError("Evaluation summary contains multiple ALL rows.")

    patient_id_set = {str(value) for value in patient_ids}
    selected = evaluation_summary.loc[
        evaluation_summary["patient_id"].astype(str).isin(patient_id_set)
    ].copy()
    if selected.empty:
        return _blank_evaluation_fields()

    status_series = selected.get(
        "evaluation_status",
        pd.Series("OK", index=selected.index, dtype=object),
    ).fillna("OK").astype(str)
    successful = selected.loc[~status_series.str.upper().str.startswith("FAILED")].copy()
    if successful.empty:
        fields = _blank_evaluation_fields()
        fields["evaluation_status"] = f"OK_PATIENTS=0/{len(patient_id_set)}"
        return fields

    def numeric_sum(column: str) -> int | pd._libs.missing.NAType:
        if column not in successful.columns:
            return pd.NA
        values = pd.to_numeric(successful[column], errors="coerce")
        return int(values.sum()) if values.notna().any() else pd.NA

    gt = numeric_sum("ground_truth_links")
    correct = numeric_sum("correct_matches")
    incorrect = numeric_sum("incorrect_matches")
    missed = numeric_sum("missed_matches")
    predicted = numeric_sum("predicted_links")

    gt_num = None if pd.isna(gt) else int(gt)
    correct_num = None if pd.isna(correct) else int(correct)
    incorrect_num = None if pd.isna(incorrect) else int(incorrect)
    missed_num = None if pd.isna(missed) else int(missed)
    predicted_num = None if pd.isna(predicted) else int(predicted)

    tracking_accuracy = (
        float(correct_num / gt_num)
        if gt_num and correct_num is not None
        else math.nan
    )
    precision = (
        float(correct_num / predicted_num)
        if predicted_num and correct_num is not None
        else math.nan
    )
    recall = tracking_accuracy
    if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = math.nan

    ok_count = len(successful)
    return {
        "ground_truth_available": math.isfinite(tracking_accuracy),
        "ground_truth_links": gt,
        "correct_matches": correct,
        "incorrect_matches": incorrect,
        "missed_matches": missed,
        "tracking_accuracy": tracking_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "topology_accuracy": math.nan,
        "evaluation_status": f"OK_PATIENTS={ok_count}/{len(patient_id_set)}",
        "evaluation_error": "",
    }


def build_tracking_summary(
    matches: pd.DataFrame,
    *,
    evaluation_summary: pd.DataFrame | None = None,
    include_overall: bool = True,
) -> pd.DataFrame:
    """Build per-patient structural + optional ground-truth tracking summaries."""
    missing = sorted(REQUIRED_MATCH_COLUMNS - set(matches.columns))
    if missing:
        raise TrackingSummaryError(
            "Matching results are missing required column(s): " + ", ".join(missing)
        )
    if matches.empty:
        raise TrackingSummaryError("Matching results are empty.")

    patient_ids = sorted(
        {
            str(value).strip()
            for value in matches["patient_id"].dropna().astype(str)
            if str(value).strip() and str(value).strip().upper() != "ALL"
        }
    )
    if not patient_ids:
        raise TrackingSummaryError("Matching results contain no patient IDs.")

    rows = [
        summarise_patient_tracking(
            matches,
            patient_id,
            evaluation_summary=evaluation_summary,
        )
        for patient_id in patient_ids
    ]

    if include_overall:
        structural_fields = [
            "bl_lesions",
            "fu_lesions",
            "matched_lesions",
            "matched_fu_lesions",
            "matched_links",
            "unmatched_lesions",
            "appearing_lesions",
            "disappearing_lesions",
            "merging_links",
            "merged_fu_lesions",
        ]
        overall: dict[str, object] = {"patient_id": "ALL"}
        for field in structural_fields:
            overall[field] = int(sum(int(row[field]) for row in rows))
        overall.update(_overall_evaluation_fields(evaluation_summary, patient_ids))
        rows.append(overall)

    frame = pd.DataFrame(rows)
    for column in SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[SUMMARY_COLUMNS]


def build_tracking_summary_from_files(
    matches_path: str | Path,
    *,
    evaluation_summary_path: str | Path | None = None,
    include_overall: bool = True,
) -> pd.DataFrame:
    matches = load_tracking_matches(matches_path)
    evaluation_summary = None
    if evaluation_summary_path is not None and str(evaluation_summary_path).strip():
        evaluation_summary = load_evaluation_summary(evaluation_summary_path)
    return build_tracking_summary(
        matches,
        evaluation_summary=evaluation_summary,
        include_overall=include_overall,
    )
