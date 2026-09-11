from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd

from .lesion_min_cost_flow import MATCH_COLUMNS, MatcherConfig, match_patient_cost_matrix


class MatchingDashboardError(ValueError):
    """Raised when dashboard matching input or output cannot be used safely."""


REQUIRED_RESULT_COLUMNS = {
    "patient_id",
    "bl_lesion_id",
    "fu_lesion_id",
    "match_type",
}
REQUIRED_FEATURE_COLUMNS = {
    "patient_id",
    "timepoint",
    "lesion_id",
    "centroid_x_vox",
    "centroid_y_vox",
    "centroid_z_vox",
}


@dataclass(frozen=True)
class PatientMatchSummary:
    patient_id: str
    baseline_lesions: int
    followup_lesions: int
    outcome_counts: dict[str, int]


@dataclass(frozen=True)
class LesionLocation:
    lesion_id: str
    timepoint: str
    centroid_voxel: tuple[float, float, float]


@dataclass(frozen=True)
class MatchImageFocus:
    match_type: str
    baseline: LesionLocation | None
    followup: LesionLocation | None


def _read_match_results(path: str | Path) -> pd.DataFrame:
    result_path = Path(path).expanduser()
    if not result_path.exists():
        raise FileNotFoundError(f"Matching results CSV was not found: {result_path}")

    try:
        matches = pd.read_csv(result_path, dtype={"patient_id": str})
    except Exception as exc:
        raise MatchingDashboardError(
            f"Could not read matching results CSV '{result_path}': {exc}"
        ) from exc

    missing = sorted(REQUIRED_RESULT_COLUMNS - set(matches.columns))
    if missing:
        raise MatchingDashboardError(
            "Matching results CSV is missing required column(s): "
            + ", ".join(missing)
        )

    matches = matches.copy()
    matches["patient_id"] = matches["patient_id"].fillna("").str.strip()
    matches["match_type"] = matches["match_type"].fillna("").str.strip().str.upper()
    if (matches["patient_id"] == "").any():
        raise MatchingDashboardError("Matching results contain a blank patient_id.")
    if (matches["match_type"] == "").any():
        raise MatchingDashboardError("Matching results contain a blank match_type.")
    return matches


def load_patient_matches(path: str | Path, patient_id: str) -> pd.DataFrame:
    """Load only the selected patient's outcomes from a combined results CSV."""
    selected_id = str(patient_id).strip()
    if not selected_id:
        raise MatchingDashboardError("patient_id must not be blank.")

    matches = _read_match_results(path)
    patient_matches = matches.loc[matches["patient_id"] == selected_id].copy()
    if patient_matches.empty:
        raise MatchingDashboardError(
            f"No matching results were found for patient '{selected_id}'."
        )
    return patient_matches.reset_index(drop=True)


def load_patient_features(path: str | Path, patient_id: str) -> pd.DataFrame:
    """Load aligned lesion locations for one patient in the FU voxel grid."""
    feature_path = Path(path).expanduser()
    selected_id = str(patient_id).strip()
    if not feature_path.exists():
        raise FileNotFoundError(f"Aligned lesion feature CSV was not found: {feature_path}")

    try:
        features = pd.read_csv(feature_path, dtype={"patient_id": str})
    except Exception as exc:
        raise MatchingDashboardError(
            f"Could not read aligned lesion feature CSV '{feature_path}': {exc}"
        ) from exc

    if "lesion_id" not in features and "temporary_lesion_id" in features:
        features = features.rename(columns={"temporary_lesion_id": "lesion_id"})
    missing = sorted(REQUIRED_FEATURE_COLUMNS - set(features.columns))
    if missing:
        raise MatchingDashboardError(
            "Aligned lesion feature CSV is missing required column(s): "
            + ", ".join(missing)
        )

    table = features.copy()
    table["patient_id"] = table["patient_id"].fillna("").str.strip()
    table["timepoint"] = table["timepoint"].fillna("").str.strip().str.upper()
    table["lesion_id"] = table["lesion_id"].fillna("").str.strip()
    table = table.loc[table["patient_id"] == selected_id].copy()
    if table.empty:
        raise MatchingDashboardError(
            f"No aligned lesion features were found for patient '{selected_id}'."
        )

    invalid_timepoints = sorted(set(table["timepoint"]) - {"BL", "FU"})
    if invalid_timepoints:
        raise MatchingDashboardError(
            "Aligned lesion features contain invalid timepoint(s): "
            + ", ".join(invalid_timepoints)
        )
    if table.duplicated(["timepoint", "lesion_id"]).any():
        raise MatchingDashboardError(
            f"Aligned lesion features contain duplicate lesion IDs for patient '{selected_id}'."
        )
    if "coordinate_space" in table:
        spaces = set(table["coordinate_space"].fillna("").astype(str).str.strip())
        if spaces != {"FU_RAS_mm"}:
            raise MatchingDashboardError(
                "Image linking requires aligned features in coordinate_space 'FU_RAS_mm'."
            )

    centroid_columns = ["centroid_x_vox", "centroid_y_vox", "centroid_z_vox"]
    table[centroid_columns] = table[centroid_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if table[centroid_columns].isna().any().any():
        raise MatchingDashboardError(
            "Aligned lesion features contain missing or non-numeric voxel centroids."
        )
    return table.reset_index(drop=True)


def resolve_match_image_focus(
    match_row: pd.Series,
    features: pd.DataFrame,
) -> MatchImageFocus:
    """Resolve a matcher row to its BL/FU aligned-image voxel locations."""
    missing = sorted(REQUIRED_FEATURE_COLUMNS - set(features.columns))
    if missing:
        raise MatchingDashboardError(
            "Aligned lesion features are missing required column(s): "
            + ", ".join(missing)
        )

    def location(timepoint: str, column: str) -> LesionLocation | None:
        raw_id = match_row.get(column)
        if pd.isna(raw_id) or not str(raw_id).strip():
            return None
        lesion_id = str(raw_id).strip()
        rows = features.loc[
            (features["timepoint"].astype(str).str.upper() == timepoint)
            & (features["lesion_id"].astype(str) == lesion_id)
        ]
        if len(rows) != 1:
            raise MatchingDashboardError(
                f"Could not uniquely locate {timepoint} lesion '{lesion_id}' "
                "in the aligned feature CSV."
            )
        row = rows.iloc[0]
        centroid = tuple(
            float(row[name])
            for name in ("centroid_x_vox", "centroid_y_vox", "centroid_z_vox")
        )
        if not all(math.isfinite(value) for value in centroid):
            raise MatchingDashboardError(
                f"Aligned feature for lesion '{lesion_id}' has an invalid centroid."
            )
        return LesionLocation(lesion_id, timepoint, centroid)

    return MatchImageFocus(
        match_type=str(match_row.get("match_type", "")).strip().upper(),
        baseline=location("BL", "bl_lesion_id"),
        followup=location("FU", "fu_lesion_id"),
    )


def save_patient_matches(
    path: str | Path,
    patient_id: str,
    patient_matches: pd.DataFrame,
) -> Path:
    """Replace one patient's rows without discarding results for other patients."""
    output_path = Path(path).expanduser()
    selected_id = str(patient_id).strip()
    if not selected_id:
        raise MatchingDashboardError("patient_id must not be blank.")

    missing = sorted(REQUIRED_RESULT_COLUMNS - set(patient_matches.columns))
    if missing:
        raise MatchingDashboardError(
            "Matcher output is missing required column(s): " + ", ".join(missing)
        )

    replacement = patient_matches.copy()
    replacement["patient_id"] = replacement["patient_id"].astype(str).str.strip()
    if not replacement.empty and set(replacement["patient_id"]) != {selected_id}:
        raise MatchingDashboardError(
            "Matcher output contains rows for a different patient."
        )

    if output_path.exists():
        existing = _read_match_results(output_path)
        existing = existing.loc[existing["patient_id"] != selected_id].copy()
    else:
        existing = pd.DataFrame(columns=MATCH_COLUMNS)

    combined = pd.concat([existing, replacement], ignore_index=True, sort=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    combined.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    return output_path


def run_patient_match(
    cost_matrix_path: str | Path,
    patient_id: str,
    output_path: str | Path,
    config: MatcherConfig = MatcherConfig(),
) -> pd.DataFrame:
    """Run min-cost-flow for one selected patient and persist its outcomes."""
    selected_id = str(patient_id).strip()
    matrix_path = Path(cost_matrix_path).expanduser()
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Cost matrix CSV was not found for patient '{selected_id}': {matrix_path}"
        )

    try:
        result = match_patient_cost_matrix(
            matrix_path,
            config=config,
            patient_id=selected_id,
        )
    except Exception as exc:
        raise MatchingDashboardError(
            f"Min-cost-flow matching failed for patient '{selected_id}': {exc}"
        ) from exc

    save_patient_matches(output_path, selected_id, result.matches)
    return result.matches.copy().reset_index(drop=True)


def summarise_patient_matches(
    matches: pd.DataFrame,
    patient_id: str,
) -> PatientMatchSummary:
    missing = sorted(REQUIRED_RESULT_COLUMNS - set(matches.columns))
    if missing:
        raise MatchingDashboardError(
            "Matching results are missing required column(s): " + ", ".join(missing)
        )

    match_types = matches["match_type"].astype(str).str.strip().str.upper()
    counts = {str(name): int(value) for name, value in match_types.value_counts().items()}
    return PatientMatchSummary(
        patient_id=str(patient_id),
        baseline_lesions=int(matches["bl_lesion_id"].dropna().astype(str).nunique()),
        followup_lesions=int(matches["fu_lesion_id"].dropna().astype(str).nunique()),
        outcome_counts=counts,
    )
