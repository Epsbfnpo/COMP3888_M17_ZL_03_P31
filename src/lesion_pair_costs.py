from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import math
import re

import numpy as np
import pandas as pd


class LesionPairCostError(ValueError):
    """Raised when aligned lesion features cannot be converted into pair costs."""


@dataclass(frozen=True)
class PairCostConfig:
    #Configuration for BL-to-FU lesion candidate cost generation.
    #The cost generator does not perform lesion assignment. 
    #it only scores candidate edges that a later matcher (min-cost-flow) may use.
    

    distance_weight: float = 1.0
    size_weight: float = 0.5
    pet_weight: float = 0.25
    distance_scale_mm: float = 50.0
    pet_feature: str = "suvmax"
    max_distance_mm: float | None = None


@dataclass(frozen=True)
class PatientCostMatrix:
    patient_id: str
    bl_lesion_ids: tuple[str, ...]
    fu_lesion_ids: tuple[str, ...]
    values: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape


@dataclass(frozen=True)
class PairCostResult:
    pair_costs: pd.DataFrame
    matrices: Mapping[str, PatientCostMatrix]


PAIR_COST_COLUMNS = [
    "patient_id",
    "bl_lesion_id",
    "fu_lesion_id",
    "distance_mm",
    "distance_cost",
    "bl_volume_ml",
    "fu_volume_ml",
    "size_difference_fraction",
    "size_cost",
    "pet_feature",
    "bl_pet_value",
    "fu_pet_value",
    "pet_difference_fraction",
    "pet_cost",
    "pet_used",
    "active_weight_sum",
    "total_cost",
    "candidate_rank_for_bl",
]

_ALLOWED_PET_FEATURES = {"none", "suvmean", "suvmax", "pet_mean", "pet_max"}
_REQUIRED_BASE_COLUMNS = {
    "patient_id",
    "timepoint",
    "centroid_x_mm",
    "centroid_y_mm",
    "centroid_z_mm",
}


def _validate_config(config: PairCostConfig) -> None:
    for name, value in (
        ("distance_weight", config.distance_weight),
        ("size_weight", config.size_weight),
        ("pet_weight", config.pet_weight),
    ):
        if not np.isfinite(value) or value < 0:
            raise LesionPairCostError(f"{name} must be a finite value >= 0.")

    if config.distance_weight + config.size_weight + config.pet_weight <= 0:
        raise LesionPairCostError("At least one cost weight must be > 0.")

    if not np.isfinite(config.distance_scale_mm) or config.distance_scale_mm <= 0:
        raise LesionPairCostError("distance_scale_mm must be a finite value > 0.")

    pet_feature = str(config.pet_feature).strip().lower()
    if pet_feature not in _ALLOWED_PET_FEATURES:
        raise LesionPairCostError(
            "pet_feature must be one of: " + ", ".join(sorted(_ALLOWED_PET_FEATURES))
        )

    if config.max_distance_mm is not None:
        if not np.isfinite(config.max_distance_mm) or config.max_distance_mm < 0:
            raise LesionPairCostError("max_distance_mm must be None or a finite value >= 0.")


def _normalise_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    table = features.copy()

    if "lesion_id" not in table.columns:
        if "temporary_lesion_id" in table.columns:
            table = table.rename(columns={"temporary_lesion_id": "lesion_id"})
        else:
            raise LesionPairCostError(
                "Aligned lesion features require a 'lesion_id' column "
                "(or the legacy 'temporary_lesion_id' column)."
            )

    missing = sorted(_REQUIRED_BASE_COLUMNS - set(table.columns))
    if missing:
        raise LesionPairCostError(
            "Aligned lesion feature table is missing required column(s): "
            + ", ".join(missing)
        )

    if table.empty:
        #preserve predictable dtypes/columns for downstream code
        return table

    table["patient_id"] = table["patient_id"].astype(str).str.strip()
    table["lesion_id"] = table["lesion_id"].astype(str).str.strip()
    table["timepoint"] = table["timepoint"].astype(str).str.strip().str.upper()

    if (table["patient_id"] == "").any():
        raise LesionPairCostError("patient_id must not be blank.")
    if (table["lesion_id"] == "").any():
        raise LesionPairCostError("lesion_id must not be blank.")

    unknown_tp = sorted(set(table["timepoint"]) - {"BL", "FU"})
    if unknown_tp:
        raise LesionPairCostError(
            "timepoint must contain only BL/FU; found: " + ", ".join(unknown_tp)
        )

    duplicate = table.duplicated(["patient_id", "timepoint", "lesion_id"], keep=False)
    if duplicate.any():
        example = table.loc[duplicate, ["patient_id", "timepoint", "lesion_id"]].iloc[0]
        raise LesionPairCostError(
            "Duplicate lesion identifier within patient/timepoint: "
            f"{example['patient_id']} {example['timepoint']} {example['lesion_id']}"
        )

    coordinate_columns = ["centroid_x_mm", "centroid_y_mm", "centroid_z_mm"]
    for column in coordinate_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    coords = table[coordinate_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(coords)):
        raise LesionPairCostError(
            "Aligned lesion centroids must contain finite centroid_x/y/z_mm values."
        )

    #extract_aligned_lesion_features() records FU_RAS_mm for both timepoints.
    #Refuse raw/native coordinates when the metadata is available, 
    #because BL - FU distances would otherwise be physically misleading
    if "coordinate_space" in table.columns:
        spaces = table["coordinate_space"].fillna("").astype(str).str.strip()
        invalid = spaces != "FU_RAS_mm"

        if invalid.any():
            bad = table.loc[invalid, ["patient_id", "timepoint", "lesion_id"]].iloc[0]
            value = spaces.loc[invalid].iloc[0]
            raise LesionPairCostError(
                "Lesion coordinates must already be aligned in FU_RAS_mm. "
                f"Found coordinate_space={value!r} for "
                f"{bad['patient_id']} {bad['timepoint']} {bad['lesion_id']}."
            )

    return table


def load_aligned_feature_csvs(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and concatenate one or more aligned lesion-feature CSV files."""
    frames: list[pd.DataFrame] = []

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Aligned lesion feature CSV does not exist: {path}")
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise LesionPairCostError(f"Could not read feature CSV '{path}': {exc}") from exc
        frames.append(frame)

    if not frames:
        raise LesionPairCostError("At least one aligned lesion feature CSV is required.")

    return _normalise_feature_table(pd.concat(frames, ignore_index=True, sort=False))


def _fractional_difference(a: float, b: float) -> float:
    #Using a larger amplitude as the denominator for symmetric relative difference.
    #Equal to positive value ->0. For non negative values, the result is [0,1].

    denominator = max(abs(a), abs(b))

    if denominator == 0:
        return 0.0
    return abs(a - b) / denominator


def _coerce_optional_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pet_values_are_comparable(
    bl: pd.Series,
    fu: pd.Series,
    pet_feature: str,
) -> tuple[float | None, float | None, bool]:
    if pet_feature == "none":
        return None, None, False
    if pet_feature not in bl.index or pet_feature not in fu.index:
        return None, None, False

    bl_value = _coerce_optional_number(bl.get(pet_feature))
    fu_value = _coerce_optional_number(fu.get(pet_feature))
    if bl_value is None or fu_value is None:
        return bl_value, fu_value, False

    if pet_feature in {"suvmean", "suvmax"}:
        #Only when PET explicitly states that it is already an SUV unit, will these columns be filled with lesion_feature
        return bl_value, fu_value, True

    
    #The original PET strength characteristics are comparable only when the same unit exists at two time points.
    if "pet_units" not in bl.index or "pet_units" not in fu.index:
        return bl_value, fu_value, False
    
    bl_units = str(bl.get("pet_units", "")).strip().lower()
    fu_units = str(fu.get("pet_units", "")).strip().lower()
    comparable = bool(bl_units and fu_units and bl_units != "nan" and bl_units == fu_units)

    return bl_value, fu_value, comparable


def _validate_size_features(table: pd.DataFrame, config: PairCostConfig) -> None:
    if config.size_weight <= 0:
        return
    if "volume_ml" not in table.columns:
        raise LesionPairCostError(
            "size_weight > 0 requires the 'volume_ml' aligned lesion feature column."
        )
    
    values = pd.to_numeric(table["volume_ml"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise LesionPairCostError(
            "volume_ml must contain finite values > 0 when size_weight > 0."
        )


def generate_lesion_pair_costs(
    features: pd.DataFrame,
    config: PairCostConfig = PairCostConfig(),
) -> PairCostResult:
    #Independently generate BL to FU candidate costs in each patient.
    #Unless max_destance mm is provided, each BL x FU pair is a candidate. 
    #This function intentionally does not select allocation, so several BL lesions may still be candidates for the same FU lesion (as required for later merging treatment).
    _validate_config(config)
    table = _normalise_feature_table(features)
    _validate_size_features(table, config)

    pet_feature = str(config.pet_feature).strip().lower()
    rows: list[dict[str, object]] = []
    matrices: dict[str, PatientCostMatrix] = {}

    #Sorting makes CSV/matrix output deterministic across runs
    patient_ids = sorted(table["patient_id"].unique()) if not table.empty else []
    for patient_id in patient_ids:
        patient = table[table["patient_id"] == patient_id]
        bl = patient[patient["timepoint"] == "BL"].sort_values("lesion_id")
        fu = patient[patient["timepoint"] == "FU"].sort_values("lesion_id")

        bl_ids = tuple(bl["lesion_id"].astype(str))
        fu_ids = tuple(fu["lesion_id"].astype(str))
        matrix = np.full((len(bl_ids), len(fu_ids)), np.inf, dtype=float)

        for bl_index, (_, bl_row) in enumerate(bl.iterrows()):
            bl_xyz = np.array(
                [bl_row["centroid_x_mm"], bl_row["centroid_y_mm"], bl_row["centroid_z_mm"]],
                dtype=float,
            )
            for fu_index, (_, fu_row) in enumerate(fu.iterrows()):
                fu_xyz = np.array(
                    [fu_row["centroid_x_mm"], fu_row["centroid_y_mm"], fu_row["centroid_z_mm"]],
                    dtype=float,
                )
                distance_mm = float(np.linalg.norm(bl_xyz - fu_xyz))

                if (
                    config.max_distance_mm is not None
                    and distance_mm > config.max_distance_mm
                ):
                    continue

                distance_cost = distance_mm / config.distance_scale_mm
                numerator = config.distance_weight * distance_cost
                active_weight_sum = config.distance_weight

                bl_volume = _coerce_optional_number(bl_row.get("volume_ml"))
                fu_volume = _coerce_optional_number(fu_row.get("volume_ml"))
                size_difference: float | None = None
                size_cost: float | None = None
                if config.size_weight > 0:
                    # _validate_size_features already guarantees these values.
                    assert bl_volume is not None and fu_volume is not None
                    size_difference = _fractional_difference(bl_volume, fu_volume)
                    size_cost = size_difference
                    numerator += config.size_weight * size_cost
                    active_weight_sum += config.size_weight

                bl_pet, fu_pet, pet_used = _pet_values_are_comparable(
                    bl_row, fu_row, pet_feature
                )
                pet_difference: float | None = None
                pet_cost: float | None = None
                if config.pet_weight > 0 and pet_used:
                    assert bl_pet is not None and fu_pet is not None
                    pet_difference = _fractional_difference(bl_pet, fu_pet)
                    pet_cost = pet_difference
                    numerator += config.pet_weight * pet_cost
                    active_weight_sum += config.pet_weight

                if active_weight_sum <= 0:
                    #This can only occur when distance/size weights are zero and optional PET is unavailable for this pair.
                    raise LesionPairCostError(
                        "No active cost component is available for candidate "
                        f"{bl_row['lesion_id']} -> {fu_row['lesion_id']}."
                    )

                total_cost = float(numerator / active_weight_sum)
                matrix[bl_index, fu_index] = total_cost
                rows.append(
                    {
                        "patient_id": patient_id,
                        "bl_lesion_id": str(bl_row["lesion_id"]),
                        "fu_lesion_id": str(fu_row["lesion_id"]),
                        "distance_mm": distance_mm,
                        "distance_cost": float(distance_cost),
                        "bl_volume_ml": bl_volume,
                        "fu_volume_ml": fu_volume,
                        "size_difference_fraction": size_difference,
                        "size_cost": size_cost,
                        "pet_feature": pet_feature if pet_feature != "none" else None,
                        "bl_pet_value": bl_pet,
                        "fu_pet_value": fu_pet,
                        "pet_difference_fraction": pet_difference,
                        "pet_cost": pet_cost,
                        "pet_used": bool(config.pet_weight > 0 and pet_used),
                        "active_weight_sum": float(active_weight_sum),
                        "total_cost": total_cost,
                    }
                )

        matrices[patient_id] = PatientCostMatrix(
            patient_id=patient_id,
            bl_lesion_ids=bl_ids,
            fu_lesion_ids=fu_ids,
            values=matrix,
        )

    pair_costs = pd.DataFrame(rows)
    if pair_costs.empty:
        pair_costs = pd.DataFrame(columns=[c for c in PAIR_COST_COLUMNS if c != "candidate_rank_for_bl"])
    else:
        pair_costs = pair_costs.sort_values(
            ["patient_id", "bl_lesion_id", "total_cost", "fu_lesion_id"],
            kind="stable",
        ).reset_index(drop=True)
        pair_costs["candidate_rank_for_bl"] = (
            pair_costs.groupby(["patient_id", "bl_lesion_id"]).cumcount() + 1
        )

    for column in PAIR_COST_COLUMNS:
        if column not in pair_costs.columns:
            pair_costs[column] = pd.Series(dtype="object")
    pair_costs = pair_costs[PAIR_COST_COLUMNS]

    return PairCostResult(pair_costs=pair_costs, matrices=matrices)


def export_pair_costs(pair_costs: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pair_costs.to_csv(out, index=False)
    return out


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "patient"


def export_cost_matrices(
    matrices: Mapping[str, PatientCostMatrix],
    output_dir: str | Path,
) -> list[Path]:
    #Write a labeled CSV cost matrix for each patient.
    #Rows are BL lesions, columns are FU lesions.  Non-candidate entries are written as `inf` when distance gating is enabled
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for patient_id in sorted(matrices):
        matrix = matrices[patient_id]
        frame = pd.DataFrame(
            matrix.values,
            index=pd.Index(matrix.bl_lesion_ids, name="bl_lesion_id"),
            columns=list(matrix.fu_lesion_ids),
        )
        path = out_dir / f"{_safe_filename(patient_id)}_cost_matrix.csv"
        frame.to_csv(path)
        written.append(path)

    return written
