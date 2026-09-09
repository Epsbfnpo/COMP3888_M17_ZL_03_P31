from __future__ import annotations

r"""
Batch tracking evaluation for a fixed Cohort A patient subset.

This script starts from already-extracted aligned lesion features. It then:

1. reads patient IDs + expert reference CSV paths from cohort_a_subset_pairs.csv;
2. keeps only the selected patients' aligned lesion features;
3. routes BL+FU patients through the unchanged external pair-cost generator
   and NetworkX min-cost-flow matcher;
4. directly labels BL-only patients as DISAPPEARING and FU-only patients as NEW;
5. combines solver and deterministic outcomes into one lesion_matches.csv;
6. splits the combined output into per-patient CSVs;
7. filters each expert CSV to the manifest-selected img_id_bl/img_id_fu pair;
8. maps the selected Cohort A expert correspondences to temporary lesion IDs;
9. evaluates correct / incorrect / missed matches plus precision / recall / F1;
10. writes a per-patient + overall evaluation summary.

It deliberately does NOT run CT registration or aligned-feature extraction.
Those steps should be completed before this batch evaluation.

Example
-------
python tools/run_tracking_evaluation_batch.py ^
  --pairs outputs/cohort_a_15/cohort_a_subset_pairs.csv ^
  --features outputs/aligned_lesion_features.csv ^
  --data-root "C:\Users\Bill5\Desktop\Longitudinal_CT_v2" ^
  --out-dir outputs/tracking_evaluation_15 ^
  --max-bl-per-fu 3
"""

import argparse
import math
import os
import re
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import pandas as pd


TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:
    from evaluate_lesion_tracking_accuracy import (
        TrackingEvaluationError,
        evaluate_tracking,
        filter_ground_truth_to_scan_pair,
        load_aligned_features,
        load_ground_truths,
        load_predictions,
        map_ground_truth_to_temporary_ids,
    )
except ImportError as exc:
    raise SystemExit(
        "Could not import tools/evaluate_lesion_tracking_accuracy.py. "
        "Place that evaluator in the tools/ directory before running this batch tool."
    ) from exc


class BatchEvaluationError(RuntimeError):
    """Raised when the batch pipeline cannot proceed safely."""


def _read_csv(path: str | Path, description: str) -> pd.DataFrame:
    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"{description} CSV does not exist: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        raise BatchEvaluationError(
            f"Could not read {description} CSV '{csv_path}': {exc}"
        ) from exc


def _normalise_patient_ids(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.add(text)
            result.append(text)

    return result


def _parse_requested_patient_ids(text: str) -> list[str]:
    return _normalise_patient_ids(part for part in text.split(","))


def _load_pairs(
    path: str | Path,
    *,
    requested_patient_ids: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    pairs = _read_csv(path, "pair manifest")

    required = {
        "patient_id",
        "bl_scan_id",
        "fu_scan_id",
        "reference_csv_path",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise BatchEvaluationError(
            "Pair manifest is missing required column(s): " + ", ".join(missing)
        )

    pairs = pairs.copy()
    pairs["patient_id"] = pairs["patient_id"].astype(str).str.strip()

    duplicated = pairs["patient_id"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(pairs.loc[duplicated, "patient_id"].unique())
        raise BatchEvaluationError(
            "Batch evaluation expects one BL/FU pair row per patient. "
            "Duplicate patient_id value(s): " + ", ".join(duplicate_ids)
        )

    available = _normalise_patient_ids(pairs["patient_id"])

    if requested_patient_ids:
        unknown = [pid for pid in requested_patient_ids if pid not in set(available)]
        if unknown:
            raise BatchEvaluationError(
                "Requested patient ID(s) not present in pair manifest: "
                + ", ".join(unknown)
            )
        patient_ids = requested_patient_ids
        pairs = pairs[pairs["patient_id"].isin(patient_ids)].copy()
        order = {pid: i for i, pid in enumerate(patient_ids)}
        pairs["_order"] = pairs["patient_id"].map(order)
        pairs = pairs.sort_values("_order").drop(columns="_order")
    else:
        patient_ids = available

    if not patient_ids:
        raise BatchEvaluationError("No patients were selected for evaluation.")

    return pairs.reset_index(drop=True), patient_ids



def _scan_image_id(scan_id: object, *, timepoint: str) -> int:
    """
    Extract Cohort A image index from manifest scan IDs such as:
        01161aaa0b_BL_00 -> 0
        0266e33d3f_FU_01 -> 1
    """
    if pd.isna(scan_id):
        raise BatchEvaluationError(
            f"{timepoint} scan ID is blank; cannot select ground-truth image pair."
        )

    text = str(scan_id).strip()
    pattern = rf"_{re.escape(timepoint.upper())}_(\d+)$"
    match = re.search(pattern, text, flags=re.IGNORECASE)

    if not match:
        raise BatchEvaluationError(
            f"Could not extract {timepoint.upper()} image index from "
            f"scan ID {text!r}. Expected a suffix such as "
            f"_{timepoint.upper()}_00."
        )

    return int(match.group(1))


def _resolve_reference_csv(
    raw_value: object,
    *,
    manifest_path: Path,
    data_root: Path | None,
) -> Path:
    if pd.isna(raw_value):
        raise BatchEvaluationError("reference_csv_path is blank.")

    raw_text = str(raw_value).strip()
    if not raw_text or raw_text.lower() == "nan":
        raise BatchEvaluationError("reference_csv_path is blank.")

    path = Path(raw_text).expanduser()

    if path.is_absolute():
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"Ground-truth CSV does not exist: {path}")

    candidates: list[Path] = []

    # Explicit --data-root has highest priority for manifests written with
    # path-mode=relative-to-root.
    if data_root is not None:
        candidates.append(data_root / path)

    # Supports path-mode=relative-to-manifest.
    candidates.append(manifest_path.parent / path)

    # Convenient fallbacks for repository-relative paths.
    candidates.append(PROJECT_ROOT / path)
    candidates.append(Path.cwd() / path)

    checked: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(key)
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve ground-truth CSV path "
        f"{raw_text!r}. Checked:\n  - " + "\n  - ".join(checked)
    )


def _load_and_select_features(
    paths: list[str],
    *,
    patient_ids: list[str],
) -> pd.DataFrame:
    """
    Load the selected aligned lesion features.

    A patient is allowed to have lesion rows at only one timepoint:

        BL > 0, FU = 0  -> valid DISAPPEARING-only case
        BL = 0, FU > 0  -> valid NEW-only case

    We still require at least one feature row for each selected patient. If a
    patient has no rows at all, the batch cannot distinguish "both masks are
    genuinely empty" from "feature generation failed/missing", so it is rejected
    rather than silently scored.
    """
    if not paths:
        raise BatchEvaluationError("At least one --features CSV is required.")

    frames = [_read_csv(path, "aligned lesion feature") for path in paths]
    features = pd.concat(frames, ignore_index=True, sort=False)

    required = {"patient_id", "timepoint"}
    missing_columns = sorted(required - set(features.columns))
    if missing_columns:
        raise BatchEvaluationError(
            "Aligned feature CSV is missing required column(s): "
            + ", ".join(missing_columns)
        )

    if "lesion_id" not in features.columns:
        if "temporary_lesion_id" in features.columns:
            features = features.rename(
                columns={"temporary_lesion_id": "lesion_id"}
            )
        else:
            raise BatchEvaluationError(
                "Aligned feature CSV requires lesion_id or temporary_lesion_id."
            )

    features = features.copy()
    features["patient_id"] = features["patient_id"].astype(str).str.strip()
    features["timepoint"] = (
        features["timepoint"].astype(str).str.strip().str.upper()
    )
    features["lesion_id"] = features["lesion_id"].astype(str).str.strip()

    unknown_timepoints = sorted(
        set(features["timepoint"]) - {"BL", "FU"}
    )
    if unknown_timepoints:
        raise BatchEvaluationError(
            "Aligned feature timepoint must contain only BL/FU; found: "
            + ", ".join(unknown_timepoints)
        )

    selected = features[features["patient_id"].isin(patient_ids)].copy()

    found_ids = set(_normalise_patient_ids(selected["patient_id"]))
    missing_ids = [pid for pid in patient_ids if pid not in found_ids]
    if missing_ids:
        raise BatchEvaluationError(
            "No aligned lesion feature rows found for selected patient(s): "
            + ", ".join(missing_ids)
            + ". A BL-only or FU-only patient is valid, but a patient with zero "
              "rows at both timepoints must be handled explicitly rather than "
              "silently treated as a successful extraction."
        )

    # Do NOT require both BL and FU here. Single-sided patients are valid
    # longitudinal cases. They are routed around the external pair-cost
    # generator later because there is no BL->FU matching decision to optimise.
    return selected


MATCH_BASE_COLUMNS = [
    "patient_id",
    "bl_lesion_id",
    "fu_lesion_id",
    "match_type",
    "match_cost",
]


def _route_patients_for_matching(
    features: pd.DataFrame,
    *,
    patient_ids: list[str],
    disappearing_penalty: float,
    new_lesion_penalty: float,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """
    Route BL+FU patients through the existing optimisation pipeline and handle
    single-sided patients deterministically.

    BL > 0, FU > 0:
        external generate_lesion_pair_costs.py -> NetworkX min-cost-flow

    BL > 0, FU = 0:
        every BL lesion is DISAPPEARING

    BL = 0, FU > 0:
        every FU lesion is NEW

    BL = 0, FU = 0:
        rejected because no feature row exists for the patient and the batch
        cannot safely distinguish a true empty/empty case from missing output.
    """
    solver_patient_ids: list[str] = []
    deterministic_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []

    for patient_id in patient_ids:
        patient = features[features["patient_id"] == patient_id]

        bl_ids = sorted(
            patient.loc[
                patient["timepoint"] == "BL", "lesion_id"
            ].astype(str).unique()
        )
        fu_ids = sorted(
            patient.loc[
                patient["timepoint"] == "FU", "lesion_id"
            ].astype(str).unique()
        )

        bl_count = len(bl_ids)
        fu_count = len(fu_ids)

        if bl_count > 0 and fu_count > 0:
            route = "MIN_COST_FLOW"
            solver_patient_ids.append(patient_id)

        elif bl_count > 0 and fu_count == 0:
            route = "DETERMINISTIC_DISAPPEARING"
            for bl_id in bl_ids:
                deterministic_rows.append(
                    {
                        "patient_id": patient_id,
                        "bl_lesion_id": bl_id,
                        "fu_lesion_id": None,
                        "match_type": "DISAPPEARING",
                        "match_cost": float(disappearing_penalty),
                    }
                )

        elif bl_count == 0 and fu_count > 0:
            route = "DETERMINISTIC_NEW"
            for fu_id in fu_ids:
                deterministic_rows.append(
                    {
                        "patient_id": patient_id,
                        "bl_lesion_id": None,
                        "fu_lesion_id": fu_id,
                        "match_type": "NEW",
                        "match_cost": float(new_lesion_penalty),
                    }
                )

        else:
            raise BatchEvaluationError(
                f"Patient {patient_id} has no BL or FU lesion feature rows. "
                "Handle this patient explicitly before batch evaluation."
            )

        routing_rows.append(
            {
                "patient_id": patient_id,
                "bl_lesions": bl_count,
                "fu_lesions": fu_count,
                "routing_mode": route,
            }
        )

    deterministic = pd.DataFrame(
        deterministic_rows,
        columns=MATCH_BASE_COLUMNS,
    )
    routing = pd.DataFrame(routing_rows)

    return solver_patient_ids, deterministic, routing


def _combine_predictions(
    *,
    solver_match_path: Path | None,
    deterministic_predictions: pd.DataFrame,
    patient_ids: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Combine solver and deterministic outcomes into one lesion_matches.csv."""
    frames: list[pd.DataFrame] = []

    if solver_match_path is not None and solver_match_path.exists():
        frames.append(_read_csv(solver_match_path, "solver lesion match"))

    if not deterministic_predictions.empty:
        frames.append(deterministic_predictions.copy())

    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame(columns=MATCH_BASE_COLUMNS)

    required = {"patient_id", "bl_lesion_id", "fu_lesion_id", "match_type"}
    missing = sorted(required - set(combined.columns))
    if missing:
        raise BatchEvaluationError(
            "Combined prediction table is missing required column(s): "
            + ", ".join(missing)
        )

    if "match_cost" not in combined.columns:
        combined["match_cost"] = 0.0

    combined["patient_id"] = combined["patient_id"].astype(str).str.strip()

    unexpected = sorted(set(combined["patient_id"]) - set(patient_ids))
    if unexpected:
        raise BatchEvaluationError(
            "Combined matcher output contains unexpected patient(s): "
            + ", ".join(unexpected)
        )

    patient_order = {pid: i for i, pid in enumerate(patient_ids)}
    type_order = {
        "MATCHED": 0,
        "MERGING": 1,
        "DISAPPEARING": 2,
        "NEW": 3,
    }

    combined["_patient_order"] = combined["patient_id"].map(patient_order)
    combined["_type_order"] = (
        combined["match_type"]
        .astype(str)
        .str.upper()
        .map(type_order)
        .fillna(99)
    )

    combined = (
        combined.sort_values(
            ["_patient_order", "_type_order", "bl_lesion_id", "fu_lesion_id"],
            kind="mergesort",
            na_position="last",
        )
        .drop(columns=["_patient_order", "_type_order"])
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined


def _run_command(command: list[str], *, label: str) -> None:
    print()
    print("=" * 78)
    print(label)
    print("=" * 78)
    print(subprocess.list2cmdline(command))
    print()

    try:
        subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise BatchEvaluationError(
            f"{label} failed with exit code {exc.returncode}."
        ) from exc


def _generate_cost_matrices(
    *,
    selected_feature_path: Path,
    out_pairs: Path,
    matrix_dir: Path,
    distance_weight: float,
    size_weight: float,
    pet_weight: float,
    distance_scale_mm: float,
    pet_feature: str,
    max_distance_mm: float | None,
) -> None:
    script = TOOLS_DIR / "generate_lesion_pair_costs.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Required tool does not exist: {script}"
        )

    command = [
        sys.executable,
        str(script),
        "--features",
        str(selected_feature_path),
        "--out-pairs",
        str(out_pairs),
        "--matrix-dir",
        str(matrix_dir),
        "--distance-weight",
        str(distance_weight),
        "--size-weight",
        str(size_weight),
        "--pet-weight",
        str(pet_weight),
        "--distance-scale-mm",
        str(distance_scale_mm),
        "--pet-feature",
        pet_feature,
    ]

    if max_distance_mm is not None:
        command.extend(["--max-distance-mm", str(max_distance_mm)])

    _run_command(command, label="STEP 1/3 - Generate lesion pair cost matrices")


def _expected_matrix_paths(
    matrix_dir: Path,
    patient_ids: list[str],
) -> list[Path]:
    paths: list[Path] = []
    missing: list[str] = []

    for patient_id in patient_ids:
        path = matrix_dir / f"{patient_id}_cost_matrix.csv"
        if path.exists():
            paths.append(path.resolve())
        else:
            missing.append(str(path))

    if missing:
        raise BatchEvaluationError(
            "Missing expected cost matrix file(s):\n  - "
            + "\n  - ".join(missing)
        )

    return paths


def _run_matcher(
    *,
    matrix_paths: list[Path],
    match_path: Path,
    graph_dir: Path,
    disappearing_penalty: float,
    new_lesion_penalty: float,
    merge_penalty: float,
    max_bl_per_fu: int,
    cost_scale: int,
    cost_column: str,
) -> None:
    script = TOOLS_DIR / "run_lesion_min_cost_flow.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Required tool does not exist: {script}"
        )

    command = [
        sys.executable,
        str(script),
        "--cost-matrix",
        *[str(path) for path in matrix_paths],
        "--out",
        str(match_path),
        "--graph-dir",
        str(graph_dir),
        "--disappearing-penalty",
        str(disappearing_penalty),
        "--new-lesion-penalty",
        str(new_lesion_penalty),
        "--merge-penalty",
        str(merge_penalty),
        "--max-bl-per-fu",
        str(max_bl_per_fu),
        "--cost-scale",
        str(cost_scale),
        "--cost-column",
        cost_column,
    ]

    _run_command(command, label="STEP 2/3 - Run NetworkX min-cost-flow matcher")


def _split_patient_matches(
    predictions: pd.DataFrame,
    *,
    patient_ids: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    for patient_id in patient_ids:
        patient = predictions[predictions["patient_id"] == patient_id].copy()
        path = output_dir / f"{patient_id}_lesion_matches.csv"
        patient.to_csv(path, index=False)
        result[patient_id] = path

    return result


def _evaluate_patients(
    *,
    pairs: pd.DataFrame,
    patient_ids: list[str],
    manifest_path: Path,
    data_root: Path | None,
    selected_feature_path: Path,
    combined_match_path: Path,
    matrix_dir: Path,
    out_dir: Path,
    max_map_distance_vox: float,
    include_non_challenge: bool,
    include_unclear: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print()
    print("=" * 78)
    print("STEP 3/3 - Compare predictions with Cohort A ground truth")
    print("=" * 78)
    print("Ground-truth cog_* coordinates are interpreted as voxel indices.")
    print("Ground truth is filtered to each manifest-selected BL/FU image pair.")

    features = load_aligned_features([selected_feature_path])
    predictions = load_predictions(combined_match_path)

    patient_match_dir = out_dir / "patient_matches"
    patient_match_paths = _split_patient_matches(
        predictions,
        patient_ids=patient_ids,
        output_dir=patient_match_dir,
    )

    mapped_frames: list[pd.DataFrame] = []
    successful_prediction_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []

    pair_lookup = pairs.set_index("patient_id")

    for index, patient_id in enumerate(patient_ids, start=1):
        feature_patient = features[features["patient_id"] == patient_id].copy()
        prediction_patient = predictions[
            predictions["patient_id"] == patient_id
        ].copy()

        bl_count = int(
            feature_patient.loc[
                feature_patient["timepoint"] == "BL", "lesion_id"
            ].nunique()
        )
        fu_count = int(
            feature_patient.loc[
                feature_patient["timepoint"] == "FU", "lesion_id"
            ].nunique()
        )

        if bl_count > 0 and fu_count > 0:
            routing_mode = "MIN_COST_FLOW"
            patient_matrix_path = matrix_dir / f"{patient_id}_cost_matrix.csv"
        elif bl_count > 0 and fu_count == 0:
            routing_mode = "DETERMINISTIC_DISAPPEARING"
            patient_matrix_path = None
        elif bl_count == 0 and fu_count > 0:
            routing_mode = "DETERMINISTIC_NEW"
            patient_matrix_path = None
        else:
            routing_mode = "NO_LESIONS"
            patient_matrix_path = None

        pair_row = pair_lookup.loc[patient_id]
        bl_image_id = _scan_image_id(
            pair_row["bl_scan_id"],
            timepoint="BL",
        )
        fu_image_id = _scan_image_id(
            pair_row["fu_scan_id"],
            timepoint="FU",
        )

        raw_reference = pair_row["reference_csv_path"]
        reference_path: Path | None = None
        error_text = ""
        gt_rows_loaded = 0
        gt_rows_selected = 0

        try:
            reference_path = _resolve_reference_csv(
                raw_reference,
                manifest_path=manifest_path,
                data_root=data_root,
            )

            gt = load_ground_truths(
                [reference_path],
                patient_id_override=patient_id,
                include_non_challenge=include_non_challenge,
                include_unclear=include_unclear,
            )
            gt_rows_loaded = int(len(gt))

            gt = filter_ground_truth_to_scan_pair(
                gt,
                bl_image_id=bl_image_id,
                fu_image_id=fu_image_id,
            )
            gt_rows_selected = int(len(gt))

            mapped_gt = map_ground_truth_to_temporary_ids(
                gt,
                feature_patient,
                max_distance_vox=max_map_distance_vox,
            )

            _, patient_details = evaluate_tracking(
                mapped_gt,
                prediction_patient,
            )

            mapped_frames.append(mapped_gt)
            successful_prediction_frames.append(prediction_patient)
            detail_frames.append(patient_details)

            print(
                f"[{index:02d}/{len(patient_ids):02d}] {patient_id}: "
                f"OK  BL={bl_count} FU={fu_count} "
                f"GTpair={len(mapped_gt)}/{gt_rows_loaded} "
                f"Pred={len(prediction_patient)} "
                f"(BL_{bl_image_id:02d}->FU_{fu_image_id:02d})"
            )

            status = "OK"

        except Exception as exc:
            status = "FAILED"
            error_text = f"{type(exc).__name__}: {exc}"
            print(
                f"[{index:02d}/{len(patient_ids):02d}] {patient_id}: "
                f"FAILED - {error_text}"
            )

        status_rows.append(
            {
                "patient_id": patient_id,
                "status": status,
                "bl_lesions": bl_count,
                "fu_lesions": fu_count,
                "prediction_rows": int(len(prediction_patient)),
                "routing_mode": routing_mode,
                "bl_image_id": bl_image_id,
                "fu_image_id": fu_image_id,
                "gt_rows_loaded": gt_rows_loaded,
                "gt_rows_selected": gt_rows_selected,
                "ground_truth_csv": (
                    str(reference_path)
                    if reference_path is not None
                    else str(raw_reference)
                ),
                "cost_matrix": (
                    str(patient_matrix_path)
                    if patient_matrix_path is not None
                    else ""
                ),
                "patient_matches": str(patient_match_paths[patient_id]),
                "error": error_text,
            }
        )

    status_table = pd.DataFrame(status_rows)

    if not mapped_frames:
        raise BatchEvaluationError(
            "All patient evaluations failed. See batch_status.csv for details."
        )

    all_mapped_gt = pd.concat(
        mapped_frames,
        ignore_index=True,
        sort=False,
    )

    if successful_prediction_frames:
        all_success_predictions = pd.concat(
            successful_prediction_frames,
            ignore_index=True,
            sort=False,
        )
    else:
        all_success_predictions = predictions.iloc[0:0].copy()

    summary, combined_details = evaluate_tracking(
        all_mapped_gt,
        all_success_predictions,
    )

    # Add explicit status/error information and preserve failed patients in the
    # summary instead of silently dropping them.
    summary["evaluation_status"] = "OK"
    summary["error"] = ""

    failed_status = status_table[status_table["status"] == "FAILED"]

    failure_rows: list[dict[str, object]] = []
    for row in failed_status.itertuples(index=False):
        failure_rows.append(
            {
                "patient_id": row.patient_id,
                "ground_truth_links": pd.NA,
                "predicted_links": row.prediction_rows,
                "correct_matches": pd.NA,
                "incorrect_matches": pd.NA,
                "missed_matches": pd.NA,
                "tracking_accuracy": math.nan,
                "precision": math.nan,
                "recall": math.nan,
                "f1": math.nan,
                "strict_link_accuracy": math.nan,
                "topology_correct": pd.NA,
                "topology_accuracy": math.nan,
                "topology_precision": math.nan,
                "topology_recall": math.nan,
                "topology_f1": math.nan,
                "evaluation_status": "FAILED",
                "error": row.error,
            }
        )

    # Keep selected-patient order, then ALL at the bottom.
    success_no_all = summary[summary["patient_id"] != "ALL"].copy()
    overall = summary[summary["patient_id"] == "ALL"].copy()

    if failure_rows:
        success_no_all = pd.concat(
            [success_no_all, pd.DataFrame(failure_rows)],
            ignore_index=True,
            sort=False,
        )

    patient_order = {pid: i for i, pid in enumerate(patient_ids)}
    success_no_all["_order"] = success_no_all["patient_id"].map(patient_order)
    success_no_all = (
        success_no_all.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    ok_count = int((status_table["status"] == "OK").sum())
    overall.loc[:, "evaluation_status"] = (
        f"OK_PATIENTS={ok_count}/{len(patient_ids)}"
    )

    final_summary = pd.concat(
        [success_no_all, overall],
        ignore_index=True,
        sort=False,
    )

    # evaluate_tracking() already regenerates details for all successful patients,
    # so use that canonical combined table.
    final_details = combined_details

    return final_summary, final_details, all_mapped_gt, status_table


def _validate_numeric_args(args: argparse.Namespace) -> None:
    nonnegative = {
        "--distance-weight": args.distance_weight,
        "--size-weight": args.size_weight,
        "--pet-weight": args.pet_weight,
        "--disappearing-penalty": args.disappearing_penalty,
        "--new-lesion-penalty": args.new_lesion_penalty,
        "--merge-penalty": args.merge_penalty,
    }

    for name, value in nonnegative.items():
        if not math.isfinite(value) or value < 0:
            raise BatchEvaluationError(
                f"{name} must be a finite value >= 0."
            )

    if (
        args.distance_weight
        + args.size_weight
        + args.pet_weight
        <= 0
    ):
        raise BatchEvaluationError(
            "At least one pair-cost weight must be > 0."
        )

    positive = {
        "--distance-scale-mm": args.distance_scale_mm,
        "--max-map-distance-vox": args.max_map_distance_vox,
    }

    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise BatchEvaluationError(
                f"{name} must be a finite value > 0."
            )

    if args.max_distance_mm is not None:
        if (
            not math.isfinite(args.max_distance_mm)
            or args.max_distance_mm < 0
        ):
            raise BatchEvaluationError(
                "--max-distance-mm must be a finite value >= 0."
            )

    if args.max_bl_per_fu < 1:
        raise BatchEvaluationError("--max-bl-per-fu must be >= 1.")

    if args.cost_scale < 1:
        raise BatchEvaluationError("--cost-scale must be >= 1.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-generate cost matrices, run min-cost-flow lesion tracking, "
            "and evaluate the selected Cohort A patients against expert ground truth."
        )
    )

    parser.add_argument(
        "--pairs",
        required=True,
        help="Selected cohort_a_subset_pairs.csv containing the test patients.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help=(
            "One or more aligned lesion-feature CSVs. They may contain extra "
            "patients; only patients selected from --pairs are retained."
        ),
    )
    parser.add_argument(
        "--data-root",
        default="",
        help=(
            "Cohort A root used to resolve relative reference_csv_path values. "
            "If omitted, COHORT_A_ROOT is used when available."
        ),
    )
    parser.add_argument(
        "--patient-ids",
        default="",
        help=(
            "Optional comma-separated subset of IDs from --pairs. "
            "If omitted, every patient in --pairs is evaluated."
        ),
    )
    parser.add_argument(
        "--expected-patients",
        type=int,
        default=15,
        help=(
            "Fail if the selected patient count differs from this value. "
            "Default: 15. Use 0 to disable this check."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/tracking_evaluation_15",
        help="Batch output directory.",
    )
    parser.add_argument(
        "--reuse-matrices",
        action="store_true",
        help=(
            "Skip cost generation and reuse '<patient>_cost_matrix.csv' files "
            "already present under --matrix-dir."
        ),
    )
    parser.add_argument(
        "--matrix-dir",
        default="",
        help=(
            "Cost matrix directory. Default: <out-dir>/lesion_pair_cost_matrices."
        ),
    )

    # Pair-cost settings: same defaults as generate_lesion_pair_costs.py.
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--size-weight", type=float, default=0.5)
    parser.add_argument("--pet-weight", type=float, default=0.25)
    parser.add_argument("--distance-scale-mm", type=float, default=50.0)
    parser.add_argument(
        "--pet-feature",
        choices=["none", "suvmean", "suvmax", "pet_mean", "pet_max"],
        default="suvmax",
    )
    parser.add_argument(
        "--max-distance-mm",
        type=float,
        default=None,
        help="Optional candidate distance gate used during cost generation.",
    )

    # Matcher settings: same defaults as run_lesion_min_cost_flow.py.
    parser.add_argument("--disappearing-penalty", type=float, default=1.0)
    parser.add_argument("--new-lesion-penalty", type=float, default=1.0)
    parser.add_argument("--merge-penalty", type=float, default=0.2)
    parser.add_argument(
        "--max-bl-per-fu",
        type=int,
        default=1,
        help="1 disables many-to-one merges; 2+ enables merge slots.",
    )
    parser.add_argument("--cost-scale", type=int, default=1000)
    parser.add_argument("--cost-column", default="total_cost")

    # Evaluation settings.
    parser.add_argument(
        "--max-map-distance-vox",
        type=float,
        default=30.0,
        help=(
            "Maximum Cohort A expert CoG -> temporary-lesion centroid distance "
            "in voxel-index coordinates. Default: 30 voxels."
        ),
    )
    parser.add_argument(
        "--include-non-challenge",
        action="store_true",
        help="Include use_for_challenge=False expert rows.",
    )
    parser.add_argument(
        "--include-unclear",
        action="store_true",
        help="Include linking_unclear=True expert rows.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        _validate_numeric_args(args)

        manifest_path = Path(args.pairs).expanduser().resolve()

        requested_ids = _parse_requested_patient_ids(args.patient_ids)
        pairs, patient_ids = _load_pairs(
            manifest_path,
            requested_patient_ids=requested_ids or None,
        )

        if args.expected_patients > 0 and len(patient_ids) != args.expected_patients:
            raise BatchEvaluationError(
                f"Expected {args.expected_patients} patients but selected "
                f"{len(patient_ids)}. Check --pairs/--patient-ids, or use "
                "--expected-patients 0 to disable this check."
            )

        root_text = str(args.data_root).strip()
        if not root_text:
            root_text = os.environ.get("COHORT_A_ROOT", "").strip()

        data_root = (
            Path(root_text).expanduser().resolve()
            if root_text
            else None
        )

        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        matrix_dir = (
            Path(args.matrix_dir).expanduser().resolve()
            if str(args.matrix_dir).strip()
            else out_dir / "lesion_pair_cost_matrices"
        )
        matrix_dir.mkdir(parents=True, exist_ok=True)

        selected_features = _load_and_select_features(
            args.features,
            patient_ids=patient_ids,
        )

        selected_feature_path = out_dir / "selected_aligned_lesion_features.csv"
        selected_features.to_csv(selected_feature_path, index=False)

        print("Tracking evaluation batch")
        print("-------------------------")
        print(f"Patients          : {len(patient_ids)}")
        print(f"Pairs manifest    : {manifest_path}")
        print(f"Selected features : {selected_feature_path}")
        print(f"Output directory  : {out_dir}")
        print(f"Merge slots/FU    : {args.max_bl_per_fu}")
        print(f"Merge penalty     : {args.merge_penalty}")
        print()
        print("Patient IDs:")
        for patient_id in patient_ids:
            print(f"  - {patient_id}")

        (
            solver_patient_ids,
            deterministic_predictions,
            routing_table,
        ) = _route_patients_for_matching(
            selected_features,
            patient_ids=patient_ids,
            disappearing_penalty=args.disappearing_penalty,
            new_lesion_penalty=args.new_lesion_penalty,
        )

        routing_path = out_dir / "patient_routing.csv"
        routing_table.to_csv(routing_path, index=False)

        solver_feature_path = out_dir / "solver_aligned_lesion_features.csv"
        solver_features = selected_features[
            selected_features["patient_id"].isin(solver_patient_ids)
        ].copy()
        solver_features.to_csv(solver_feature_path, index=False)

        print()
        print("Patient routing")
        print("---------------")
        for row in routing_table.itertuples(index=False):
            print(
                f"  {row.patient_id}: {row.routing_mode} "
                f"(BL={row.bl_lesions}, FU={row.fu_lesions})"
            )

        print()
        print(f"Min-cost-flow patients : {len(solver_patient_ids)}")
        print(
            f"Deterministic patients : "
            f"{len(patient_ids) - len(solver_patient_ids)}"
        )
        print(f"Deterministic rows     : {len(deterministic_predictions)}")

        pair_cost_path = out_dir / "lesion_pair_costs.csv"
        solver_match_path = out_dir / "solver_lesion_matches.csv"
        graph_dir = out_dir / "lesion_flow_graphs"

        if solver_patient_ids:
            if args.reuse_matrices:
                print()
                print(
                    "STEP 1/3 - Reusing existing lesion pair cost matrices "
                    "for BL+FU patients"
                )
            else:
                _generate_cost_matrices(
                    selected_feature_path=solver_feature_path,
                    out_pairs=pair_cost_path,
                    matrix_dir=matrix_dir,
                    distance_weight=args.distance_weight,
                    size_weight=args.size_weight,
                    pet_weight=args.pet_weight,
                    distance_scale_mm=args.distance_scale_mm,
                    pet_feature=args.pet_feature,
                    max_distance_mm=args.max_distance_mm,
                )

            matrix_paths = _expected_matrix_paths(
                matrix_dir,
                solver_patient_ids,
            )

            _run_matcher(
                matrix_paths=matrix_paths,
                match_path=solver_match_path,
                graph_dir=graph_dir,
                disappearing_penalty=args.disappearing_penalty,
                new_lesion_penalty=args.new_lesion_penalty,
                merge_penalty=args.merge_penalty,
                max_bl_per_fu=args.max_bl_per_fu,
                cost_scale=args.cost_scale,
                cost_column=args.cost_column,
            )
        else:
            print()
            print(
                "STEP 1/3 - No BL+FU patients; skipping external pair-cost generation."
            )
            print(
                "STEP 2/3 - No optimisation decision required; skipping min-cost-flow."
            )
            pd.DataFrame(columns=MATCH_BASE_COLUMNS).to_csv(
                solver_match_path,
                index=False,
            )

        combined_match_path = out_dir / "lesion_matches.csv"

        combined_predictions = _combine_predictions(
            solver_match_path=solver_match_path,
            deterministic_predictions=deterministic_predictions,
            patient_ids=patient_ids,
            output_path=combined_match_path,
        )

        print()
        print("Combined predictions")
        print("--------------------")
        counts = (
            combined_predictions["match_type"].value_counts()
            if not combined_predictions.empty
            else {}
        )
        print(f"  MATCHED       : {int(counts.get('MATCHED', 0))}")
        print(f"  MERGING       : {int(counts.get('MERGING', 0))}")
        print(f"  DISAPPEARING  : {int(counts.get('DISAPPEARING', 0))}")
        print(f"  NEW           : {int(counts.get('NEW', 0))}")
        print(f"  Total rows    : {len(combined_predictions)}")

        summary, details, mapping, status = _evaluate_patients(
            pairs=pairs,
            patient_ids=patient_ids,
            manifest_path=manifest_path,
            data_root=data_root,
            selected_feature_path=selected_feature_path,
            combined_match_path=combined_match_path,
            matrix_dir=matrix_dir,
            out_dir=out_dir,
            max_map_distance_vox=args.max_map_distance_vox,
            include_non_challenge=args.include_non_challenge,
            include_unclear=args.include_unclear,
        )

        summary_path = out_dir / "lesion_tracking_summary.csv"
        details_path = out_dir / "lesion_tracking_details.csv"
        mapping_path = out_dir / "lesion_tracking_gt_mapping.csv"
        status_path = out_dir / "batch_status.csv"

        summary.to_csv(summary_path, index=False)
        details.to_csv(details_path, index=False)
        mapping.to_csv(mapping_path, index=False)
        status.to_csv(status_path, index=False)

        overall = summary[summary["patient_id"] == "ALL"]

        print()
        print("=" * 78)
        print("BATCH COMPLETE")
        print("=" * 78)

        ok_count = int((status["status"] == "OK").sum())
        failed_count = int((status["status"] == "FAILED").sum())

        print(f"Patients OK       : {ok_count}/{len(patient_ids)}")
        print(f"Patients failed   : {failed_count}/{len(patient_ids)}")

        if not overall.empty:
            row = overall.iloc[0]
            print(f"Correct matches   : {int(row['correct_matches'])}")
            print(f"Incorrect matches : {int(row['incorrect_matches'])}")
            print(f"Missed matches    : {int(row['missed_matches'])}")
            print(f"Tracking accuracy : {row['tracking_accuracy']:.3f}")
            print(f"Precision         : {row['precision']:.3f}")
            print(f"Recall            : {row['recall']:.3f}")
            print(f"F1                : {row['f1']:.3f}")
            print(f"Topology accuracy : {row['topology_accuracy']:.3f}")
            print(f"Topology F1       : {row['topology_f1']:.3f}")

        print()
        print(f"Matches           : {combined_match_path}")
        print(f"Solver matches    : {solver_match_path}")
        print(f"Patient routing   : {routing_path}")
        print(f"Summary           : {summary_path}")
        print(f"Details           : {details_path}")
        print(f"GT mapping        : {mapping_path}")
        print(f"Batch status      : {status_path}")
        print(f"Flow graphs       : {graph_dir}")
        print(f"Cost matrices     : {matrix_dir}")

        if failed_count:
            print()
            print(
                "Some patient evaluations failed. The successful patients are "
                "still summarised; inspect batch_status.csv before reporting the "
                "overall result as a 15-patient accuracy."
            )
            raise SystemExit(2)

    except (
        BatchEvaluationError,
        TrackingEvaluationError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
