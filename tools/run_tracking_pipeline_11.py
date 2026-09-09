#!/usr/bin/env python
"""
Run the reduced Cohort A lesion tracking pipeline for the 11 currently
evaluable patients.

Pipeline
--------
1. Extract aligned lesion features for the selected patients.
2. Route patients:
      BL > 0 and FU > 0  -> pair-cost generation -> min-cost-flow
      BL > 0 and FU = 0  -> deterministic DISAPPEARING
      BL = 0 and FU > 0  -> deterministic NEW
3. Generate pair-cost matrices only for patients that have lesions at both
   timepoints. The existing tools/generate_lesion_pair_costs.py is not modified.
4. Run the existing NetworkX min-cost-flow matcher.
5. Combine solver and deterministic outcomes into one lesion_matches.csv.

The four currently excluded patients are:
    013d407166
    03c6b06952
    06138bc5af
    069059b7ef

They are excluded because their rigid registration currently shows a large
superior/inferior anatomical mismatch and is being tracked separately as a
registration bug.

This script intentionally stops at MATCH output. Ground-truth evaluation is a
separate step.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_EXCLUDED_PATIENTS = [
    "013d407166",
    "03c6b06952",
    "06138bc5af",
    "069059b7ef",
]

BASE_MATCH_COLUMNS = [
    "patient_id",
    "bl_lesion_id",
    "fu_lesion_id",
    "match_type",
    "match_cost",
]


class TrackingPipelineError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: str | Path, *, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _run(command: list[str], *, title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(" ".join(command))
    subprocess.run(command, check=True)


def _read_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise TrackingPipelineError(f"Pair manifest not found: {path}")

    pairs = pd.read_csv(path, dtype=str)
    if "patient_id" not in pairs.columns:
        raise TrackingPipelineError(
            f"Pair manifest is missing patient_id column: {path}"
        )

    pairs["patient_id"] = pairs["patient_id"].astype(str).str.strip()

    duplicated = pairs["patient_id"].duplicated(keep=False)
    if duplicated.any():
        ids = sorted(pairs.loc[duplicated, "patient_id"].unique())
        raise TrackingPipelineError(
            "Pair manifest must contain one row per patient. Duplicates: "
            + ", ".join(ids)
        )

    return pairs


def _select_patients(
    pairs: pd.DataFrame,
    *,
    excluded: list[str],
    expected_patients: int,
) -> tuple[pd.DataFrame, list[str]]:
    excluded_set = {str(x).strip() for x in excluded}

    missing_excluded = sorted(
        excluded_set - set(pairs["patient_id"].tolist())
    )
    if missing_excluded:
        raise TrackingPipelineError(
            "Excluded patient(s) were not found in the pair manifest: "
            + ", ".join(missing_excluded)
        )

    selected = pairs[~pairs["patient_id"].isin(excluded_set)].copy()
    patient_ids = selected["patient_id"].tolist()

    if expected_patients > 0 and len(patient_ids) != expected_patients:
        raise TrackingPipelineError(
            f"Expected {expected_patients} selected patients after exclusion, "
            f"but found {len(patient_ids)}."
        )

    return selected, patient_ids


def _load_features(
    feature_path: Path,
    patient_ids: list[str],
) -> pd.DataFrame:
    if not feature_path.exists():
        raise TrackingPipelineError(
            f"Aligned feature CSV was not generated: {feature_path}"
        )

    features = pd.read_csv(feature_path, dtype={"patient_id": str})
    required = {"patient_id", "timepoint", "lesion_id"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise TrackingPipelineError(
            "Aligned feature CSV is missing required column(s): "
            + ", ".join(missing)
        )

    features["patient_id"] = features["patient_id"].astype(str).str.strip()
    features["timepoint"] = features["timepoint"].astype(str).str.strip().str.upper()

    unexpected_timepoints = sorted(
        set(features["timepoint"]) - {"BL", "FU"}
    )
    if unexpected_timepoints:
        raise TrackingPipelineError(
            "Unexpected timepoint value(s): "
            + ", ".join(unexpected_timepoints)
        )

    selected = features[features["patient_id"].isin(patient_ids)].copy()

    missing_patients = sorted(
        set(patient_ids) - set(selected["patient_id"].unique())
    )
    if missing_patients:
        raise TrackingPipelineError(
            "No lesion feature rows were generated for selected patient(s): "
            + ", ".join(missing_patients)
        )

    return selected


def _route_patients(
    features: pd.DataFrame,
    patient_ids: list[str],
    *,
    disappearing_penalty: float,
    new_lesion_penalty: float,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    solver_ids: list[str] = []
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
            solver_ids.append(patient_id)

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
            raise TrackingPipelineError(
                f"{patient_id}: no BL or FU lesion rows are available."
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
        columns=BASE_MATCH_COLUMNS,
    )
    routing = pd.DataFrame(routing_rows)

    return solver_ids, deterministic, routing


def _combine_matches(
    *,
    solver_match_path: Path,
    deterministic: pd.DataFrame,
    patient_ids: list[str],
    output_path: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if solver_match_path.exists():
        solver = pd.read_csv(
            solver_match_path,
            dtype={"patient_id": str},
        )
        frames.append(solver)

    if not deterministic.empty:
        det = deterministic.copy()

        if frames:
            solver_columns = list(frames[0].columns)
            for column in solver_columns:
                if column not in det.columns:
                    det[column] = pd.NA
            for column in det.columns:
                if column not in solver_columns:
                    frames[0][column] = pd.NA
            det = det[frames[0].columns]

        frames.append(det)

    if not frames:
        combined = pd.DataFrame(columns=BASE_MATCH_COLUMNS)
    else:
        combined = pd.concat(frames, ignore_index=True, sort=False)

    required = {
        "patient_id",
        "bl_lesion_id",
        "fu_lesion_id",
        "match_type",
    }
    missing = sorted(required - set(combined.columns))
    if missing:
        raise TrackingPipelineError(
            "Combined match output is missing required column(s): "
            + ", ".join(missing)
        )

    combined["patient_id"] = combined["patient_id"].astype(str).str.strip()

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
            [
                "_patient_order",
                "_type_order",
                "bl_lesion_id",
                "fu_lesion_id",
            ],
            kind="mergesort",
            na_position="last",
        )
        .drop(columns=["_patient_order", "_type_order"])
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run extract -> pair-cost generation -> min-cost-flow matching "
            "for the reduced 11-patient Cohort A set."
        )
    )

    parser.add_argument(
        "--pairs",
        default="outputs/cohort_a_15/cohort_a_subset_pairs.csv",
        help="Original Cohort A pair manifest.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Cohort A dataset root.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/tracking_pipeline_11",
        help="Pipeline output directory.",
    )
    parser.add_argument(
        "--excluded-patients",
        nargs="+",
        default=DEFAULT_EXCLUDED_PATIENTS,
        help="Patients excluded because of the current registration bug.",
    )
    parser.add_argument(
        "--expected-patients",
        type=int,
        default=11,
        help="Expected number of selected patients after exclusion.",
    )

    parser.add_argument(
        "--connectivity",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--distance-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--size-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--pet-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--distance-scale-mm",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--pet-feature",
        default="suvmax",
    )
    parser.add_argument(
        "--max-distance-mm",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--disappearing-penalty",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--new-lesion-penalty",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--merge-penalty",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--max-bl-per-fu",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--cost-scale",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--cost-column",
        default="total_cost",
    )

    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse out-dir/aligned_lesion_features.csv.",
    )
    parser.add_argument(
        "--reuse-matrices",
        action="store_true",
        help="Reuse existing solver cost matrices.",
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = _repo_root()

    pairs_path = _resolve(args.pairs, root=root)
    data_root = _resolve(args.data_root, root=root)
    out_dir = _resolve(args.out_dir, root=root)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_path = out_dir / "aligned_lesion_features.csv"
    feature_status_path = out_dir / "aligned_lesion_features_status.csv"
    selected_pairs_path = out_dir / "selected_pairs_11.csv"
    solver_feature_path = out_dir / "solver_aligned_lesion_features.csv"
    pair_cost_path = out_dir / "lesion_pair_costs.csv"
    matrix_dir = out_dir / "lesion_pair_cost_matrices"
    solver_match_path = out_dir / "solver_lesion_matches.csv"
    final_match_path = out_dir / "lesion_matches.csv"
    routing_path = out_dir / "patient_routing.csv"
    graph_dir = out_dir / "lesion_flow_graphs"

    pairs = _read_pairs(pairs_path)
    selected_pairs, patient_ids = _select_patients(
        pairs,
        excluded=args.excluded_patients,
        expected_patients=args.expected_patients,
    )
    selected_pairs.to_csv(selected_pairs_path, index=False)

    print("Reduced lesion tracking pipeline")
    print("--------------------------------")
    print(f"Selected patients : {len(patient_ids)}")
    print(f"Excluded patients : {len(args.excluded_patients)}")
    print(f"Pair manifest     : {pairs_path}")
    print(f"Data root         : {data_root}")
    print(f"Output directory  : {out_dir}")

    print()
    print("Selected patient IDs:")
    for patient_id in patient_ids:
        print(f"  - {patient_id}")

    print()
    print("Excluded registration-bug patients:")
    for patient_id in args.excluded_patients:
        print(f"  - {patient_id}")

    if not args.reuse_features:
        feature_tool = (
            root / "tools" / "generate_aligned_lesion_features_batch.py"
        )

        command = [
            sys.executable,
            str(feature_tool),
            "--pairs",
            str(pairs_path),
            "--data-root",
            str(data_root),
            "--patient-ids",
            ",".join(patient_ids),
            "--expected-patients",
            str(len(patient_ids)),
            "--connectivity",
            str(args.connectivity),
            "--out",
            str(feature_path),
            "--status-out",
            str(feature_status_path),
        ]

        _run(
            command,
            title="STEP 1/3 - Extract aligned lesion features",
        )
    else:
        print()
        print("=" * 78)
        print("STEP 1/3 - Reuse aligned lesion features")
        print("=" * 78)
        print(feature_path)

    features = _load_features(feature_path, patient_ids)

    solver_ids, deterministic, routing = _route_patients(
        features,
        patient_ids,
        disappearing_penalty=args.disappearing_penalty,
        new_lesion_penalty=args.new_lesion_penalty,
    )
    routing.to_csv(routing_path, index=False)

    solver_features = features[
        features["patient_id"].isin(solver_ids)
    ].copy()
    solver_features.to_csv(solver_feature_path, index=False)

    print()
    print("Patient routing")
    print("---------------")
    for row in routing.itertuples(index=False):
        print(
            f"{row.patient_id}: {row.routing_mode} "
            f"(BL={row.bl_lesions}, FU={row.fu_lesions})"
        )

    print()
    print(f"Pair/matcher patients : {len(solver_ids)}")
    print(
        f"Deterministic patients: "
        f"{len(patient_ids) - len(solver_ids)}"
    )
    print(f"Deterministic rows    : {len(deterministic)}")

    if solver_ids:
        matrix_dir.mkdir(parents=True, exist_ok=True)

        if not args.reuse_matrices:
            pair_tool = root / "tools" / "generate_lesion_pair_costs.py"

            command = [
                sys.executable,
                str(pair_tool),
                "--features",
                str(solver_feature_path),
                "--out-pairs",
                str(pair_cost_path),
                "--matrix-dir",
                str(matrix_dir),
                "--distance-weight",
                str(args.distance_weight),
                "--size-weight",
                str(args.size_weight),
                "--pet-weight",
                str(args.pet_weight),
                "--distance-scale-mm",
                str(args.distance_scale_mm),
                "--pet-feature",
                str(args.pet_feature),
            ]

            if args.max_distance_mm is not None:
                command.extend(
                    ["--max-distance-mm", str(args.max_distance_mm)]
                )

            _run(
                command,
                title="STEP 2/3 - Generate lesion pair costs",
            )
        else:
            print()
            print("=" * 78)
            print("STEP 2/3 - Reuse lesion pair cost matrices")
            print("=" * 78)
            print(matrix_dir)

        matrix_paths = [
            matrix_dir / f"{patient_id}_cost_matrix.csv"
            for patient_id in solver_ids
        ]

        missing_matrices = [
            str(path)
            for path in matrix_paths
            if not path.exists()
        ]
        if missing_matrices:
            raise TrackingPipelineError(
                "Missing expected cost matrix file(s):\n  "
                + "\n  ".join(missing_matrices)
            )

        matcher_tool = root / "tools" / "run_lesion_min_cost_flow.py"

        command = [
            sys.executable,
            str(matcher_tool),
            "--cost-matrix",
            *[str(path) for path in matrix_paths],
            "--out",
            str(solver_match_path),
            "--graph-dir",
            str(graph_dir),
            "--disappearing-penalty",
            str(args.disappearing_penalty),
            "--new-lesion-penalty",
            str(args.new_lesion_penalty),
            "--merge-penalty",
            str(args.merge_penalty),
            "--max-bl-per-fu",
            str(args.max_bl_per_fu),
            "--cost-scale",
            str(args.cost_scale),
            "--cost-column",
            str(args.cost_column),
        ]

        _run(
            command,
            title="STEP 3/3 - Run NetworkX min-cost-flow matcher",
        )

    else:
        print()
        print("=" * 78)
        print("STEP 2/3 - No BL+FU patients; pair-cost generation skipped")
        print("=" * 78)
        print()
        print("=" * 78)
        print("STEP 3/3 - No optimisation decision; matcher skipped")
        print("=" * 78)

        pd.DataFrame(columns=BASE_MATCH_COLUMNS).to_csv(
            solver_match_path,
            index=False,
        )

    combined = _combine_matches(
        solver_match_path=solver_match_path,
        deterministic=deterministic,
        patient_ids=patient_ids,
        output_path=final_match_path,
    )

    counts = combined["match_type"].astype(str).str.upper().value_counts()

    print()
    print("=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)
    print(f"Selected patients : {len(patient_ids)}")
    print(f"Solver patients   : {len(solver_ids)}")
    print(
        f"Deterministic     : "
        f"{len(patient_ids) - len(solver_ids)}"
    )
    print(f"MATCHED rows      : {int(counts.get('MATCHED', 0))}")
    print(f"MERGING rows      : {int(counts.get('MERGING', 0))}")
    print(
        f"DISAPPEARING rows : "
        f"{int(counts.get('DISAPPEARING', 0))}"
    )
    print(f"NEW rows          : {int(counts.get('NEW', 0))}")
    print(f"Total outcomes    : {len(combined)}")

    print()
    print(f"Selected manifest : {selected_pairs_path}")
    print(f"Features          : {feature_path}")
    print(f"Routing           : {routing_path}")
    print(f"Pair candidates   : {pair_cost_path}")
    print(f"Cost matrices     : {matrix_dir}")
    print(f"Solver matches    : {solver_match_path}")
    print(f"Final matches     : {final_match_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(
            f"ERROR: subprocess exited with code {exc.returncode}.",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode)
    except TrackingPipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
