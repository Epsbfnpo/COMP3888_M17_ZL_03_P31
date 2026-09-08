from __future__ import annotations

import argparse
from pathlib import Path
import sys

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lesion_min_cost_flow import (
    MATCH_COLUMNS,
    FlowMatchResult,
    LesionMinCostFlowError,
    MatcherConfig,
    PatientFlowResult,
    build_patient_flow_graph,
    decode_patient_flow,
    load_patient_cost_matrix,
    solve_patient_flow,
)


def _read_csv(path: str | Path, description: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{description} CSV does not exist: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        raise LesionMinCostFlowError(
            f"Could not read {description} CSV '{csv_path}': {exc}"
        ) from exc


def _normalise_feature_ids(
    feature_path: str | Path,
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Recover complete BL/FU lesion ID lists from aligned lesion features."""
    features = _read_csv(feature_path, "aligned lesion feature")

    if "lesion_id" not in features.columns:
        if "temporary_lesion_id" in features.columns:
            features = features.rename(columns={"temporary_lesion_id": "lesion_id"})
        else:
            raise LesionMinCostFlowError(
                "Feature CSV requires lesion_id (or temporary_lesion_id)."
            )

    required = {"patient_id", "timepoint", "lesion_id"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise LesionMinCostFlowError(
            "Feature CSV is missing required column(s): " + ", ".join(missing)
        )

    table = features.copy()
    table["patient_id"] = table["patient_id"].astype(str).str.strip()
    table["timepoint"] = table["timepoint"].astype(str).str.strip().str.upper()
    table["lesion_id"] = table["lesion_id"].astype(str).str.strip()

    unknown_tp = sorted(set(table["timepoint"]) - {"BL", "FU"})
    if unknown_tp:
        raise LesionMinCostFlowError(
            "Feature timepoint must contain only BL/FU; found: "
            + ", ".join(unknown_tp)
        )

    result: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for patient_id in sorted(table["patient_id"].unique()):
        patient = table[table["patient_id"] == patient_id]
        bl_ids = tuple(
            sorted(patient.loc[patient["timepoint"] == "BL", "lesion_id"].unique())
        )
        fu_ids = tuple(
            sorted(patient.loc[patient["timepoint"] == "FU", "lesion_id"].unique())
        )
        result[str(patient_id)] = (bl_ids, fu_ids)

    return result


def _node_label(node: object) -> str:
    if isinstance(node, tuple):
        return "::".join(str(part) for part in node)
    return str(node)


def _export_graph_edges(
    patient_id: str,
    graph: nx.DiGraph,
    flow: dict | object,
    output_dir: str | Path,
) -> Path:
    """Export graph structure plus selected flow for debugging/inspection."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for source, target, data in graph.edges(data=True):
        amount = int(flow.get(source, {}).get(target, 0))  # type: ignore[attr-defined]
        rows.append(
            {
                "patient_id": patient_id,
                "source": _node_label(source),
                "target": _node_label(target),
                "edge_type": data.get("edge_type"),
                "capacity": data.get("capacity"),
                "weight": data.get("weight"),
                "original_cost": data.get("original_cost"),
                "base_pair_cost": data.get("base_pair_cost"),
                "event_cost": data.get("event_cost"),
                "selected_flow": amount,
            }
        )

    path = out_dir / f"{patient_id}_flow_graph_edges.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_matches(matches: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(path, index=False)
    return path


def _pair_cost_patient_inputs(
    pair_cost_path: str | Path,
    feature_path: str | Path | None,
) -> dict[str, tuple[pd.DataFrame, tuple[str, ...] | None, tuple[str, ...] | None]]:
    """Prepare matcher inputs from the legacy long-form pair-cost CSV."""
    pair_costs = _read_csv(pair_cost_path, "pair cost")
    complete_ids = _normalise_feature_ids(feature_path) if feature_path else None

    patient_ids: set[str] = set()
    if not pair_costs.empty:
        if "patient_id" not in pair_costs.columns:
            raise LesionMinCostFlowError("Pair-cost CSV requires patient_id.")
        patient_ids.update(pair_costs["patient_id"].astype(str).str.strip())
    if complete_ids is not None:
        patient_ids.update(complete_ids)

    result = {}
    for patient_id in sorted(patient_ids):
        if pair_costs.empty:
            patient_pairs = pair_costs.copy()
        else:
            patient_pairs = pair_costs[
                pair_costs["patient_id"].astype(str).str.strip() == patient_id
            ].copy()

        if complete_ids is not None:
            if patient_id not in complete_ids:
                raise LesionMinCostFlowError(
                    f"Patient {patient_id} appears in pair costs but not in feature CSV."
                )
            bl_ids, fu_ids = complete_ids[patient_id]
        else:
            bl_ids = None
            fu_ids = None

        result[patient_id] = (patient_pairs, bl_ids, fu_ids)

    return result


def _matrix_patient_inputs(
    matrix_paths: list[str],
    *,
    patient_id_override: str | None,
    cost_column: str,
) -> dict[str, tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]]:
    """Load one or more generated BL x FU matrices directly."""
    if patient_id_override is not None and len(matrix_paths) != 1:
        raise LesionMinCostFlowError(
            "--patient-id can only be used when exactly one --cost-matrix is supplied."
        )

    result = {}
    for path in matrix_paths:
        loaded = load_patient_cost_matrix(
            path,
            patient_id=patient_id_override,
            cost_column=cost_column,
        )
        if loaded.patient_id in result:
            raise LesionMinCostFlowError(
                f"Duplicate patient cost matrix supplied: {loaded.patient_id}"
            )
        result[loaded.patient_id] = (
            loaded.pair_costs,
            loaded.bl_lesion_ids,
            loaded.fu_lesion_ids,
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run longitudinal lesion matching with NetworkX min-cost flow. "
            "The matcher can read the generated labelled BL x FU cost matrix "
            "directly, or the legacy long-form pair-cost CSV. Supports MATCHED, "
            "DISAPPEARING, NEW, and optional slot-based MERGING through explicit "
            "MERGE_EVENT nodes."
        )
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cost-matrix",
        nargs="+",
        help=(
            "One or more labelled BL x FU cost-matrix CSVs produced in "
            "outputs/lesion_pair_cost_matrices. This is the preferred direct input."
        ),
    )
    source.add_argument(
        "--pair-costs",
        help="Legacy long-form CSV produced by generate_lesion_pair_costs.py.",
    )

    parser.add_argument(
        "--patient-id",
        help=(
            "Optional patient ID override for a single cost matrix. Normally the "
            "ID is inferred from '<patient_id>_cost_matrix.csv'."
        ),
    )
    parser.add_argument(
        "--features",
        help=(
            "Optional aligned lesion-feature CSV for --pair-costs mode. It is not "
            "needed with --cost-matrix because matrix rows/columns already preserve "
            "the complete lesion ID lists."
        ),
    )
    parser.add_argument(
        "--out",
        default="outputs/lesion_matches.csv",
        help="Decoded lesion outcome CSV.",
    )
    parser.add_argument(
        "--graph-dir",
        default="outputs/lesion_flow_graphs",
        help="Directory for per-patient graph-edge CSVs.",
    )
    parser.add_argument("--disappearing-penalty", type=float, default=1.0)
    parser.add_argument("--new-lesion-penalty", type=float, default=1.0)
    parser.add_argument(
        "--merge-penalty",
        type=float,
        default=0.2,
        help="Additional cost charged for each BL routed through MERGE_EVENT.",
    )
    parser.add_argument(
        "--max-bl-per-fu",
        type=int,
        default=1,
        help=(
            "Maximum BL lesions assignable to one FU. Use 1 for one-to-one; "
            "use 2+ to enable merge slots."
        ),
    )
    parser.add_argument("--cost-scale", type=int, default=1000)
    parser.add_argument("--cost-column", default="total_cost")
    args = parser.parse_args()

    if args.cost_matrix and args.features:
        parser.error("--features is only used with --pair-costs, not --cost-matrix.")
    if args.pair_costs and args.patient_id:
        parser.error("--patient-id is only used with --cost-matrix.")

    config = MatcherConfig(
        disappearing_penalty=args.disappearing_penalty,
        new_lesion_penalty=args.new_lesion_penalty,
        merge_penalty=args.merge_penalty,
        max_bl_per_fu=args.max_bl_per_fu,
        cost_scale=args.cost_scale,
        cost_column=args.cost_column,
    )

    if args.cost_matrix:
        patient_inputs = _matrix_patient_inputs(
            args.cost_matrix,
            patient_id_override=args.patient_id,
            cost_column=args.cost_column,
        )
        input_description = "cost matrix"
    else:
        patient_inputs = _pair_cost_patient_inputs(args.pair_costs, args.features)
        input_description = "pair-cost table"

    if not patient_inputs:
        raise LesionMinCostFlowError(f"No patients found in {input_description} input.")

    patient_results: dict[str, PatientFlowResult] = {}
    match_frames: list[pd.DataFrame] = []

    for patient_id in sorted(patient_inputs):
        patient_pairs, bl_ids, fu_ids = patient_inputs[patient_id]

        # PART 1 — build lesion-specific network.
        graph = build_patient_flow_graph(
            patient_pairs,
            config=config,
            bl_lesion_ids=bl_ids,
            fu_lesion_ids=fu_ids,
        )

        # PART 2 — let NetworkX solve the generic min-cost-flow problem.
        flow = solve_patient_flow(graph)

        # PART 3 — decode graph flow back to lesion-level events.
        matches = decode_patient_flow(patient_id, graph, flow)

        scaled_total_cost = int(nx.cost_of_flow(graph, flow, weight="weight"))
        total_cost = float(matches["match_cost"].sum()) if not matches.empty else 0.0

        patient_result = PatientFlowResult(
            patient_id=patient_id,
            matches=matches,
            flow=flow,
            scaled_total_cost=scaled_total_cost,
            total_cost=total_cost,
        )
        patient_results[patient_id] = patient_result
        match_frames.append(matches)

        graph_path = _export_graph_edges(patient_id, graph, flow, args.graph_dir)

        counts = matches["match_type"].value_counts() if not matches.empty else {}
        matched = int(counts.get("MATCHED", 0))
        merging = int(counts.get("MERGING", 0))
        disappearing = int(counts.get("DISAPPEARING", 0))
        new = int(counts.get("NEW", 0))
        merge_events = (
            int(matches.loc[matches["match_type"] == "MERGING", "event_id"].nunique())
            if not matches.empty
            else 0
        )

        print(
            f"  {patient_id}: matched={matched}, merging_rows={merging}, "
            f"merge_events={merge_events}, disappearing={disappearing}, new={new}, "
            f"objective={total_cost:.6f}, scaled_objective={scaled_total_cost}"
        )
        print(f"    graph: {graph_path}")

    if match_frames:
        concat_frames = [
            frame.dropna(axis=1, how="all")
            for frame in match_frames
            if not frame.empty
        ]
        if concat_frames:
            all_matches = pd.concat(
                concat_frames,
                ignore_index=True,
                sort=False,
            ).reindex(columns=MATCH_COLUMNS)
        else:
            all_matches = pd.DataFrame(columns=MATCH_COLUMNS)
    else:
        all_matches = pd.DataFrame(columns=MATCH_COLUMNS)

    result = FlowMatchResult(
        matches=all_matches,
        patient_results=patient_results,
    )

    output_path = _write_matches(result.matches, args.out)
    print(f"Input mode: {input_description}")
    print(f"Wrote {len(result.matches)} lesion outcome(s): {output_path}")


if __name__ == "__main__":
    main()
