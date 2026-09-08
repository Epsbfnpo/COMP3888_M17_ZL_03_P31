from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import networkx as nx
import numpy as np
import pandas as pd


class LesionMinCostFlowError(ValueError):
    """Raised when lesion pair costs cannot be converted into a valid flow problem."""


@dataclass(frozen=True)
class MatcherConfig:
    """Configuration for longitudinal lesion min-cost-flow matching.

    The matcher supports:
    - one-to-one BL -> FU matching
    - disappearing BL lesions
    - new FU lesions
    - optional many-to-one MERGING through explicit merge-event nodes

    ``max_bl_per_fu`` controls whether merging is enabled:
    - 1: one-to-one behaviour only (backward-compatible baseline)
    - 2+: one primary FU slot plus additional merge slots

    ``merge_penalty`` is charged once for each additional BL lesion routed
    through a MERGE event.  It is deliberately separate from the BL/FU pair
    cost so that the merge-event model can be replaced or improved later.
    """

    disappearing_penalty: float = 1.0
    new_lesion_penalty: float = 1.0
    merge_penalty: float = 0.2
    max_bl_per_fu: int = 1
    cost_scale: int = 1000
    cost_column: str = "total_cost"


@dataclass(frozen=True)
class PatientFlowResult:
    patient_id: str
    matches: pd.DataFrame
    flow: Mapping[object, Mapping[object, int]]
    scaled_total_cost: int
    total_cost: float


@dataclass(frozen=True)
class FlowMatchResult:
    matches: pd.DataFrame
    patient_results: Mapping[str, PatientFlowResult]



@dataclass(frozen=True)
class LoadedCostMatrix:
    """A labelled BL x FU cost matrix converted into matcher-ready inputs.

    ``pair_costs`` contains only finite candidate edges. Positive infinity in
    the matrix is treated as a gated/non-candidate edge, while the complete BL
    and FU lesion ID lists are preserved separately so unmatched lesions remain
    representable.
    """

    patient_id: str
    pair_costs: pd.DataFrame
    bl_lesion_ids: tuple[str, ...]
    fu_lesion_ids: tuple[str, ...]


MATCH_COLUMNS = [
    "patient_id",
    "bl_lesion_id",
    "fu_lesion_id",
    "match_type",
    "assignment_role",
    "event_id",
    "pair_cost",
    "event_cost",
    "match_cost",
]


# =============================================================================
# Shared validation / utility helpers
# =============================================================================


def _validate_config(config: MatcherConfig) -> None:
    for name, value in (
        ("disappearing_penalty", config.disappearing_penalty),
        ("new_lesion_penalty", config.new_lesion_penalty),
        ("merge_penalty", config.merge_penalty),
    ):
        if not np.isfinite(value) or value < 0:
            raise LesionMinCostFlowError(
                f"{name} must be a finite value >= 0."
            )

    if not isinstance(config.max_bl_per_fu, int) or config.max_bl_per_fu < 1:
        raise LesionMinCostFlowError(
            "max_bl_per_fu must be an integer >= 1."
        )

    if not isinstance(config.cost_scale, int) or config.cost_scale <= 0:
        raise LesionMinCostFlowError(
            "cost_scale must be a positive integer."
        )

    if not str(config.cost_column).strip():
        raise LesionMinCostFlowError(
            "cost_column must not be blank."
        )


def _scale_cost(value: float, cost_scale: int) -> int:
    """Convert a floating-point cost into an integer NetworkX edge weight."""
    if not np.isfinite(value) or value < 0:
        raise LesionMinCostFlowError(
            f"Flow edge cost must be finite and >= 0; found {value!r}."
        )

    return int(round(float(value) * cost_scale))


def _normalise_ids(values: Iterable[object]) -> tuple[str, ...]:
    ids = tuple(sorted({str(value).strip() for value in values}))

    if any(value == "" for value in ids):
        raise LesionMinCostFlowError(
            "Lesion identifiers must not be blank."
        )

    return ids


def _validate_patient_pair_costs(
    pair_costs: pd.DataFrame,
    config: MatcherConfig,
) -> pd.DataFrame:
    required = {
        "patient_id",
        "bl_lesion_id",
        "fu_lesion_id",
        config.cost_column,
    }

    missing = sorted(required - set(pair_costs.columns))
    if missing:
        raise LesionMinCostFlowError(
            "Pair-cost table is missing required column(s): "
            + ", ".join(missing)
        )

    table = pair_costs.copy()

    if table.empty:
        return table

    table["patient_id"] = table["patient_id"].astype(str).str.strip()
    table["bl_lesion_id"] = table["bl_lesion_id"].astype(str).str.strip()
    table["fu_lesion_id"] = table["fu_lesion_id"].astype(str).str.strip()

    if (table["patient_id"] == "").any():
        raise LesionMinCostFlowError("patient_id must not be blank.")

    if (table["bl_lesion_id"] == "").any():
        raise LesionMinCostFlowError("bl_lesion_id must not be blank.")

    if (table["fu_lesion_id"] == "").any():
        raise LesionMinCostFlowError("fu_lesion_id must not be blank.")

    table[config.cost_column] = pd.to_numeric(
        table[config.cost_column],
        errors="coerce",
    )

    costs = table[config.cost_column].to_numpy(dtype=float)

    if not np.all(np.isfinite(costs)):
        raise LesionMinCostFlowError(
            f"{config.cost_column} must contain finite values."
        )

    if np.any(costs < 0):
        raise LesionMinCostFlowError(
            f"{config.cost_column} must contain values >= 0."
        )

    duplicate = table.duplicated(
        ["patient_id", "bl_lesion_id", "fu_lesion_id"],
        keep=False,
    )

    if duplicate.any():
        row = table.loc[
            duplicate,
            ["patient_id", "bl_lesion_id", "fu_lesion_id"],
        ].iloc[0]

        raise LesionMinCostFlowError(
            "Duplicate BL/FU candidate edge found: "
            f"{row['patient_id']} "
            f"{row['bl_lesion_id']} -> {row['fu_lesion_id']}"
        )

    return table



def _infer_patient_id_from_matrix_path(path: Path) -> str:
    """Infer patient ID from ``<patient>_cost_matrix.csv`` output naming."""
    name = path.name
    suffix = "_cost_matrix.csv"
    if name.lower().endswith(suffix):
        patient_id = name[: -len(suffix)].strip()
        if patient_id:
            return patient_id

    raise LesionMinCostFlowError(
        "Could not infer patient_id from cost-matrix filename. Expected "
        "'<patient_id>_cost_matrix.csv'; pass patient_id explicitly instead."
    )


def load_patient_cost_matrix(
    path: str | Path,
    *,
    patient_id: str | None = None,
    cost_column: str = "total_cost",
) -> LoadedCostMatrix:
    """Load a labelled BL x FU cost-matrix CSV produced by pair-cost generation.

    Expected format::

        bl_lesion_id,FU_L001,FU_L002
        BL_L001,0.12,0.83
        BL_L002,inf,0.09

    Matrix rows are BL lesion IDs and columns are FU lesion IDs. Finite cells
    become candidate edges in a long-form matcher table. ``inf`` cells are
    treated as gated/non-candidate edges and are omitted from ``pair_costs``.
    Complete row/column lesion ID lists are retained so a fully gated lesion can
    still become DISAPPEARING or NEW in the flow network.
    """
    matrix_path = Path(path)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Cost matrix CSV does not exist: {matrix_path}")

    try:
        matrix = pd.read_csv(matrix_path)
    except Exception as exc:
        raise LesionMinCostFlowError(
            f"Could not read cost matrix CSV '{matrix_path}': {exc}"
        ) from exc

    if "bl_lesion_id" not in matrix.columns:
        raise LesionMinCostFlowError(
            "Cost matrix must contain a 'bl_lesion_id' first/index column."
        )

    resolved_patient_id = (
        str(patient_id).strip()
        if patient_id is not None
        else _infer_patient_id_from_matrix_path(matrix_path)
    )
    if not resolved_patient_id:
        raise LesionMinCostFlowError("patient_id must not be blank.")

    if not str(cost_column).strip():
        raise LesionMinCostFlowError("cost_column must not be blank.")

    bl_series = matrix["bl_lesion_id"].astype(str).str.strip()
    if (bl_series == "").any():
        raise LesionMinCostFlowError("Cost matrix BL lesion IDs must not be blank.")
    if bl_series.duplicated().any():
        duplicate = bl_series[bl_series.duplicated(keep=False)].iloc[0]
        raise LesionMinCostFlowError(
            f"Duplicate BL lesion ID in cost matrix: {duplicate}"
        )

    fu_ids = tuple(str(column).strip() for column in matrix.columns if column != "bl_lesion_id")
    if any(not fu_id for fu_id in fu_ids):
        raise LesionMinCostFlowError("Cost matrix FU lesion IDs must not be blank.")
    if len(set(fu_ids)) != len(fu_ids):
        raise LesionMinCostFlowError("Cost matrix contains duplicate FU lesion IDs.")

    bl_ids = tuple(bl_series.tolist())
    rows: list[dict[str, object]] = []

    for row_index, bl_id in enumerate(bl_ids):
        for fu_id in fu_ids:
            raw = matrix.iloc[row_index][fu_id]
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise LesionMinCostFlowError(
                    f"Cost matrix value for {bl_id} -> {fu_id} is not numeric: {raw!r}."
                ) from exc

            if np.isnan(value):
                raise LesionMinCostFlowError(
                    f"Cost matrix value for {bl_id} -> {fu_id} must not be NaN."
                )
            if np.isneginf(value) or value < 0:
                raise LesionMinCostFlowError(
                    f"Cost matrix value for {bl_id} -> {fu_id} must be >= 0 or inf."
                )
            if np.isposinf(value):
                continue

            rows.append(
                {
                    "patient_id": resolved_patient_id,
                    "bl_lesion_id": bl_id,
                    "fu_lesion_id": fu_id,
                    str(cost_column): value,
                }
            )

    pair_costs = pd.DataFrame(
        rows,
        columns=[
            "patient_id",
            "bl_lesion_id",
            "fu_lesion_id",
            str(cost_column),
        ],
    )

    return LoadedCostMatrix(
        patient_id=resolved_patient_id,
        pair_costs=pair_costs,
        bl_lesion_ids=bl_ids,
        fu_lesion_ids=fu_ids,
    )


def match_patient_cost_matrix(
    path: str | Path,
    config: MatcherConfig = MatcherConfig(),
    *,
    patient_id: str | None = None,
) -> PatientFlowResult:
    """Load a generated cost matrix and run matching directly from it."""
    loaded = load_patient_cost_matrix(
        path,
        patient_id=patient_id,
        cost_column=config.cost_column,
    )
    return match_patient_lesions(
        loaded.pair_costs,
        config=config,
        patient_id=loaded.patient_id,
        bl_lesion_ids=loaded.bl_lesion_ids,
        fu_lesion_ids=loaded.fu_lesion_ids,
    )


def _primary_fu_node(fu_id: str) -> tuple[str, str]:
    return ("FU_PRIMARY", fu_id)


def _merge_event_node(fu_id: str) -> tuple[str, str]:
    return ("MERGE_EVENT", fu_id)


def _merge_slot_node(fu_id: str, slot_index: int) -> tuple[str, str, int]:
    return ("FU_MERGE_SLOT", fu_id, slot_index)


# =============================================================================
# PART 1 — NETWORK CONSTRUCTION
# =============================================================================


def build_patient_flow_graph(
    patient_pair_costs: pd.DataFrame,
    config: MatcherConfig = MatcherConfig(),
    *,
    bl_lesion_ids: Iterable[str] | None = None,
    fu_lesion_ids: Iterable[str] | None = None,
) -> nx.DiGraph:
    """Build the lesion matching flow network for one patient.

    Base graph events
    -----------------
    BL -> FU_PRIMARY
        Normal one-to-one candidate match.

    BL -> DISAPPEAR
        Baseline lesion disappears.

    NEW -> FU_PRIMARY
        Follow-up lesion is new.

    NEW -> DISAPPEAR
        Zero-cost balancing flow with no biological meaning.

    Merge extension
    ---------------
    When ``max_bl_per_fu > 1`` each FU lesion receives:

        one FU_PRIMARY slot
        one explicit MERGE_EVENT node
        max_bl_per_fu - 1 optional FU_MERGE_SLOT nodes

    A BL lesion can therefore take either:

        BL -> FU_PRIMARY

    or:

        BL -> MERGE_EVENT -> FU_MERGE_SLOT

    The BL -> MERGE_EVENT edge carries the ordinary pair cost.  The
    MERGE_EVENT -> FU_MERGE_SLOT edge carries ``merge_penalty``.  Keeping the
    event as its own node is intentional: later versions can replace the simple
    fixed merge penalty with event-level features such as combined lesion
    volume, geometry, or PET evidence without redesigning the entire matcher.

    Every BL lesion still supplies exactly one unit, so one BL cannot match
    multiple FU lesions.  Each FU receives one required primary slot, while
    merge slots are optional because NEW can fill them at zero cost.
    """
    _validate_config(config)
    table = _validate_patient_pair_costs(patient_pair_costs, config)

    patient_ids = sorted(table["patient_id"].unique()) if not table.empty else []

    if len(patient_ids) > 1:
        raise LesionMinCostFlowError(
            "build_patient_flow_graph() accepts one patient at a time."
        )

    if bl_lesion_ids is None:
        bl_ids = _normalise_ids(table["bl_lesion_id"]) if not table.empty else ()
    else:
        bl_ids = _normalise_ids(bl_lesion_ids)

    if fu_lesion_ids is None:
        fu_ids = _normalise_ids(table["fu_lesion_id"]) if not table.empty else ()
    else:
        fu_ids = _normalise_ids(fu_lesion_ids)

    bl_set = set(bl_ids)
    fu_set = set(fu_ids)

    if not table.empty:
        unknown_bl = sorted(set(table["bl_lesion_id"]) - bl_set)
        unknown_fu = sorted(set(table["fu_lesion_id"]) - fu_set)

        if unknown_bl:
            raise LesionMinCostFlowError(
                "Candidate table contains BL lesion IDs not present in "
                "bl_lesion_ids: " + ", ".join(unknown_bl)
            )

        if unknown_fu:
            raise LesionMinCostFlowError(
                "Candidate table contains FU lesion IDs not present in "
                "fu_lesion_ids: " + ", ".join(unknown_fu)
            )

    graph = nx.DiGraph()

    new_node = ("SPECIAL", "NEW")
    disappear_node = ("SPECIAL", "DISAPPEAR")

    merge_slots_per_fu = config.max_bl_per_fu - 1
    total_fu_slots = len(fu_ids) * config.max_bl_per_fu

    graph.graph["max_bl_per_fu"] = config.max_bl_per_fu
    graph.graph["merge_penalty"] = float(config.merge_penalty)
    graph.graph["cost_scale"] = config.cost_scale
    graph.graph["cost_column"] = config.cost_column

    # -------------------------------------------------------------------------
    # Nodes and demands
    # -------------------------------------------------------------------------
    # NetworkX convention:
    #   negative demand = supply
    #   positive demand = required flow
    #
    # BL lesions each supply one unit.
    # NEW supplies one unit for every FU slot (primary + optional merge slots).
    # Every FU slot consumes one unit.
    # DISAPPEAR consumes one unit for every BL lesion.
    # -------------------------------------------------------------------------

    for lesion_id in bl_ids:
        graph.add_node(
            ("BL", lesion_id),
            demand=-1,
            lesion_id=lesion_id,
            node_type="BL",
        )

    for fu_id in fu_ids:
        primary_node = _primary_fu_node(fu_id)
        graph.add_node(
            primary_node,
            demand=1,
            lesion_id=fu_id,
            node_type="FU_PRIMARY",
        )

        if merge_slots_per_fu > 0:
            event_node = _merge_event_node(fu_id)
            graph.add_node(
                event_node,
                demand=0,
                lesion_id=fu_id,
                fu_lesion_id=fu_id,
                node_type="MERGE_EVENT",
                event_type="MERGING",
                event_cost=float(config.merge_penalty),
            )

            for slot_index in range(1, merge_slots_per_fu + 1):
                slot_node = _merge_slot_node(fu_id, slot_index)
                graph.add_node(
                    slot_node,
                    demand=1,
                    lesion_id=fu_id,
                    fu_lesion_id=fu_id,
                    slot_index=slot_index,
                    node_type="FU_MERGE_SLOT",
                )

    graph.add_node(
        new_node,
        demand=-total_fu_slots,
        node_type="NEW",
    )

    graph.add_node(
        disappear_node,
        demand=len(bl_ids),
        node_type="DISAPPEAR",
    )

    # -------------------------------------------------------------------------
    # 1. BL -> FU primary candidate edges
    # 2. BL -> MERGE_EVENT candidate edges
    # -------------------------------------------------------------------------

    for _, row in table.iterrows():
        bl_id = str(row["bl_lesion_id"])
        fu_id = str(row["fu_lesion_id"])
        pair_cost = float(row[config.cost_column])
        pair_weight = _scale_cost(pair_cost, config.cost_scale)
        bl_node = ("BL", bl_id)

        graph.add_edge(
            bl_node,
            _primary_fu_node(fu_id),
            capacity=1,
            weight=pair_weight,
            base_pair_cost=pair_cost,
            event_cost=0.0,
            original_cost=pair_cost,
            edge_type="MATCH",
        )

        if merge_slots_per_fu > 0:
            # The pair similarity stays on this edge.  The merge-specific cost
            # is deliberately kept on MERGE_EVENT -> FU_MERGE_SLOT below.
            graph.add_edge(
                bl_node,
                _merge_event_node(fu_id),
                capacity=1,
                weight=pair_weight,
                base_pair_cost=pair_cost,
                event_cost=0.0,
                original_cost=pair_cost,
                edge_type="MERGE_CANDIDATE",
            )

    # -------------------------------------------------------------------------
    # 3. MERGE_EVENT -> FU merge slots
    # -------------------------------------------------------------------------

    if merge_slots_per_fu > 0:
        merge_weight = _scale_cost(
            config.merge_penalty,
            config.cost_scale,
        )

        for fu_id in fu_ids:
            event_node = _merge_event_node(fu_id)

            for slot_index in range(1, merge_slots_per_fu + 1):
                slot_node = _merge_slot_node(fu_id, slot_index)
                graph.add_edge(
                    event_node,
                    slot_node,
                    capacity=1,
                    weight=merge_weight,
                    base_pair_cost=0.0,
                    event_cost=float(config.merge_penalty),
                    original_cost=float(config.merge_penalty),
                    edge_type="MERGE_EVENT",
                )

    # -------------------------------------------------------------------------
    # 4. BL -> DISAPPEAR
    # -------------------------------------------------------------------------

    disappearing_weight = _scale_cost(
        config.disappearing_penalty,
        config.cost_scale,
    )

    for bl_id in bl_ids:
        graph.add_edge(
            ("BL", bl_id),
            disappear_node,
            capacity=1,
            weight=disappearing_weight,
            base_pair_cost=0.0,
            event_cost=float(config.disappearing_penalty),
            original_cost=float(config.disappearing_penalty),
            edge_type="DISAPPEARING",
        )

    # -------------------------------------------------------------------------
    # 5. NEW -> FU primary
    # -------------------------------------------------------------------------

    new_weight = _scale_cost(
        config.new_lesion_penalty,
        config.cost_scale,
    )

    for fu_id in fu_ids:
        graph.add_edge(
            new_node,
            _primary_fu_node(fu_id),
            capacity=1,
            weight=new_weight,
            base_pair_cost=0.0,
            event_cost=float(config.new_lesion_penalty),
            original_cost=float(config.new_lesion_penalty),
            edge_type="NEW",
        )

    # -------------------------------------------------------------------------
    # 6. NEW -> FU merge slots
    # -------------------------------------------------------------------------
    # A merge slot is optional.  NEW filling a merge slot at zero cost means
    # only "this additional merge position is unused".  It is NOT decoded as a
    # biological NEW lesion event.
    # -------------------------------------------------------------------------

    if merge_slots_per_fu > 0:
        for fu_id in fu_ids:
            for slot_index in range(1, merge_slots_per_fu + 1):
                graph.add_edge(
                    new_node,
                    _merge_slot_node(fu_id, slot_index),
                    capacity=1,
                    weight=0,
                    base_pair_cost=0.0,
                    event_cost=0.0,
                    original_cost=0.0,
                    edge_type="UNUSED_MERGE_SLOT",
                )

    # -------------------------------------------------------------------------
    # 7. NEW -> DISAPPEAR balancing edge
    # -------------------------------------------------------------------------
    # Every BL routed into any FU slot replaces one unit that otherwise would
    # have been supplied by NEW.  The displaced NEW unit is sent here.
    # -------------------------------------------------------------------------

    graph.add_edge(
        new_node,
        disappear_node,
        capacity=min(len(bl_ids), total_fu_slots),
        weight=0,
        base_pair_cost=0.0,
        event_cost=0.0,
        original_cost=0.0,
        edge_type="BALANCE",
    )

    total_demand = sum(
        int(data.get("demand", 0))
        for _, data in graph.nodes(data=True)
    )

    if total_demand != 0:
        raise LesionMinCostFlowError(
            f"Flow graph demand must sum to zero; found {total_demand}."
        )

    return graph


# =============================================================================
# PART 2 — NETWORKX MIN-COST-FLOW SOLVING
# =============================================================================


def solve_patient_flow(
    graph: nx.DiGraph,
) -> Mapping[object, Mapping[object, int]]:
    """Solve a prepared lesion graph with NetworkX ``min_cost_flow``.

    NetworkX is intentionally isolated in this function.  The project owns the
    lesion-specific graph design; NetworkX only solves the generic constrained
    minimum-cost-flow problem represented by that graph.
    """
    try:
        return nx.min_cost_flow(
            graph,
            demand="demand",
            capacity="capacity",
            weight="weight",
        )
    except nx.NetworkXUnfeasible as exc:
        raise LesionMinCostFlowError(
            "No feasible lesion matching flow exists for this patient."
        ) from exc
    except nx.NetworkXError as exc:
        raise LesionMinCostFlowError(
            f"NetworkX could not solve lesion matching flow: {exc}"
        ) from exc


# =============================================================================
# PART 3 — FLOW DECODING
# =============================================================================


def decode_patient_flow(
    patient_id: str,
    graph: nx.DiGraph,
    flow: Mapping[object, Mapping[object, int]],
) -> pd.DataFrame:
    """Decode NetworkX flow into lesion-level events.

    Important merge behaviour
    -------------------------
    The graph keeps MERGE_EVENT separate from ordinary matching.  During
    decoding, however, topology is determined by the final set of BL lesions
    mapped to each FU lesion:

    - one BL -> one FU: MATCHED
    - multiple BL -> one FU: all involved rows are labelled MERGING

    All rows belonging to one merge share the same ``event_id``.  Internal
    primary/merge-slot structure is preserved only in ``assignment_role`` and
    is not treated as biological ground truth.
    """
    rows: list[dict[str, object]] = []

    new_node = ("SPECIAL", "NEW")
    disappear_node = ("SPECIAL", "DISAPPEAR")

    # -------------------------------------------------------------------------
    # Decode one selected outcome for every BL lesion.
    # -------------------------------------------------------------------------

    bl_nodes = sorted(
        (
            node
            for node, data in graph.nodes(data=True)
            if data.get("node_type") == "BL"
        ),
        key=lambda node: str(node[1]),
    )

    for bl_node in bl_nodes:
        bl_id = str(bl_node[1])

        selected_edges = [
            (target, amount)
            for target, amount in flow.get(bl_node, {}).items()
            if amount > 0
        ]

        if len(selected_edges) != 1:
            raise LesionMinCostFlowError(
                f"BL lesion {bl_id} must have exactly one selected outcome; "
                f"found {len(selected_edges)}."
            )

        target, _ = selected_edges[0]
        edge_data = graph[bl_node][target]

        if target == disappear_node:
            event_cost = float(edge_data["original_cost"])
            rows.append(
                {
                    "patient_id": patient_id,
                    "bl_lesion_id": bl_id,
                    "fu_lesion_id": None,
                    "match_type": "DISAPPEARING",
                    "assignment_role": "DISAPPEAR",
                    "event_id": None,
                    "pair_cost": None,
                    "event_cost": event_cost,
                    "match_cost": event_cost,
                }
            )
            continue

        if (
            isinstance(target, tuple)
            and len(target) == 2
            and target[0] == "FU_PRIMARY"
        ):
            fu_id = str(target[1])
            pair_cost = float(edge_data["base_pair_cost"])
            rows.append(
                {
                    "patient_id": patient_id,
                    "bl_lesion_id": bl_id,
                    "fu_lesion_id": fu_id,
                    "match_type": "MATCHED",  # may become MERGING below
                    "assignment_role": "PRIMARY",
                    "event_id": None,
                    "pair_cost": pair_cost,
                    "event_cost": 0.0,
                    "match_cost": pair_cost,
                }
            )
            continue

        if (
            isinstance(target, tuple)
            and len(target) == 2
            and target[0] == "MERGE_EVENT"
        ):
            fu_id = str(target[1])
            pair_cost = float(edge_data["base_pair_cost"])
            merge_event_cost = float(
                graph.nodes[target].get("event_cost", 0.0)
            )

            rows.append(
                {
                    "patient_id": patient_id,
                    "bl_lesion_id": bl_id,
                    "fu_lesion_id": fu_id,
                    "match_type": "MERGING",
                    "assignment_role": "MERGE",
                    "event_id": None,  # populated after grouping below
                    "pair_cost": pair_cost,
                    "event_cost": merge_event_cost,
                    "match_cost": pair_cost + merge_event_cost,
                }
            )
            continue

        raise LesionMinCostFlowError(
            f"Unexpected selected target for BL lesion {bl_id}: {target!r}"
        )

    # -------------------------------------------------------------------------
    # Decode biological NEW events from NEW -> FU_PRIMARY only.
    # NEW -> FU_MERGE_SLOT means an unused optional slot and is ignored.
    # -------------------------------------------------------------------------

    for target, amount in flow.get(new_node, {}).items():
        if amount <= 0:
            continue

        if target == disappear_node:
            continue

        if (
            isinstance(target, tuple)
            and len(target) == 2
            and target[0] == "FU_PRIMARY"
        ):
            edge_data = graph[new_node][target]
            event_cost = float(edge_data["original_cost"])

            rows.append(
                {
                    "patient_id": patient_id,
                    "bl_lesion_id": None,
                    "fu_lesion_id": str(target[1]),
                    "match_type": "NEW",
                    "assignment_role": "NEW",
                    "event_id": None,
                    "pair_cost": None,
                    "event_cost": event_cost,
                    "match_cost": event_cost,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    # -------------------------------------------------------------------------
    # Final topology classification.
    # -------------------------------------------------------------------------
    # The internal MERGE_EVENT path is a modelling mechanism.  A biological
    # MERGING event exists only when >1 BL lesion ends up mapped to the same FU.
    # When that occurs, ALL participating BL rows are labelled MERGING and share
    # one event_id, including the BL that happened to occupy FU_PRIMARY.
    # -------------------------------------------------------------------------

    bl_to_fu_mask = (
        result["bl_lesion_id"].notna()
        & result["fu_lesion_id"].notna()
    )

    mapped = result.loc[bl_to_fu_mask]

    if not mapped.empty:
        counts = mapped.groupby("fu_lesion_id")["bl_lesion_id"].count()
        merging_fu_ids = set(counts[counts > 1].index.astype(str))

        for fu_id in merging_fu_ids:
            event_id = f"MERGE::{patient_id}::{fu_id}"
            mask = (
                result["fu_lesion_id"].astype("string") == fu_id
            ) & result["bl_lesion_id"].notna()

            result.loc[mask, "match_type"] = "MERGING"
            result.loc[mask, "event_id"] = event_id

        # Defensive fallback: if a MERGE_EVENT edge was used but the final FU
        # has only one BL assignment (possible only in a cost tie / degenerate
        # configuration), treat the biological topology as MATCHED.  The role is
        # still kept as MERGE so the internal solver choice remains inspectable.
        single_fu_ids = set(counts[counts == 1].index.astype(str))
        for fu_id in single_fu_ids:
            mask = (
                result["fu_lesion_id"].astype("string") == fu_id
            ) & result["bl_lesion_id"].notna()
            result.loc[mask, "match_type"] = "MATCHED"
            result.loc[mask, "event_id"] = None

    return result[MATCH_COLUMNS].reset_index(drop=True)


def match_patient_lesions(
    patient_pair_costs: pd.DataFrame,
    config: MatcherConfig = MatcherConfig(),
    *,
    patient_id: str | None = None,
    bl_lesion_ids: Iterable[str] | None = None,
    fu_lesion_ids: Iterable[str] | None = None,
) -> PatientFlowResult:
    """Build, solve, and decode one patient's lesion flow problem."""
    _validate_config(config)
    table = _validate_patient_pair_costs(patient_pair_costs, config)

    discovered_ids = (
        sorted(table["patient_id"].unique())
        if not table.empty
        else []
    )

    if patient_id is None:
        if len(discovered_ids) != 1:
            raise LesionMinCostFlowError(
                "patient_id must be supplied when the input table does not "
                "contain exactly one patient."
            )
        resolved_patient_id = str(discovered_ids[0])
    else:
        resolved_patient_id = str(patient_id).strip()

        if not resolved_patient_id:
            raise LesionMinCostFlowError(
                "patient_id must not be blank."
            )

        if discovered_ids and discovered_ids != [resolved_patient_id]:
            raise LesionMinCostFlowError(
                "Supplied patient_id does not match the pair-cost table."
            )

    graph = build_patient_flow_graph(
        table,
        config=config,
        bl_lesion_ids=bl_lesion_ids,
        fu_lesion_ids=fu_lesion_ids,
    )

    flow = solve_patient_flow(graph)

    matches = decode_patient_flow(
        resolved_patient_id,
        graph,
        flow,
    )

    scaled_total_cost = int(
        nx.cost_of_flow(
            graph,
            flow,
            weight="weight",
        )
    )

    total_cost = (
        float(matches["match_cost"].sum())
        if not matches.empty
        else 0.0
    )

    return PatientFlowResult(
        patient_id=resolved_patient_id,
        matches=matches,
        flow=flow,
        scaled_total_cost=scaled_total_cost,
        total_cost=total_cost,
    )


def match_lesions(
    pair_costs: pd.DataFrame,
    config: MatcherConfig = MatcherConfig(),
) -> FlowMatchResult:
    """Run lesion matching independently for every patient in a pair-cost table.

    This convenience function infers BL/FU lesion IDs from candidate rows.  If
    distance gating can remove every candidate for a lesion, call
    ``match_patient_lesions`` with complete ``bl_lesion_ids`` / ``fu_lesion_ids``
    instead so that fully gated lesions remain representable as
    DISAPPEARING/NEW events.
    """
    _validate_config(config)
    table = _validate_patient_pair_costs(pair_costs, config)

    patient_results: dict[str, PatientFlowResult] = {}
    frames: list[pd.DataFrame] = []

    for patient_id in sorted(table["patient_id"].unique()):
        patient_table = table[
            table["patient_id"] == patient_id
        ].copy()

        result = match_patient_lesions(
            patient_table,
            config=config,
            patient_id=str(patient_id),
        )

        patient_results[str(patient_id)] = result
        frames.append(result.matches)

    if frames:
        concat_frames = [
            frame.dropna(axis=1, how="all")
            for frame in frames
            if not frame.empty
        ]
        if concat_frames:
            matches = pd.concat(
                concat_frames,
                ignore_index=True,
                sort=False,
            ).reindex(columns=MATCH_COLUMNS)
        else:
            matches = pd.DataFrame(columns=MATCH_COLUMNS)
    else:
        matches = pd.DataFrame(columns=MATCH_COLUMNS)

    return FlowMatchResult(
        matches=matches,
        patient_results=patient_results,
    )
