from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from src.lesion_min_cost_flow import (
    LesionMinCostFlowError,
    MatcherConfig,
    build_patient_flow_graph,
    decode_patient_flow,
    load_patient_cost_matrix,
    match_lesions,
    match_patient_cost_matrix,
    match_patient_lesions,
    solve_patient_flow,
)


def make_pair_costs(
    rows: list[tuple[str, str, str, float]],
) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "patient_id",
            "bl_lesion_id",
            "fu_lesion_id",
            "total_cost",
        ],
    )


def current_patient_pair_costs() -> pd.DataFrame:
    """Real pair-cost pattern previously produced for patient 0a09c8844b."""
    return make_pair_costs(
        [
            ("0a09c8844b", "0a09c8844b_BL_L001", "0a09c8844b_FU_L002", 0.8185685435970055),
            ("0a09c8844b", "0a09c8844b_BL_L001", "0a09c8844b_FU_L001", 4.107460035272056),
            ("0a09c8844b", "0a09c8844b_BL_L002", "0a09c8844b_FU_L002", 0.7672627587632945),
            ("0a09c8844b", "0a09c8844b_BL_L002", "0a09c8844b_FU_L001", 4.203405193082504),
            ("0a09c8844b", "0a09c8844b_BL_L003", "0a09c8844b_FU_L001", 0.1267919876185001),
            ("0a09c8844b", "0a09c8844b_BL_L003", "0a09c8844b_FU_L002", 4.526919868215774),
            ("0a09c8844b", "0a09c8844b_BL_L004", "0a09c8844b_FU_L002", 0.4917211476443426),
            ("0a09c8844b", "0a09c8844b_BL_L004", "0a09c8844b_FU_L001", 4.685437987141547),
        ]
    )


def lesion_outcomes(matches: pd.DataFrame) -> set[tuple[object, object, str]]:
    return {
        (
            None if pd.isna(row.bl_lesion_id) else row.bl_lesion_id,
            None if pd.isna(row.fu_lesion_id) else row.fu_lesion_id,
            row.match_type,
        )
        for row in matches.itertuples()
    }


# =============================================================================
# PART 1 — NETWORK CONSTRUCTION TESTS
# =============================================================================


def test_one_to_one_graph_has_no_merge_nodes() -> None:
    pair_costs = make_pair_costs([("p", "BL1", "FU1", 0.25)])
    graph = build_patient_flow_graph(
        pair_costs,
        MatcherConfig(max_bl_per_fu=1),
    )

    node_types = {data.get("node_type") for _, data in graph.nodes(data=True)}
    assert "MERGE_EVENT" not in node_types
    assert "FU_MERGE_SLOT" not in node_types
    assert ("FU_PRIMARY", "FU1") in graph

    total_demand = sum(data.get("demand", 0) for _, data in graph.nodes(data=True))
    assert total_demand == 0


def test_merge_graph_creates_explicit_event_and_slots() -> None:
    pair_costs = make_pair_costs([("p", "BL1", "FU1", 0.40)])
    config = MatcherConfig(
        max_bl_per_fu=3,
        merge_penalty=0.2,
        cost_scale=1000,
    )
    graph = build_patient_flow_graph(pair_costs, config)

    merge_event = ("MERGE_EVENT", "FU1")
    slot1 = ("FU_MERGE_SLOT", "FU1", 1)
    slot2 = ("FU_MERGE_SLOT", "FU1", 2)

    assert merge_event in graph
    assert slot1 in graph
    assert slot2 in graph
    assert graph.nodes[merge_event]["event_type"] == "MERGING"

    # Pair similarity is charged before entering the event.
    assert graph[("BL", "BL1")][merge_event]["edge_type"] == "MERGE_CANDIDATE"
    assert graph[("BL", "BL1")][merge_event]["weight"] == 400

    # Merge-specific cost is a separate event edge.
    assert graph[merge_event][slot1]["edge_type"] == "MERGE_EVENT"
    assert graph[merge_event][slot1]["weight"] == 200
    assert graph[merge_event][slot1]["event_cost"] == pytest.approx(0.2)

    # Optional merge slots can remain unused at zero cost.
    new_node = ("SPECIAL", "NEW")
    assert graph[new_node][slot1]["edge_type"] == "UNUSED_MERGE_SLOT"
    assert graph[new_node][slot1]["weight"] == 0

    total_demand = sum(data.get("demand", 0) for _, data in graph.nodes(data=True))
    assert total_demand == 0


def test_max_bl_per_fu_controls_number_of_merge_slots() -> None:
    pair_costs = make_pair_costs([("p", "BL1", "FU1", 0.20)])
    graph = build_patient_flow_graph(
        pair_costs,
        MatcherConfig(max_bl_per_fu=4),
    )

    slots = [
        node
        for node, data in graph.nodes(data=True)
        if data.get("node_type") == "FU_MERGE_SLOT"
    ]
    assert len(slots) == 3


# =============================================================================
# PART 2 — NETWORKX SOLVER TESTS
# =============================================================================


def test_networkx_solver_returns_feasible_flow() -> None:
    pair_costs = make_pair_costs(
        [
            ("p", "BL1", "FU1", 0.2),
            ("p", "BL2", "FU1", 0.4),
        ]
    )
    graph = build_patient_flow_graph(
        pair_costs,
        MatcherConfig(max_bl_per_fu=2, merge_penalty=0.2),
    )
    flow = solve_patient_flow(graph)

    # NetworkX's own flow-cost function must accept the returned solution.
    assert nx.cost_of_flow(graph, flow, weight="weight") >= 0

    # Every BL supplies exactly one unit into one selected outcome.
    for bl_id in ("BL1", "BL2"):
        assert sum(flow[("BL", bl_id)].values()) == 1


def test_duplicate_candidate_edge_is_rejected_before_solver() -> None:
    pair_costs = make_pair_costs(
        [
            ("p", "BL1", "FU1", 0.1),
            ("p", "BL1", "FU1", 0.2),
        ]
    )
    with pytest.raises(LesionMinCostFlowError, match="Duplicate BL/FU candidate edge"):
        build_patient_flow_graph(pair_costs)


# =============================================================================
# PART 3 — DECODING / EVENT TESTS
# =============================================================================


def test_baseline_max_one_reproduces_original_one_to_one_result() -> None:
    result = match_patient_lesions(
        current_patient_pair_costs(),
        MatcherConfig(
            disappearing_penalty=1.0,
            new_lesion_penalty=1.0,
            merge_penalty=0.2,
            max_bl_per_fu=1,
            cost_scale=1000,
        ),
    )

    assert lesion_outcomes(result.matches) == {
        ("0a09c8844b_BL_L001", None, "DISAPPEARING"),
        ("0a09c8844b_BL_L002", None, "DISAPPEARING"),
        ("0a09c8844b_BL_L003", "0a09c8844b_FU_L001", "MATCHED"),
        ("0a09c8844b_BL_L004", "0a09c8844b_FU_L002", "MATCHED"),
    }


def test_slot_merging_occurs_as_explicit_event() -> None:
    result = match_patient_lesions(
        current_patient_pair_costs(),
        MatcherConfig(
            disappearing_penalty=1.0,
            new_lesion_penalty=1.0,
            merge_penalty=0.2,
            max_bl_per_fu=3,
            cost_scale=1000,
        ),
    )

    matches = result.matches
    assert lesion_outcomes(matches) == {
        ("0a09c8844b_BL_L001", None, "DISAPPEARING"),
        ("0a09c8844b_BL_L002", "0a09c8844b_FU_L002", "MERGING"),
        ("0a09c8844b_BL_L003", "0a09c8844b_FU_L001", "MATCHED"),
        ("0a09c8844b_BL_L004", "0a09c8844b_FU_L002", "MERGING"),
    }

    merge_rows = matches[matches["fu_lesion_id"] == "0a09c8844b_FU_L002"]
    assert len(merge_rows) == 2
    assert set(merge_rows["match_type"]) == {"MERGING"}
    assert merge_rows["event_id"].nunique() == 1
    assert merge_rows["event_id"].iloc[0] == (
        "MERGE::0a09c8844b::0a09c8844b_FU_L002"
    )
    assert set(merge_rows["assignment_role"]) == {"PRIMARY", "MERGE"}

    merge_assignment = merge_rows[merge_rows["assignment_role"] == "MERGE"].iloc[0]
    assert merge_assignment["event_cost"] == pytest.approx(0.2)
    assert merge_assignment["match_cost"] == pytest.approx(
        merge_assignment["pair_cost"] + 0.2
    )


def test_higher_merge_penalty_prevents_current_patient_merge() -> None:
    result = match_patient_lesions(
        current_patient_pair_costs(),
        MatcherConfig(
            disappearing_penalty=1.0,
            new_lesion_penalty=1.0,
            merge_penalty=0.25,
            max_bl_per_fu=3,
            cost_scale=1000,
        ),
    )

    assert "MERGING" not in set(result.matches["match_type"])
    assert lesion_outcomes(result.matches) == {
        ("0a09c8844b_BL_L001", None, "DISAPPEARING"),
        ("0a09c8844b_BL_L002", None, "DISAPPEARING"),
        ("0a09c8844b_BL_L003", "0a09c8844b_FU_L001", "MATCHED"),
        ("0a09c8844b_BL_L004", "0a09c8844b_FU_L002", "MATCHED"),
    }


def test_expensive_pair_decodes_to_disappearing_and_new() -> None:
    pair_costs = make_pair_costs([("p", "BL1", "FU1", 3.0)])
    result = match_patient_lesions(
        pair_costs,
        MatcherConfig(
            disappearing_penalty=1.0,
            new_lesion_penalty=1.0,
            max_bl_per_fu=3,
        ),
    )

    assert lesion_outcomes(result.matches) == {
        ("BL1", None, "DISAPPEARING"),
        (None, "FU1", "NEW"),
    }


def test_fully_gated_bl_is_preserved_with_complete_ids() -> None:
    pair_costs = make_pair_costs([("p", "BL1", "FU1", 0.2)])
    result = match_patient_lesions(
        pair_costs,
        MatcherConfig(max_bl_per_fu=2),
        patient_id="p",
        bl_lesion_ids=["BL1", "BL2"],
        fu_lesion_ids=["FU1"],
    )

    assert lesion_outcomes(result.matches) == {
        ("BL1", "FU1", "MATCHED"),
        ("BL2", None, "DISAPPEARING"),
    }


def test_multiple_patients_are_solved_independently_in_baseline_mode() -> None:
    pair_costs = make_pair_costs(
        [
            ("patient_A", "BL1", "FU1", 0.1),
            ("patient_B", "BL1", "FU1", 3.0),
        ]
    )
    result = match_lesions(
        pair_costs,
        MatcherConfig(
            disappearing_penalty=1.0,
            new_lesion_penalty=1.0,
            max_bl_per_fu=1,
        ),
    )

    outcomes = {
        (
            row.patient_id,
            None if pd.isna(row.bl_lesion_id) else row.bl_lesion_id,
            None if pd.isna(row.fu_lesion_id) else row.fu_lesion_id,
            row.match_type,
        )
        for row in result.matches.itertuples()
    }

    assert ("patient_A", "BL1", "FU1", "MATCHED") in outcomes
    assert ("patient_B", "BL1", None, "DISAPPEARING") in outcomes
    assert ("patient_B", None, "FU1", "NEW") in outcomes


def test_decode_can_be_called_separately_after_build_and_solve() -> None:
    pair_costs = make_pair_costs(
        [
            ("p", "BL1", "FU1", 0.1),
            ("p", "BL2", "FU1", 0.2),
        ]
    )
    config = MatcherConfig(max_bl_per_fu=2, merge_penalty=0.1)

    graph = build_patient_flow_graph(pair_costs, config)
    flow = solve_patient_flow(graph)
    decoded = decode_patient_flow("p", graph, flow)

    assert len(decoded) == 2
    assert set(decoded["match_type"]) == {"MERGING"}
    assert decoded["event_id"].nunique() == 1


def test_max_bl_per_fu_limits_selected_merge_assignments() -> None:
    """With one merge slot, FU_L002 can receive at most two BL lesions total."""
    result = match_patient_lesions(
        current_patient_pair_costs(),
        MatcherConfig(
            disappearing_penalty=1.0,
            new_lesion_penalty=1.0,
            merge_penalty=0.1,
            max_bl_per_fu=2,
            cost_scale=1000,
        ),
    )

    matches = result.matches
    fu2_rows = matches[
        matches["fu_lesion_id"] == "0a09c8844b_FU_L002"
    ]

    # max_bl_per_fu=2 means: one PRIMARY + one MERGE slot.
    assert len(fu2_rows) == 2
    assert set(fu2_rows["bl_lesion_id"]) == {
        "0a09c8844b_BL_L002",
        "0a09c8844b_BL_L004",
    }
    assert set(fu2_rows["match_type"]) == {"MERGING"}
    assert set(fu2_rows["assignment_role"]) == {"PRIMARY", "MERGE"}
    assert fu2_rows["event_id"].nunique() == 1

    # BL_L001 would also merge when max_bl_per_fu=3 at penalty=0.1,
    # but the single merge slot is already occupied by the cheaper BL_L002.
    bl1 = matches[matches["bl_lesion_id"] == "0a09c8844b_BL_L001"].iloc[0]
    assert bl1["match_type"] == "DISAPPEARING"
    assert pd.isna(bl1["fu_lesion_id"])


# =============================================================================
# DIRECT COST-MATRIX LOADER TESTS
# =============================================================================


def test_cost_matrix_loader_converts_labelled_matrix_to_pair_costs(tmp_path) -> None:
    matrix_path = tmp_path / "patient_A_cost_matrix.csv"
    pd.DataFrame(
        {
            "bl_lesion_id": ["BL1", "BL2"],
            "FU1": [0.10, 0.30],
            "FU2": [0.20, 0.40],
        }
    ).to_csv(matrix_path, index=False)

    loaded = load_patient_cost_matrix(matrix_path)

    assert loaded.patient_id == "patient_A"
    assert loaded.bl_lesion_ids == ("BL1", "BL2")
    assert loaded.fu_lesion_ids == ("FU1", "FU2")
    assert len(loaded.pair_costs) == 4
    assert set(loaded.pair_costs.columns) == {
        "patient_id",
        "bl_lesion_id",
        "fu_lesion_id",
        "total_cost",
    }

    row = loaded.pair_costs[
        (loaded.pair_costs["bl_lesion_id"] == "BL2")
        & (loaded.pair_costs["fu_lesion_id"] == "FU1")
    ].iloc[0]
    assert row["total_cost"] == pytest.approx(0.30)


def test_cost_matrix_loader_treats_inf_as_gated_edge_but_preserves_ids(tmp_path) -> None:
    matrix_path = tmp_path / "patient_A_cost_matrix.csv"
    pd.DataFrame(
        {
            "bl_lesion_id": ["BL1", "BL2"],
            "FU1": [0.10, float("inf")],
        }
    ).to_csv(matrix_path, index=False)

    loaded = load_patient_cost_matrix(matrix_path)

    assert loaded.bl_lesion_ids == ("BL1", "BL2")
    assert loaded.fu_lesion_ids == ("FU1",)
    assert len(loaded.pair_costs) == 1

    result = match_patient_cost_matrix(
        matrix_path,
        MatcherConfig(
            disappearing_penalty=1.0,
            new_lesion_penalty=1.0,
            max_bl_per_fu=1,
        ),
    )

    assert lesion_outcomes(result.matches) == {
        ("BL1", "FU1", "MATCHED"),
        ("BL2", None, "DISAPPEARING"),
    }


def test_direct_matrix_matching_reproduces_current_patient_baseline(tmp_path) -> None:
    matrix_path = tmp_path / "0a09c8844b_cost_matrix.csv"
    pd.DataFrame(
        {
            "bl_lesion_id": [
                "0a09c8844b_BL_L001",
                "0a09c8844b_BL_L002",
                "0a09c8844b_BL_L003",
                "0a09c8844b_BL_L004",
            ],
            "0a09c8844b_FU_L001": [
                4.107460035272056,
                4.203405193082504,
                0.1267919876185001,
                4.685437987141547,
            ],
            "0a09c8844b_FU_L002": [
                0.8185685435970055,
                0.7672627587632945,
                4.526919868215774,
                0.4917211476443426,
            ],
        }
    ).to_csv(matrix_path, index=False)

    result = match_patient_cost_matrix(
        matrix_path,
        MatcherConfig(
            disappearing_penalty=1.0,
            new_lesion_penalty=1.0,
            max_bl_per_fu=1,
            cost_scale=1000,
        ),
    )

    assert lesion_outcomes(result.matches) == {
        ("0a09c8844b_BL_L001", None, "DISAPPEARING"),
        ("0a09c8844b_BL_L002", None, "DISAPPEARING"),
        ("0a09c8844b_BL_L003", "0a09c8844b_FU_L001", "MATCHED"),
        ("0a09c8844b_BL_L004", "0a09c8844b_FU_L002", "MATCHED"),
    }


def test_direct_matrix_matching_supports_slot_merging(tmp_path) -> None:
    matrix_path = tmp_path / "0a09c8844b_cost_matrix.csv"
    pd.DataFrame(
        {
            "bl_lesion_id": [
                "0a09c8844b_BL_L001",
                "0a09c8844b_BL_L002",
                "0a09c8844b_BL_L003",
                "0a09c8844b_BL_L004",
            ],
            "0a09c8844b_FU_L001": [4.107460, 4.203405, 0.126792, 4.685438],
            "0a09c8844b_FU_L002": [0.818569, 0.767263, 4.526920, 0.491721],
        }
    ).to_csv(matrix_path, index=False)

    result = match_patient_cost_matrix(
        matrix_path,
        MatcherConfig(
            disappearing_penalty=1.0,
            new_lesion_penalty=1.0,
            merge_penalty=0.2,
            max_bl_per_fu=3,
            cost_scale=1000,
        ),
    )

    assert lesion_outcomes(result.matches) == {
        ("0a09c8844b_BL_L001", None, "DISAPPEARING"),
        ("0a09c8844b_BL_L002", "0a09c8844b_FU_L002", "MERGING"),
        ("0a09c8844b_BL_L003", "0a09c8844b_FU_L001", "MATCHED"),
        ("0a09c8844b_BL_L004", "0a09c8844b_FU_L002", "MERGING"),
    }


def test_cost_matrix_loader_rejects_nan_cell(tmp_path) -> None:
    matrix_path = tmp_path / "patient_A_cost_matrix.csv"
    pd.DataFrame(
        {
            "bl_lesion_id": ["BL1"],
            "FU1": [float("nan")],
        }
    ).to_csv(matrix_path, index=False)

    with pytest.raises(LesionMinCostFlowError, match="must not be NaN"):
        load_patient_cost_matrix(matrix_path)
