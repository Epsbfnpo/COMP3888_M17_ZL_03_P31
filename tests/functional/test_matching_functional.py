from __future__ import annotations

import pandas as pd

from src.lesion_min_cost_flow import MatcherConfig, match_patient_lesions


CONFIG = MatcherConfig(
    disappearing_penalty=1.0,
    new_lesion_penalty=1.0,
    merge_penalty=0.2,
    max_bl_per_fu=2,
    cost_scale=1000,
    cost_column="total_cost",
)


def _empty_pair_costs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "patient_id",
            "bl_lesion_id",
            "fu_lesion_id",
            "total_cost",
        ]
    )


def _pair_costs(patient_id: str, rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "bl_lesion_id": bl_id,
                "fu_lesion_id": fu_id,
                "total_cost": cost,
            }
            for bl_id, fu_id, cost in rows
        ]
    )


def test_one_bl_to_one_fu_is_matched() -> None:
    """
    Scenario:
        One BL lesion -> one FU lesion -> MATCHED.

    Acceptance coverage:
        - Matching completes for a valid cost matrix.
        - The BL lesion receives exactly one final outcome.
        - BL/FU IDs and match type are correct.
    """
    patient_id = "FT_MATCHED"

    pair_costs = _pair_costs(
        patient_id,
        [
            ("BL_L001", "FU_L001", 0.10),
        ],
    )

    result = match_patient_lesions(
        pair_costs,
        config=CONFIG,
        patient_id=patient_id,
        bl_lesion_ids=["BL_L001"],
        fu_lesion_ids=["FU_L001"],
    )

    matches = result.matches

    assert len(matches) == 1

    row = matches.iloc[0]
    assert row["patient_id"] == patient_id
    assert row["bl_lesion_id"] == "BL_L001"
    assert row["fu_lesion_id"] == "FU_L001"
    assert row["match_type"] == "MATCHED"
    assert row["assignment_role"] in {"PRIMARY", "MERGE"}

    # The BL lesion must appear in exactly one decoded outcome.
    assert matches["bl_lesion_id"].eq("BL_L001").sum() == 1


def test_bl_without_fu_correspondence_is_disappearing() -> None:
    """
    Scenario:
        BL lesion without FU correspondence -> DISAPPEARING.

    The complete BL/FU ID lists are passed explicitly so a lesion with no
    candidate edge is still represented by the flow model.
    """
    patient_id = "FT_DISAPPEARING"

    result = match_patient_lesions(
        _empty_pair_costs(),
        config=CONFIG,
        patient_id=patient_id,
        bl_lesion_ids=["BL_L001"],
        fu_lesion_ids=[],
    )

    matches = result.matches

    assert len(matches) == 1

    row = matches.iloc[0]
    assert row["patient_id"] == patient_id
    assert row["bl_lesion_id"] == "BL_L001"
    assert pd.isna(row["fu_lesion_id"])
    assert row["match_type"] == "DISAPPEARING"
    assert row["assignment_role"] == "DISAPPEAR"

    # Every BL lesion must receive exactly one final outcome.
    assert matches["bl_lesion_id"].eq("BL_L001").sum() == 1


def test_fu_without_bl_correspondence_is_new() -> None:
    """
    Scenario:
        FU lesion without BL correspondence -> NEW.
    """
    patient_id = "FT_NEW"

    result = match_patient_lesions(
        _empty_pair_costs(),
        config=CONFIG,
        patient_id=patient_id,
        bl_lesion_ids=[],
        fu_lesion_ids=["FU_L001"],
    )

    matches = result.matches

    assert len(matches) == 1

    row = matches.iloc[0]
    assert row["patient_id"] == patient_id
    assert pd.isna(row["bl_lesion_id"])
    assert row["fu_lesion_id"] == "FU_L001"
    assert row["match_type"] == "NEW"
    assert row["assignment_role"] == "NEW"


def test_multiple_bl_to_one_fu_is_merging() -> None:
    """
    Scenario:
        Multiple BL lesions -> one FU lesion -> MERGING.

    Both BL->FU pair costs are intentionally cheaper than assigning a
    DISAPPEARING event, even after the merge penalty is added. This makes the
    expected minimum-cost solution unambiguous.
    """
    patient_id = "FT_MERGING"

    pair_costs = _pair_costs(
        patient_id,
        [
            ("BL_L001", "FU_L001", 0.10),
            ("BL_L002", "FU_L001", 0.12),
        ],
    )

    result = match_patient_lesions(
        pair_costs,
        config=CONFIG,
        patient_id=patient_id,
        bl_lesion_ids=["BL_L001", "BL_L002"],
        fu_lesion_ids=["FU_L001"],
    )

    matches = result.matches

    # Two BL lesions participate in one many-to-one merge event.
    assert len(matches) == 2
    assert set(matches["bl_lesion_id"]) == {"BL_L001", "BL_L002"}
    assert set(matches["fu_lesion_id"]) == {"FU_L001"}
    assert set(matches["match_type"]) == {"MERGING"}

    # Each BL lesion still receives exactly one final outcome.
    assert matches["bl_lesion_id"].value_counts().to_dict() == {
        "BL_L001": 1,
        "BL_L002": 1,
    }

    # The decoder should expose one shared merge-event identity.
    event_ids = matches["event_id"].dropna().unique()
    assert len(event_ids) == 1
    assert event_ids[0] == f"MERGE::{patient_id}::FU_L001"

    # Internally one BL occupies the primary FU slot and one occupies the
    # additional merge slot, but both must be reported biologically as MERGING.
    assert set(matches["assignment_role"]) == {"PRIMARY", "MERGE"}
