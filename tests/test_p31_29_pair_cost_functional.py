"""P31-29 functional tests: BL/FU candidate generation and pair costs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lesion_min_cost_flow import MatcherConfig, match_patient_cost_matrix
from src.lesion_pair_costs import (
    PairCostConfig,
    export_cost_matrices,
    generate_lesion_pair_costs,
)


def _feature(
    timepoint: str,
    lesion_id: str,
    xyz: tuple[float, float, float],
    volume_ml: float,
    *,
    suvmax: float | None = None,
    pet_mean: float | None = None,
    pet_units: str | None = None,
) -> dict[str, object]:
    return {
        "patient_id": "p1",
        "timepoint": timepoint,
        "lesion_id": lesion_id,
        "centroid_x_mm": xyz[0],
        "centroid_y_mm": xyz[1],
        "centroid_z_mm": xyz[2],
        "volume_ml": volume_ml,
        "coordinate_space": "FU_RAS_mm",
        "suvmax": suvmax,
        "pet_mean": pet_mean,
        "pet_units": pet_units,
    }


def test_p31_29_all_valid_bl_fu_combinations_are_generated() -> None:
    features = pd.DataFrame(
        [
            _feature("BL", "BL_1", (0, 0, 0), 1.0),
            _feature("BL", "BL_2", (10, 0, 0), 2.0),
            _feature("FU", "FU_1", (1, 0, 0), 1.1),
            _feature("FU", "FU_2", (11, 0, 0), 2.2),
            _feature("FU", "FU_3", (30, 0, 0), 3.0),
        ]
    )

    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(pet_weight=0, pet_feature="none"),
    )

    assert len(result.pair_costs) == 2 * 3
    assert set(zip(result.pair_costs.bl_lesion_id, result.pair_costs.fu_lesion_id)) == {
        ("BL_1", "FU_1"), ("BL_1", "FU_2"), ("BL_1", "FU_3"),
        ("BL_2", "FU_1"), ("BL_2", "FU_2"), ("BL_2", "FU_3"),
    }
    assert result.matrices["p1"].shape == (2, 3)


def test_p31_29_distance_and_size_costs_match_known_values() -> None:
    features = pd.DataFrame(
        [
            _feature("BL", "BL_1", (0, 0, 0), 2.0),
            _feature("FU", "FU_1", (3, 4, 12), 8.0),
        ]
    )

    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(
            distance_weight=1.0,
            size_weight=1.0,
            pet_weight=0.0,
            pet_feature="none",
            distance_scale_mm=10.0,
        ),
    )
    row = result.pair_costs.iloc[0]

    assert row["distance_mm"] == pytest.approx(13.0)
    assert row["distance_cost"] == pytest.approx(1.3)
    assert row["size_difference_fraction"] == pytest.approx(0.75)
    assert row["size_cost"] == pytest.approx(0.75)
    assert row["total_cost"] == pytest.approx((1.3 + 0.75) / 2.0)


def test_p31_29_pet_cost_is_used_only_when_information_is_comparable() -> None:
    features = pd.DataFrame(
        [
            _feature("BL", "BL_1", (0, 0, 0), 1.0, suvmax=8.0),
            _feature("FU", "FU_SUV", (1, 0, 0), 1.0, suvmax=10.0),
            _feature("FU", "FU_NO_PET", (2, 0, 0), 1.0, suvmax=None),
        ]
    )

    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(
            distance_weight=1.0,
            size_weight=0.0,
            pet_weight=1.0,
            pet_feature="suvmax",
            distance_scale_mm=10.0,
        ),
    ).pair_costs.set_index("fu_lesion_id")

    assert bool(result.loc["FU_SUV", "pet_used"]) is True
    assert result.loc["FU_SUV", "pet_difference_fraction"] == pytest.approx(0.2)
    assert bool(result.loc["FU_NO_PET", "pet_used"]) is False
    assert pd.isna(result.loc["FU_NO_PET", "pet_cost"])

    # Raw PET values are not comparable when the units disagree.
    raw = pd.DataFrame(
        [
            _feature("BL", "BL_1", (0, 0, 0), 1.0, pet_mean=4.0, pet_units="Bq/mL"),
            _feature("FU", "FU_1", (1, 0, 0), 1.0, pet_mean=5.0, pet_units="counts"),
        ]
    )
    raw_result = generate_lesion_pair_costs(
        raw,
        PairCostConfig(size_weight=0, pet_weight=1, pet_feature="pet_mean"),
    ).pair_costs.iloc[0]
    assert bool(raw_result["pet_used"]) is False
    assert pd.isna(raw_result["pet_cost"])


def test_p31_29_distance_gate_removes_far_candidate_and_marks_matrix_inf() -> None:
    features = pd.DataFrame(
        [
            _feature("BL", "BL_1", (0, 0, 0), 1.0),
            _feature("FU", "FU_NEAR", (5, 0, 0), 1.0),
            _feature("FU", "FU_FAR", (100, 0, 0), 1.0),
        ]
    )
    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(pet_weight=0, pet_feature="none", max_distance_mm=20.0),
    )

    assert result.pair_costs["fu_lesion_id"].tolist() == ["FU_NEAR"]
    matrix = result.matrices["p1"]
    assert np.isfinite(matrix.values[0, matrix.fu_lesion_ids.index("FU_NEAR")])
    assert np.isinf(matrix.values[0, matrix.fu_lesion_ids.index("FU_FAR")])


def test_p31_29_labelled_matrix_exports_and_is_accepted_by_matcher(tmp_path: Path) -> None:
    features = pd.DataFrame(
        [
            _feature("BL", "BL_1", (0, 0, 0), 1.0),
            _feature("BL", "BL_2", (20, 0, 0), 1.0),
            _feature("FU", "FU_1", (1, 0, 0), 1.0),
            _feature("FU", "FU_2", (21, 0, 0), 1.0),
        ]
    )
    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(pet_weight=0, pet_feature="none", distance_scale_mm=50.0),
    )

    paths = export_cost_matrices(result.matrices, tmp_path)
    assert len(paths) == 1
    matrix_path = paths[0]
    exported = pd.read_csv(matrix_path)

    assert exported.columns.tolist() == ["bl_lesion_id", "FU_1", "FU_2"]
    assert exported["bl_lesion_id"].tolist() == ["BL_1", "BL_2"]

    # Expected result from the Jira story: the exported matrix is directly
    # consumable by the matching algorithm without reshaping or relabelling.
    matched = match_patient_cost_matrix(
        matrix_path,
        MatcherConfig(
            disappearing_penalty=2.0,
            new_lesion_penalty=2.0,
            max_bl_per_fu=1,
        ),
        patient_id="p1",
    )
    assert not matched.matches.empty
    assert set(matched.matches["match_type"]).issubset(
        {"MATCHED", "NEW", "DISAPPEARING", "MERGING"}
    )
