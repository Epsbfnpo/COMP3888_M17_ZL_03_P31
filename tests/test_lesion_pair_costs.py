from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.lesion_pair_costs import (
    LesionPairCostError,
    PairCostConfig,
    generate_lesion_pair_costs,
)


def feature_row(
    patient_id: str,
    timepoint: str,
    lesion_id: str,
    xyz: tuple[float, float, float],
    volume_ml: float,
    *,
    suvmax: float | None = None,
    pet_mean: float | None = None,
    pet_units: str | None = None,
    coordinate_space: str = "FU_RAS_mm",
) -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "timepoint": timepoint,
        "lesion_id": lesion_id,
        "centroid_x_mm": xyz[0],
        "centroid_y_mm": xyz[1],
        "centroid_z_mm": xyz[2],
        "volume_ml": volume_ml,
        "coordinate_space": coordinate_space,
        "suvmax": suvmax,
        "pet_mean": pet_mean,
        "pet_units": pet_units,
    }


def test_generates_candidates_only_within_each_patient_and_builds_matrices() -> None:
    features = pd.DataFrame(
        [
            feature_row("p1", "BL", "p1_BL_L001", (0, 0, 0), 1.0),
            feature_row("p1", "BL", "p1_BL_L002", (10, 0, 0), 2.0),
            feature_row("p1", "FU", "p1_FU_L001", (1, 0, 0), 1.1),
            feature_row("p1", "FU", "p1_FU_L002", (11, 0, 0), 2.2),
            feature_row("p2", "BL", "p2_BL_L001", (100, 0, 0), 3.0),
            feature_row("p2", "FU", "p2_FU_L001", (102, 0, 0), 3.0),
        ]
    )

    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(pet_weight=0, pet_feature="none"),
    )

    assert len(result.pair_costs) == 5  # p1: 2x2, p2: 1x1
    assert set(result.pair_costs["patient_id"]) == {"p1", "p2"}
    assert result.matrices["p1"].shape == (2, 2)
    assert result.matrices["p2"].shape == (1, 1)
    assert not (
        (result.pair_costs["bl_lesion_id"].str.startswith("p1"))
        & (result.pair_costs["fu_lesion_id"].str.startswith("p2"))
    ).any()


def test_spatial_distance_uses_aligned_world_mm_coordinates() -> None:
    features = pd.DataFrame(
        [
            feature_row("p", "BL", "bl", (0, 0, 0), 1.0),
            feature_row("p", "FU", "fu", (3, 4, 12), 1.0),
        ]
    )
    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(
            distance_weight=1,
            size_weight=0,
            pet_weight=0,
            pet_feature="none",
            distance_scale_mm=10,
        ),
    )
    row = result.pair_costs.iloc[0]
    assert row["distance_mm"] == pytest.approx(13.0)
    assert row["distance_cost"] == pytest.approx(1.3)
    assert row["total_cost"] == pytest.approx(1.3)


def test_size_difference_contributes_to_cost() -> None:
    features = pd.DataFrame(
        [
            feature_row("p", "BL", "bl", (0, 0, 0), 2.0),
            feature_row("p", "FU", "fu_same", (5, 0, 0), 2.0),
            feature_row("p", "FU", "fu_big", (5, 0, 0), 8.0),
        ]
    )
    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(
            distance_weight=1,
            size_weight=1,
            pet_weight=0,
            pet_feature="none",
            distance_scale_mm=10,
        ),
    )
    rows = result.pair_costs.set_index("fu_lesion_id")
    assert rows.loc["fu_same", "size_difference_fraction"] == pytest.approx(0.0)
    assert rows.loc["fu_big", "size_difference_fraction"] == pytest.approx(0.75)
    assert rows.loc["fu_same", "total_cost"] < rows.loc["fu_big", "total_cost"]


def test_pet_difference_is_optional_and_only_used_when_comparable() -> None:
    features = pd.DataFrame(
        [
            feature_row("p", "BL", "bl", (0, 0, 0), 1.0, suvmax=8.0),
            feature_row("p", "FU", "fu_pet", (1, 0, 0), 1.0, suvmax=10.0),
            feature_row("p", "FU", "fu_missing", (1, 0, 0), 1.0, suvmax=None),
        ]
    )
    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(
            distance_weight=1,
            size_weight=0,
            pet_weight=1,
            pet_feature="suvmax",
            distance_scale_mm=10,
        ),
    )
    rows = result.pair_costs.set_index("fu_lesion_id")
    assert bool(rows.loc["fu_pet", "pet_used"]) is True
    assert rows.loc["fu_pet", "pet_difference_fraction"] == pytest.approx(0.2)
    assert bool(rows.loc["fu_missing", "pet_used"]) is False
    assert pd.isna(rows.loc["fu_missing", "pet_cost"])
    assert rows.loc["fu_missing", "active_weight_sum"] == pytest.approx(1.0)


def test_raw_pet_features_require_matching_units() -> None:
    features = pd.DataFrame(
        [
            feature_row("p", "BL", "bl", (0, 0, 0), 1.0, pet_mean=4, pet_units="Bq/mL"),
            feature_row("p", "FU", "fu", (1, 0, 0), 1.0, pet_mean=5, pet_units="counts"),
        ]
    )
    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(size_weight=0, pet_weight=1, pet_feature="pet_mean"),
    )
    row = result.pair_costs.iloc[0]
    assert bool(row["pet_used"]) is False
    assert pd.isna(row["pet_cost"])


def test_distance_gate_removes_far_candidate_and_matrix_marks_inf() -> None:
    features = pd.DataFrame(
        [
            feature_row("p", "BL", "bl", (0, 0, 0), 1.0),
            feature_row("p", "FU", "near", (5, 0, 0), 1.0),
            feature_row("p", "FU", "far", (100, 0, 0), 1.0),
        ]
    )
    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(
            pet_weight=0,
            pet_feature="none",
            max_distance_mm=20,
        ),
    )
    assert list(result.pair_costs["fu_lesion_id"]) == ["near"]
    matrix = result.matrices["p"]
    assert matrix.shape == (1, 2)
    far_index = matrix.fu_lesion_ids.index("far")
    near_index = matrix.fu_lesion_ids.index("near")
    assert np.isinf(matrix.values[0, far_index])
    assert np.isfinite(matrix.values[0, near_index])


def test_many_to_one_candidates_are_preserved_for_future_merge_matching() -> None:
    features = pd.DataFrame(
        [
            feature_row("p", "BL", "bl1", (0, 0, 0), 1.0),
            feature_row("p", "BL", "bl2", (2, 0, 0), 1.0),
            feature_row("p", "FU", "fu1", (1, 0, 0), 2.0),
        ]
    )
    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(pet_weight=0, pet_feature="none"),
    )
    assert set(zip(result.pair_costs.bl_lesion_id, result.pair_costs.fu_lesion_id)) == {
        ("bl1", "fu1"),
        ("bl2", "fu1"),
    }


def test_rejects_native_or_unaligned_coordinate_space() -> None:
    features = pd.DataFrame(
        [
            feature_row("p", "BL", "bl", (0, 0, 0), 1.0, coordinate_space="BL_RAS_mm"),
            feature_row("p", "FU", "fu", (0, 0, 0), 1.0),
        ]
    )
    with pytest.raises(LesionPairCostError, match="FU_RAS_mm"):
        generate_lesion_pair_costs(features)
