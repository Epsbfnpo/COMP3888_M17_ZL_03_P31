from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.lesion_pair_costs import (
    PAIR_COST_COLUMNS,
    LesionPairCostError,
    PairCostConfig,
    export_cost_matrices,
    export_pair_costs,
    generate_lesion_pair_costs,
    load_aligned_feature_csvs,
)
def lesion(
    patient: str,
    timepoint: str,
    lesion_id: str,
    xyz: tuple[float, float, float],
    volume: float = 1.0,
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": patient,
        "timepoint": timepoint,
        "lesion_id": lesion_id,
        "centroid_x_mm": xyz[0],
        "centroid_y_mm": xyz[1],
        "centroid_z_mm": xyz[2],
        "volume_ml": volume,
        "coordinate_space": "FU_RAS_mm",
    }
    row.update(extra)
    return row

def test_calculates_weighted_cost_from_distance_size_and_pet() -> None:
    features = pd.DataFrame(
        [
            lesion("P01", "BL", "BL1", (0, 0, 0), 2.0, suvmax=8.0),
            lesion("P01", "FU", "FU1", (3, 4, 0), 4.0, suvmax=10.0),
        ]
    )
    config = PairCostConfig(
        distance_weight=2.0,
        size_weight=1.0,
        pet_weight=1.0,
        distance_scale_mm=10.0,
        pet_feature="suvmax",
    )

    row = generate_lesion_pair_costs(features, config).pair_costs.iloc[0]
    assert row["distance_mm"] == pytest.approx(5.0)
    assert row["size_difference_fraction"] == pytest.approx(0.5)
    assert row["pet_difference_fraction"] == pytest.approx(0.2)
    #according to (2 * 0.5 + 1 * 0.5 + 1 * 0.2) / 4
    assert row["total_cost"] == pytest.approx(0.425)

def test_missing_pet_value_removes_pet_weight_for_that_pair() -> None:
    features = pd.DataFrame(
        [
            lesion("P01", "BL", "BL1", (0, 0, 0), suvmax=8.0),
            lesion("P01", "FU", "FU1", (5, 0, 0), suvmax=np.nan),
        ]
    )
    config = PairCostConfig(
        distance_weight=1.0,
        size_weight=0.0,
        pet_weight=3.0,
        distance_scale_mm=10.0,
        pet_feature="suvmax",
    )
    row = generate_lesion_pair_costs(features, config).pair_costs.iloc[0]

    assert bool(row["pet_used"]) is False
    assert pd.isna(row["pet_cost"])
    assert row["active_weight_sum"] == pytest.approx(1.0)
    assert row["total_cost"] == pytest.approx(0.5)

def test_raw_pet_values_are_used_only_when_units_match() -> None:
    features = pd.DataFrame(
        [
            lesion("P01", "BL", "BL1", (0, 0, 0), pet_mean=4.0, pet_units="Bq/mL"),
            lesion("P01", "FU", "same", (0, 0, 0), pet_mean=5.0, pet_units="bq/ml"),
            lesion("P01", "FU", "different", (0, 0, 0), pet_mean=5.0, pet_units="counts"),
        ]
    )
    config = PairCostConfig(
        distance_weight=1.0,
        size_weight=0.0,
        pet_weight=1.0,
        pet_feature="pet_mean",
    )

    rows = generate_lesion_pair_costs(features, config).pair_costs.set_index("fu_lesion_id")

    assert bool(rows.loc["same", "pet_used"]) is True
    assert rows.loc["same", "pet_cost"] == pytest.approx(0.2)
    assert bool(rows.loc["different", "pet_used"]) is False

def test_distance_gate_keeps_matrix_shape_and_marks_removed_pair_as_infinity() -> None:
    features = pd.DataFrame(
        [
            lesion("P01", "BL", "BL1", (0, 0, 0)),
            lesion("P01", "FU", "near", (10, 0, 0)),
            lesion("P01", "FU", "far", (11, 0, 0)),
        ]
    )

    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(pet_weight=0.0, max_distance_mm=10.0),
    )
    assert result.pair_costs["fu_lesion_id"].tolist() == ["near"]
    matrix = result.matrices["P01"]
    assert matrix.shape == (1, 2)
    assert np.isfinite(matrix.values[0, matrix.fu_lesion_ids.index("near")])
    assert np.isinf(matrix.values[0, matrix.fu_lesion_ids.index("far")])

def test_candidates_are_ranked_by_cost_then_lesion_id() -> None:
    features = pd.DataFrame(
        [
            lesion("P01", "BL", "BL1", (0, 0, 0)),
            lesion("P01", "FU", "FU_B", (5, 0, 0)),
            lesion("P01", "FU", "FU_A", (-5, 0, 0)),
            lesion("P01", "FU", "FU_C", (20, 0, 0)),
        ]
    )
    config = PairCostConfig(size_weight=0.0, pet_weight=0.0)

    pairs = generate_lesion_pair_costs(features, config).pair_costs

    assert pairs["fu_lesion_id"].tolist() == ["FU_A", "FU_B", "FU_C"]
    assert pairs["candidate_rank_for_bl"].tolist() == [1, 2, 3]

def test_patients_are_scored_independently() -> None:
    features = pd.DataFrame(
        [
            lesion("P02", "BL", "BL2", (0, 0, 0)),
            lesion("P02", "FU", "FU2", (1, 0, 0)),
            lesion("P01", "BL", "BL1", (0, 0, 0)),
            lesion("P01", "FU", "FU1", (2, 0, 0)),
        ]
    )
    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(size_weight=0.0, pet_weight=0.0),
    )

    assert list(result.matrices) == ["P01", "P02"]
    assert set(zip(result.pair_costs["bl_lesion_id"], result.pair_costs["fu_lesion_id"])) == {
        ("BL1", "FU1"),
        ("BL2", "FU2"),
    }

def test_patient_with_no_follow_up_has_empty_matrix_and_no_pairs() -> None:
    features = pd.DataFrame([lesion("P01", "BL", "BL1", (0, 0, 0))])

    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(size_weight=0.0, pet_weight=0.0),
    )
    assert result.pair_costs.empty
    assert result.pair_costs.columns.tolist() == PAIR_COST_COLUMNS
    assert result.matrices["P01"].shape == (1, 0)

@pytest.mark.parametrize(
    ("config", "message"),
    [
        (PairCostConfig(distance_weight=-1), "distance_weight"),
        (PairCostConfig(distance_weight=0, size_weight=0, pet_weight=0), "At least one"),
        (PairCostConfig(distance_scale_mm=0), "distance_scale_mm"),
        (PairCostConfig(max_distance_mm=-1), "max_distance_mm"),
        (PairCostConfig(pet_feature="median"), "pet_feature"),
    ],
)
def test_rejects_invalid_config(config: PairCostConfig, message: str) -> None:
    features = pd.DataFrame(
        [
            lesion("P01", "BL", "BL1", (0, 0, 0)),
            lesion("P01", "FU", "FU1", (1, 0, 0)),
        ]
    )
    with pytest.raises(LesionPairCostError, match=message):
        generate_lesion_pair_costs(features, config)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"timepoint": "MID"}, "only BL/FU"),
        ({"centroid_x_mm": np.nan}, "finite"),
        ({"coordinate_space": "BL_RAS_mm"}, "FU_RAS_mm"),
        ({"volume_ml": 0}, "volume_ml"),
    ],
)
def test_rejects_invalid_feature_values(change: dict[str, object], message: str) -> None:
    rows = [
        lesion("P01", "BL", "BL1", (0, 0, 0)),
        lesion("P01", "FU", "FU1", (1, 0, 0)),
    ]
    rows[0].update(change)

    with pytest.raises(LesionPairCostError, match=message):
        generate_lesion_pair_costs(pd.DataFrame(rows))

def test_rejects_duplicate_lesion_id_within_timepoint() -> None:
    duplicate = lesion("P01", "BL", "BL1", (0, 0, 0))
    features = pd.DataFrame([duplicate, duplicate.copy()])
    with pytest.raises(LesionPairCostError, match="Duplicate lesion identifier"):
        generate_lesion_pair_costs(features)

def test_loads_legacy_lesion_id_and_exports_results(tmp_path) -> None:
    source = pd.DataFrame(
        [
            lesion("patient/01", "BL", "BL1", (0, 0, 0)),
            lesion("patient/01", "FU", "FU1", (1, 0, 0)),
        ]
    ).rename(columns={"lesion_id": "temporary_lesion_id"})
    source_path = tmp_path / "features.csv"
    source.to_csv(source_path, index=False)

    features = load_aligned_feature_csvs([source_path])
    result = generate_lesion_pair_costs(
        features,
        PairCostConfig(size_weight=0.0, pet_weight=0.0),
    )
    pair_path = export_pair_costs(result.pair_costs, tmp_path / "tables" / "pairs.csv")
    matrix_paths = export_cost_matrices(result.matrices, tmp_path / "matrices")

    assert pair_path.exists()
    assert pd.read_csv(pair_path)["total_cost"].iloc[0] == pytest.approx(0.02)
    # The slash in the patient id must not become a directory.
    assert [path.name for path in matrix_paths] == ["patient_01_cost_matrix.csv"]
    matrix = pd.read_csv(matrix_paths[0], index_col="bl_lesion_id")
    assert matrix.loc["BL1", "FU1"] == pytest.approx(0.02)

def test_pet_only_config_requires_a_comparable_pet_value() -> None:
    features = pd.DataFrame(
        [
            lesion("P01", "BL", "BL1", (0, 0, 0), suvmax=8.0),
            lesion("P01", "FU", "FU1", (1, 0, 0), suvmax=np.nan),
        ]
    )
    config = PairCostConfig(
        distance_weight=0.0,
        size_weight=0.0,
        pet_weight=1.0,
        pet_feature="suvmax",
    )
    with pytest.raises(LesionPairCostError, match="No active cost component"):
        generate_lesion_pair_costs(features, config)
        
        
        
        
        
        
        
