from __future__ import annotations

import pandas as pd
import pytest

from src.lesion_min_cost_flow import MatcherConfig
from src.matching_dashboard import (
    MatchingDashboardError,
    load_patient_matches,
    load_patient_features,
    resolve_match_image_focus,
    run_patient_match,
    save_patient_matches,
    summarise_patient_matches,
)


def _matches(patient_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "bl_lesion_id": f"{patient_id}_BL_1",
                "fu_lesion_id": f"{patient_id}_FU_1",
                "match_type": "MATCHED",
            }
        ]
    )


def test_load_patient_matches_filters_selected_patient(tmp_path) -> None:
    path = tmp_path / "lesion_matches.csv"
    pd.concat([_matches("p1"), _matches("p2")]).to_csv(path, index=False)

    selected = load_patient_matches(path, "p2")

    assert selected["patient_id"].tolist() == ["p2"]


def test_load_patient_matches_reports_missing_patient(tmp_path) -> None:
    path = tmp_path / "lesion_matches.csv"
    _matches("p1").to_csv(path, index=False)

    with pytest.raises(MatchingDashboardError, match="patient 'p2'"):
        load_patient_matches(path, "p2")


def test_save_patient_matches_replaces_only_selected_patient(tmp_path) -> None:
    path = tmp_path / "lesion_matches.csv"
    pd.concat([_matches("p1"), _matches("p2")]).to_csv(path, index=False)
    replacement = _matches("p1")
    replacement.loc[0, "match_type"] = "DISAPPEARING"
    replacement.loc[0, "fu_lesion_id"] = None

    save_patient_matches(path, "p1", replacement)

    assert load_patient_matches(path, "p1")["match_type"].tolist() == ["DISAPPEARING"]
    assert load_patient_matches(path, "p2")["match_type"].tolist() == ["MATCHED"]


def test_run_patient_match_writes_loadable_results(tmp_path) -> None:
    matrix_path = tmp_path / "p1_cost_matrix.csv"
    pd.DataFrame(
        {"bl_lesion_id": ["p1_BL_1"], "p1_FU_1": [0.1]}
    ).to_csv(matrix_path, index=False)
    output_path = tmp_path / "lesion_matches.csv"

    result = run_patient_match(
        matrix_path,
        "p1",
        output_path,
        MatcherConfig(max_bl_per_fu=1),
    )

    assert result["match_type"].tolist() == ["MATCHED"]
    assert load_patient_matches(output_path, "p1")["fu_lesion_id"].tolist() == ["p1_FU_1"]


def test_summary_counts_bl_fu_and_outcomes() -> None:
    matches = pd.DataFrame(
        [
            {"patient_id": "p", "bl_lesion_id": "b1", "fu_lesion_id": "f1", "match_type": "MATCHED"},
            {"patient_id": "p", "bl_lesion_id": "b2", "fu_lesion_id": None, "match_type": "DISAPPEARING"},
            {"patient_id": "p", "bl_lesion_id": None, "fu_lesion_id": "f2", "match_type": "NEW"},
        ]
    )

    summary = summarise_patient_matches(matches, "p")

    assert summary.baseline_lesions == 2
    assert summary.followup_lesions == 2
    assert summary.outcome_counts == {"MATCHED": 1, "DISAPPEARING": 1, "NEW": 1}


def test_resolves_match_to_aligned_bl_and_fu_centroids(tmp_path) -> None:
    feature_path = tmp_path / "aligned_lesion_features.csv"
    pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "timepoint": "BL",
                "lesion_id": "p1_BL_1",
                "centroid_x_vox": 4.0,
                "centroid_y_vox": 5.0,
                "centroid_z_vox": 6.0,
                "coordinate_space": "FU_RAS_mm",
            },
            {
                "patient_id": "p1",
                "timepoint": "FU",
                "lesion_id": "p1_FU_1",
                "centroid_x_vox": 7.0,
                "centroid_y_vox": 8.0,
                "centroid_z_vox": 9.0,
                "coordinate_space": "FU_RAS_mm",
            },
        ]
    ).to_csv(feature_path, index=False)

    features = load_patient_features(feature_path, "p1")
    focus = resolve_match_image_focus(_matches("p1").iloc[0], features)

    assert focus.baseline is not None
    assert focus.followup is not None
    assert focus.baseline.centroid_voxel == (4.0, 5.0, 6.0)
    assert focus.followup.centroid_voxel == (7.0, 8.0, 9.0)


def test_new_lesion_focus_has_only_followup_location(tmp_path) -> None:
    feature_path = tmp_path / "aligned_lesion_features.csv"
    pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "timepoint": "FU",
                "lesion_id": "p1_FU_2",
                "centroid_x_vox": 1.0,
                "centroid_y_vox": 2.0,
                "centroid_z_vox": 3.0,
            }
        ]
    ).to_csv(feature_path, index=False)
    match = pd.Series(
        {
            "patient_id": "p1",
            "bl_lesion_id": None,
            "fu_lesion_id": "p1_FU_2",
            "match_type": "NEW",
        }
    )

    focus = resolve_match_image_focus(
        match,
        load_patient_features(feature_path, "p1"),
    )

    assert focus.baseline is None
    assert focus.followup is not None


def test_feature_loader_rejects_non_aligned_coordinate_space(tmp_path) -> None:
    feature_path = tmp_path / "aligned_lesion_features.csv"
    pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "timepoint": "BL",
                "lesion_id": "p1_BL_1",
                "centroid_x_vox": 1.0,
                "centroid_y_vox": 2.0,
                "centroid_z_vox": 3.0,
                "coordinate_space": "BL_NATIVE",
            }
        ]
    ).to_csv(feature_path, index=False)

    with pytest.raises(MatchingDashboardError, match="FU_RAS_mm"):
        load_patient_features(feature_path, "p1")
