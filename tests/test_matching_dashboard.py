from types import SimpleNamespace
import pandas as pd
import pytest
import src.matching_dashboard as dashboard
from src.lesion_min_cost_flow import MATCH_COLUMNS
from src.matching_dashboard import MatchingDashboardError

def make_matches(patient_id: str, match_type: str = "MATCHED") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "bl_lesion_id": f"{patient_id}_BL_1",
                "fu_lesion_id": f"{patient_id}_FU_1",
                "match_type": match_type,
            }
        ]
    )

def make_features(patient_id: str = "p1") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "timepoint": "BL",
                "lesion_id": f"{patient_id}_BL_1",
                "centroid_x_vox": 1.0,
                "centroid_y_vox": 2.0,
                "centroid_z_vox": 3.0,
                "component_index": 4,
                "connectivity": 26,
                "mask_path": "bl_mask.nii.gz",
                "mask_role": "baseline",
                "coordinate_space": "FU_RAS_mm",
            },
            
            {
                "patient_id": patient_id,
                "timepoint": "FU",
                "lesion_id": f"{patient_id}_FU_1",
                "centroid_x_vox": 5.0,
                "centroid_y_vox": 6.0,
                "centroid_z_vox": 7.0,
                "component_index": 8,
                "connectivity": 18,
                "mask_path": "fu_mask.nii.gz",
                "mask_role": "followup",
                "coordinate_space": "FU_RAS_mm",
            },
        ]
    )

def test_load_patient_matches_filters_and_normalises_values(tmp_path) -> None:
    path = tmp_path / "matches.csv"
    table = pd.concat([make_matches("p1"), make_matches("p2")], ignore_index=True)
    table.loc[1, "patient_id"] = " p2 "
    table.loc[1, "match_type"] = " matched "
    table.to_csv(path, index=False)

    result = dashboard.load_patient_matches(path, "p2")
    assert result["patient_id"].tolist() == ["p2"]
    assert result["match_type"].tolist() == ["MATCHED"]

def test_load_patient_matches_rejects_missing_columns(tmp_path) -> None:
    path = tmp_path / "matches.csv"
    pd.DataFrame({"patient_id": ["p1"]}).to_csv(path, index=False)
    with pytest.raises(MatchingDashboardError, match="missing required"):
        dashboard.load_patient_matches(path, "p1")


def test_load_patient_matches_reports_unknown_patient(tmp_path) -> None:
    path = tmp_path / "matches.csv"
    make_matches("p1").to_csv(path, index=False)
    with pytest.raises(MatchingDashboardError, match="patient 'p2'"):
        dashboard.load_patient_matches(path, "p2")

def test_load_patient_features_accepts_temporary_lesion_id(tmp_path) -> None:
    path = tmp_path / "features.csv"
    features = make_features().rename(columns={"lesion_id": "temporary_lesion_id"})
    features.to_csv(path, index=False)
    result = dashboard.load_patient_features(path, "p1")

    assert "lesion_id" in result.columns
    assert result["timepoint"].tolist() == ["BL", "FU"]

@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"timepoint": "LATER"}, "invalid timepoint"),
        ({"centroid_x_vox": "not-a-number"}, "non-numeric"),
        ({"coordinate_space": "BL_NATIVE"}, "FU_RAS_mm"),
    ],
)
def test_load_patient_features_rejects_invalid_rows(tmp_path, change, message) -> None:
    path = tmp_path / "features.csv"
    features = make_features()
    for column, value in change.items():
        if isinstance(value, str) and column.startswith("centroid_"):
            features[column] = features[column].astype(object)
        features.loc[0, column] = value
    features.to_csv(path, index=False)
    with pytest.raises(MatchingDashboardError, match=message):
        dashboard.load_patient_features(path, "p1")

def test_load_patient_features_rejects_duplicate_lesion_ids(tmp_path) -> None:
    path = tmp_path / "features.csv"
    features = make_features()
    duplicate = features.iloc[[0]].copy()
    pd.concat([features, duplicate], ignore_index=True).to_csv(path, index=False)

    with pytest.raises(MatchingDashboardError, match="duplicate lesion IDs"):
        dashboard.load_patient_features(path, "p1")

def test_resolve_match_image_focus_returns_both_locations() -> None:
    match = make_matches("p1").iloc[0]
    focus = dashboard.resolve_match_image_focus(match, make_features())

    assert focus.match_type == "MATCHED"
    assert focus.baseline is not None
    assert focus.followup is not None
    assert focus.baseline.centroid_voxel == (1.0, 2.0, 3.0)
    assert focus.followup.centroid_voxel == (5.0, 6.0, 7.0)
    assert focus.baseline.component_index == 4
    assert focus.baseline.connectivity == 26


def test_resolve_new_lesion_has_no_baseline_location() -> None:
    match = pd.Series(
        {
            "bl_lesion_id": None,
            "fu_lesion_id": "p1_FU_1",
            "match_type": "NEW",
        }
    )
    focus = dashboard.resolve_match_image_focus(match, make_features())
    assert focus.baseline is None
    assert focus.followup is not None


def test_resolve_match_rejects_unknown_lesion() -> None:
    match = make_matches("p1").iloc[0].copy()
    match["bl_lesion_id"] = "missing"
    with pytest.raises(MatchingDashboardError, match="Could not uniquely locate"):
        dashboard.resolve_match_image_focus(match, make_features())


def test_save_patient_matches_replaces_only_selected_patient(tmp_path) -> None:
    path = tmp_path / "matches.csv"
    pd.concat([make_matches("p1"), make_matches("p2")]).to_csv(path, index=False)
    replacement = make_matches("p1", "DISAPPEARING")
    replacement["fu_lesion_id"] = None
    dashboard.save_patient_matches(path, "p1", replacement)

    saved = pd.read_csv(path, dtype={"patient_id": str})
    assert set(saved["patient_id"]) == {"p1", "p2"}
    assert saved.loc[saved["patient_id"] == "p1", "match_type"].item() == "DISAPPEARING"
    assert not path.with_name("matches.csv.tmp").exists()


def test_save_patient_matches_rejects_rows_for_another_patient(tmp_path) -> None:
    with pytest.raises(MatchingDashboardError, match="different patient"):
        dashboard.save_patient_matches(tmp_path / "matches.csv", "p1", make_matches("p2"))

def test_run_patient_match_calls_matcher_and_saves_result(tmp_path, monkeypatch) -> None:
    matrix_path = tmp_path / "p1_cost_matrix.csv"
    matrix_path.write_text("placeholder", encoding="utf-8")
    expected = make_matches("p1")
    called = {}
    def fake_match(path, *, config, patient_id):
        called.update(path=path, config=config, patient_id=patient_id)
        return SimpleNamespace(matches=expected)
    monkeypatch.setattr(dashboard, "match_patient_cost_matrix", fake_match)
    output_path = tmp_path / "matches.csv"

    result = dashboard.run_patient_match(matrix_path, "p1", output_path)

    #wrapper should pass context through and persist the copy of the res
    assert called["path"] == matrix_path
    assert called["patient_id"] == "p1"
    pd.testing.assert_frame_equal(result, expected)
    assert output_path.exists()

def test_run_patient_match_wraps_matcher_errors(tmp_path, monkeypatch) -> None:
    matrix_path = tmp_path / "p1_cost_matrix.csv"
    matrix_path.write_text("placeholder", encoding="utf-8")

    def fail_match(*args, **kwargs):
        raise ValueError("bad matrix")
    monkeypatch.setattr(dashboard, "match_patient_cost_matrix", fail_match)
    with pytest.raises(MatchingDashboardError, match="bad matrix"):
        dashboard.run_patient_match(matrix_path, "p1", tmp_path / "matches.csv")


def test_summarise_patient_matches_counts_unique_lesions_and_outcomes() -> None:
    matches = pd.DataFrame(
        [
            {"patient_id": "p1", "bl_lesion_id": "b1", "fu_lesion_id": "f1", "match_type": "matched"},
            {"patient_id": "p1", "bl_lesion_id": "b2", "fu_lesion_id": None, "match_type": "DISAPPEARING"},
            {"patient_id": "p1", "bl_lesion_id": None, "fu_lesion_id": "f2", "match_type": "NEW"},
        ]
    )
    summary = dashboard.summarise_patient_matches(matches, "p1")
    assert summary.patient_id == "p1"
    assert summary.baseline_lesions == 2
    assert summary.followup_lesions == 2
    assert summary.outcome_counts == {"MATCHED": 1, "DISAPPEARING": 1, "NEW": 1}

def test_save_new_file_uses_match_columns(tmp_path) -> None:
    path = tmp_path / "nested" / "matches.csv"
    empty_matches = pd.DataFrame(columns=MATCH_COLUMNS)
    dashboard.save_patient_matches(path, "p1", empty_matches)
    assert pd.read_csv(path).columns.tolist() == MATCH_COLUMNS
    
    
    
    
    
    
    
    
    
    
    
    
    
