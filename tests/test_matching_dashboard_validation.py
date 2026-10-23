from __future__ import annotations

import pandas as pd
import pytest

import src.matching_dashboard as dashboard
from src.matching_dashboard import MatchingDashboardError


def _matches(patient_id: str = "p1") -> pd.DataFrame:
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


def _feature_row(**overrides) -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": "p1",
        "timepoint": "BL",
        "lesion_id": "p1_BL_1",
        "centroid_x_vox": 1.0,
        "centroid_y_vox": 2.0,
        "centroid_z_vox": 3.0,
    }
    row.update(overrides)
    return row


def test_load_patient_matches_validates_file_and_selected_patient(tmp_path) -> None:
    missing_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="Matching results CSV was not found"):
        dashboard.load_patient_matches(missing_path, "p1")

    path = tmp_path / "matches.csv"
    _matches().to_csv(path, index=False)
    with pytest.raises(MatchingDashboardError, match="patient_id must not be blank"):
        dashboard.load_patient_matches(path, "  ")


def test_load_patient_matches_wraps_csv_read_errors(tmp_path, monkeypatch) -> None:
    path = tmp_path / "matches.csv"
    path.touch()

    def fail_read(*_args, **_kwargs):
        raise OSError("broken input")

    monkeypatch.setattr(dashboard.pd, "read_csv", fail_read)
    with pytest.raises(MatchingDashboardError, match="Could not read matching results CSV"):
        dashboard.load_patient_matches(path, "p1")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("patient_id", "  ", "blank patient_id"),
        ("match_type", None, "blank match_type"),
    ],
)
def test_load_patient_matches_rejects_blank_values(
    tmp_path, column: str, value: object, message: str
) -> None:
    path = tmp_path / "matches.csv"
    _matches().assign(**{column: value}).to_csv(path, index=False)

    with pytest.raises(MatchingDashboardError, match=message):
        dashboard.load_patient_matches(path, "p1")


def test_load_patient_features_validates_file_and_wraps_read_errors(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "features.csv"
    with pytest.raises(FileNotFoundError, match="feature CSV was not found"):
        dashboard.load_patient_features(path, "p1")

    path.touch()

    def fail_read(*_args, **_kwargs):
        raise OSError("broken input")

    monkeypatch.setattr(dashboard.pd, "read_csv", fail_read)
    with pytest.raises(MatchingDashboardError, match="Could not read aligned lesion"):
        dashboard.load_patient_features(path, "p1")


@pytest.mark.parametrize(
    ("table", "patient_id", "message"),
    [
        (pd.DataFrame({"patient_id": ["p1"]}), "p1", "missing required column"),
        (pd.DataFrame([_feature_row()]), "p2", "No aligned lesion features"),
    ],
)
def test_load_patient_features_rejects_missing_data(
    tmp_path, table: pd.DataFrame, patient_id: str, message: str
) -> None:
    path = tmp_path / "features.csv"
    table.to_csv(path, index=False)

    with pytest.raises(MatchingDashboardError, match=message):
        dashboard.load_patient_features(path, patient_id)


def test_load_patient_features_accepts_missing_optional_coordinate_space(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame([_feature_row()]).to_csv(path, index=False)

    result = dashboard.load_patient_features(path, "p1")

    assert result["lesion_id"].tolist() == ["p1_BL_1"]


def test_resolve_match_image_focus_rejects_invalid_feature_tables() -> None:
    match = _matches().iloc[0]
    with pytest.raises(MatchingDashboardError, match="missing required column"):
        dashboard.resolve_match_image_focus(match, pd.DataFrame({"patient_id": ["p1"]}))

    invalid_features = pd.DataFrame([_feature_row(centroid_x_vox=float("inf"))])
    with pytest.raises(MatchingDashboardError, match="invalid centroid"):
        dashboard.resolve_match_image_focus(match, invalid_features)


def test_resolve_match_image_focus_preserves_optional_mask_metadata() -> None:
    match = pd.Series(
        {
            "bl_lesion_id": "p1_BL_1",
            "fu_lesion_id": None,
            "match_type": " disappearing ",
        }
    )
    features = pd.DataFrame(
        [
            _feature_row(
                component_index=7,
                connectivity=26,
                mask_path="masks/bl.nii.gz",
                mask_role="baseline",
            )
        ]
    )

    focus = dashboard.resolve_match_image_focus(match, features)

    assert focus.match_type == "DISAPPEARING"
    assert focus.followup is None
    assert focus.baseline is not None
    assert focus.baseline.component_index == 7
    assert focus.baseline.connectivity == 26
    assert focus.baseline.mask_path == "masks/bl.nii.gz"
    assert focus.baseline.mask_role == "baseline"


def test_save_patient_matches_validates_patient_and_columns(tmp_path) -> None:
    path = tmp_path / "matches.csv"
    with pytest.raises(MatchingDashboardError, match="patient_id must not be blank"):
        dashboard.save_patient_matches(path, " ", _matches())

    with pytest.raises(MatchingDashboardError, match="missing required column"):
        dashboard.save_patient_matches(
            path, "p1", pd.DataFrame({"patient_id": ["p1"]})
        )


def test_run_patient_match_reports_missing_matrix(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Cost matrix CSV was not found"):
        dashboard.run_patient_match(
            tmp_path / "costs.csv", "p1", tmp_path / "matches.csv"
        )


def test_summarise_patient_matches_requires_result_columns() -> None:
    with pytest.raises(MatchingDashboardError, match="missing required column"):
        dashboard.summarise_patient_matches(
            pd.DataFrame({"patient_id": ["p1"]}), "p1"
        )
