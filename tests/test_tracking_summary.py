from __future__ import annotations

import math

import pandas as pd
import pytest

from src.tracking_summary import (
    TrackingSummaryError,
    build_tracking_summary,
    build_tracking_summary_from_files,
    load_evaluation_summary,
    load_tracking_matches,
    summarise_patient_tracking,
)


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            
            {"patient_id": "p1", "bl_lesion_id": "p1_BL_1", "fu_lesion_id": "p1_FU_1", "match_type": "MERGING"},
            {"patient_id": "p1", "bl_lesion_id": "p1_BL_2", "fu_lesion_id": "p1_FU_1", "match_type": "MERGING"},
            {"patient_id": "p1", "bl_lesion_id": "p1_BL_3", "fu_lesion_id": None, "match_type": "DISAPPEARING"},
            {"patient_id": "p1", "bl_lesion_id": None, "fu_lesion_id": "p1_FU_2", "match_type": "NEW"},
            # p2: one ordinary one-to-one match.
            {"patient_id": "p2", "bl_lesion_id": "p2_BL_1", "fu_lesion_id": "p2_FU_1", "match_type": "MATCHED"},
        ]
    )


def _evaluation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "ground_truth_links": 5,
                "predicted_links": 4,
                "correct_matches": 3,
                "incorrect_matches": 1,
                "missed_matches": 2,
                "tracking_accuracy": 0.6,
                "precision": 0.75,
                "recall": 0.6,
                "f1": 2 * 0.75 * 0.6 / (0.75 + 0.6),
                "topology_accuracy": 0.4,
                "evaluation_status": "OK",
                "error": "",
            },
            {
                "patient_id": "p2",
                "ground_truth_links": 1,
                "predicted_links": 1,
                "correct_matches": 1,
                "incorrect_matches": 0,
                "missed_matches": 0,
                "tracking_accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "topology_accuracy": 1.0,
                "evaluation_status": "OK",
                "error": "",
            },
            {
                "patient_id": "ALL",
                "ground_truth_links": 6,
                "predicted_links": 5,
                "correct_matches": 4,
                "incorrect_matches": 1,
                "missed_matches": 2,
                "tracking_accuracy": 4 / 6,
                "precision": 4 / 5,
                "recall": 4 / 6,
                "f1": 0.7272727273,
                "topology_accuracy": 0.5,
                "evaluation_status": "OK_PATIENTS=2/2",
                "error": "",
            },
        ]
    )


def test_patient_summary_counts_matches_appearing_disappearing_and_merge() -> None:
    row = summarise_patient_tracking(_matches(), "p1")

    assert row["bl_lesions"] == 3
    assert row["fu_lesions"] == 2
    assert row["matched_lesions"] == 2
    assert row["matched_fu_lesions"] == 1
    assert row["matched_links"] == 2
    assert row["appearing_lesions"] == 1
    assert row["disappearing_lesions"] == 1
    assert row["unmatched_lesions"] == 2
    assert row["merging_links"] == 2
    assert row["merged_fu_lesions"] == 1
    assert row["ground_truth_available"] is False
    assert math.isnan(row["tracking_accuracy"])


def test_patient_summary_attaches_ground_truth_accuracy() -> None:
    row = summarise_patient_tracking(
        _matches(),
        "p1",
        evaluation_summary=_evaluation(),
    )

    assert row["ground_truth_available"] is True
    assert row["correct_matches"] == 3
    assert row["incorrect_matches"] == 1
    assert row["missed_matches"] == 2
    assert row["tracking_accuracy"] == pytest.approx(0.6)


def test_overall_row_sums_tracking_structure_and_uses_evaluation_all_row() -> None:
    summary = build_tracking_summary(_matches(), evaluation_summary=_evaluation())
    overall = summary.loc[summary["patient_id"] == "ALL"].iloc[0]

    assert overall["bl_lesions"] == 4
    assert overall["fu_lesions"] == 3
    assert overall["matched_lesions"] == 3
    assert overall["matched_fu_lesions"] == 2
    assert overall["matched_links"] == 3
    assert overall["unmatched_lesions"] == 2
    assert overall["correct_matches"] == 4
    assert overall["incorrect_matches"] == 1
    assert overall["tracking_accuracy"] == pytest.approx(4 / 6)
    assert overall["evaluation_status"] == "OK_PATIENTS=2/2"


def test_overall_evaluation_falls_back_to_patient_rows_when_all_missing() -> None:
    evaluation = _evaluation()
    evaluation = evaluation[evaluation["patient_id"] != "ALL"].copy()

    summary = build_tracking_summary(_matches(), evaluation_summary=evaluation)
    overall = summary.loc[summary["patient_id"] == "ALL"].iloc[0]

    assert overall["ground_truth_links"] == 6
    assert overall["correct_matches"] == 4
    assert overall["incorrect_matches"] == 1
    assert overall["missed_matches"] == 2
    assert overall["tracking_accuracy"] == pytest.approx(4 / 6)
    assert overall["evaluation_status"] == "OK_PATIENTS=2/2"


def test_failed_evaluation_is_reported_as_unavailable() -> None:
    evaluation = pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "correct_matches": None,
                "incorrect_matches": None,
                "tracking_accuracy": None,
                "evaluation_status": "FAILED",
                "error": "GT mapping failed",
            }
        ]
    )

    row = summarise_patient_tracking(
        _matches(), "p1", evaluation_summary=evaluation
    )

    assert row["ground_truth_available"] is False
    assert row["evaluation_status"] == "FAILED"
    assert row["evaluation_error"] == "GT mapping failed"
    assert math.isnan(row["tracking_accuracy"])


def test_file_loaders_preserve_leading_zero_patient_ids(tmp_path) -> None:
    match_path = tmp_path / "lesion_matches.csv"
    eval_path = tmp_path / "lesion_tracking_summary.csv"

    pd.DataFrame(
        [
            {
                "patient_id": "006f52e910",
                "bl_lesion_id": "006f52e910_BL_L001",
                "fu_lesion_id": "006f52e910_FU_L001",
                "match_type": "MATCHED",
            }
        ]
    ).to_csv(match_path, index=False)
    pd.DataFrame(
        [
            {
                "patient_id": "006f52e910",
                "correct_matches": 1,
                "incorrect_matches": 0,
                "tracking_accuracy": 1.0,
            }
        ]
    ).to_csv(eval_path, index=False)

    matches = load_tracking_matches(match_path)
    evaluation = load_evaluation_summary(eval_path)
    summary = build_tracking_summary_from_files(
        match_path,
        evaluation_summary_path=eval_path,
        include_overall=False,
    )

    assert matches.loc[0, "patient_id"] == "006f52e910"
    assert evaluation.loc[0, "patient_id"] == "006f52e910"
    assert summary.loc[0, "patient_id"] == "006f52e910"


def test_missing_patient_is_clear_error() -> None:
    with pytest.raises(TrackingSummaryError, match="patient 'missing'"):
        summarise_patient_tracking(_matches(), "missing")
