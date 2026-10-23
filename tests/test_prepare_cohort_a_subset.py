from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

from tools import prepare_cohort_a_subset as subset


def _touch(path: Path, content: str = "test") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_complete_patient(root: Path, patient_id: str = "patient_alpha") -> None:
    inputs = root / "inputsTr"
    _touch(inputs / f"{patient_id}.csv", "lesion_id\n1\n")
    for timepoint in subset.TIMEPOINTS:
        prefix = f"{patient_id}_{timepoint}"
        _touch(inputs / f"{prefix}_img_00.nii.gz")
        _touch(inputs / f"{prefix}_pet_00.nii.gz")
        _touch(root / "outputsTr" / f"{prefix}_mask_00.nii.gz")
        _touch(inputs / f"{prefix}_00.json", "{}")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("scan.nii.gz", "scan"),
        ("scan.nii", "scan"),
        ("scan.txt", "scan"),
    ],
)
def test_nii_stem_handles_supported_and_plain_suffixes(name: str, expected: str) -> None:
    assert subset.nii_stem(Path(name)) == expected


def test_discovery_and_manifest_cover_complete_and_missing_scans(tmp_path: Path) -> None:
    root = tmp_path / "cohort_a"
    _make_complete_patient(root)
    _touch(root / "inputsTr" / "patient_missing.csv")
    _touch(root / "inputsTr" / "patient_alpha_BL_img_01.nii")
    _touch(root / "inputsTr" / "patient_alpha_BL_img_invalid.nii.gz")

    assert subset.discover_patient_ids(root) == ["patient_alpha", "patient_missing"]
    assert subset.image_ids_for_patient(root, "patient_alpha", "BL") == ["00", "01"]

    rows = subset.build_manifest(
        root,
        ["patient_alpha", "patient_missing"],
        tmp_path / "manifests",
        "relative-to-root",
    )

    complete = next(
        row
        for row in rows
        if row.patient_id == "patient_alpha" and row.timepoint == "FU"
    )
    assert complete.is_complete
    assert complete.lesion_mask_path == str(
        Path("outputsTr") / "patient_alpha_FU_mask_00.nii.gz"
    )

    incomplete = next(
        row
        for row in rows
        if row.patient_id == "patient_alpha" and row.image_id == "01"
    )
    assert not incomplete.is_complete
    assert set(incomplete.missing_files.split(";")) == {
        "ct_path",
        "pet_path",
        "lesion_mask_path",
        "reference_json_path",
    }

    missing_rows = [row for row in rows if row.patient_id == "patient_missing"]
    assert [row.missing_files for row in missing_rows] == ["bl_scan", "fu_scan"]
    assert all(
        row.reference_csv_path == str(Path("inputsTr") / "patient_missing.csv")
        for row in missing_rows
    )


def test_path_helpers_and_candidate_name_variants(tmp_path: Path) -> None:
    root = tmp_path / "cohort_a"
    inputs = root / "inputsTr"
    pet = _touch(inputs / "p_BL_SUV_03.nii.gz")
    mask = _touch(root / "targetsTr" / "p_BL_label_03.nii.gz")

    assert subset.first_existing([inputs / "missing.nii.gz", pet]) == pet
    assert subset.first_existing([inputs / "missing.nii.gz"]) is None
    assert subset.first_existing(subset.candidate_pet_paths(root, "p", "BL", "03")) == pet
    assert subset.first_existing(subset.candidate_mask_paths(root, "p", "BL", "03")) == mask
    assert subset.path_base(root, tmp_path / "manifests", "relative-to-root") == root
    assert subset.path_base(root, tmp_path / "manifests", "relative-to-manifest") == (
        tmp_path / "manifests"
    )
    assert subset.format_path(pet, root, "relative-to-root") == str(
        Path("inputsTr") / pet.name
    )
    assert subset.format_path(pet, root, "absolute") == str(pet.resolve())
    assert subset.format_path(None, root, "absolute") == ""


def test_copy_and_writers_create_portable_manifests(tmp_path: Path) -> None:
    root = tmp_path / "cohort_a"
    out_dir = tmp_path / "manifests"
    subset_dir = out_dir / "files"
    _make_complete_patient(root)
    rows = subset.build_manifest(root, ["patient_alpha"], out_dir, "absolute")

    copied = subset.copy_manifest_files(
        rows,
        root,
        subset_dir,
        out_dir,
        "relative-to-manifest",
    )

    assert all(row.copied for row in copied)
    assert all(not Path(row.ct_path).is_absolute() for row in copied)
    assert (subset_dir / "inputsTr" / "patient_alpha_BL_img_00.nii.gz").exists()

    scan_path = out_dir / "scan.csv"
    pair_path = out_dir / "pairs.csv"
    summary_path = out_dir / "summary.json"
    subset.write_scan_manifest(copied, scan_path)
    subset.write_pair_manifest(copied, pair_path)
    subset.write_summary(copied, summary_path, ["patient_alpha"])

    with scan_path.open(newline="", encoding="utf-8") as handle:
        scan_rows = list(csv.DictReader(handle))
    with pair_path.open(newline="", encoding="utf-8") as handle:
        pair_rows = list(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(scan_rows) == 2
    assert pair_rows[0]["is_complete"] == "True"
    assert pair_rows[0]["missing_files"] == ""
    assert summary["n_complete_patients"] == 1
    assert summary["complete_patient_ids"] == ["patient_alpha"]
    assert summary["missing_by_field"] == {}


def test_pair_manifest_and_summary_report_missing_timepoint(tmp_path: Path) -> None:
    root = tmp_path / "cohort_a"
    _touch(root / "inputsTr" / "patient_missing.csv")
    rows = subset.build_manifest(
        root,
        ["patient_missing"],
        tmp_path,
        "relative-to-root",
    )
    # Exercise the branch where a timepoint has no record at all.
    rows = [row for row in rows if row.timepoint == "BL"]
    pair_path = tmp_path / "pairs.csv"
    summary_path = tmp_path / "summary.json"

    subset.write_pair_manifest(rows, pair_path)
    subset.write_summary(rows, summary_path, ["patient_missing"])

    with pair_path.open(newline="", encoding="utf-8") as handle:
        pair = next(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert pair["missing_files"] == "bl_bl_scan;fu_scan"
    assert pair["fu_scan_id"] == ""
    assert summary["n_complete_patients"] == 0
    assert summary["missing_by_field"] == {"bl_scan": 1}


def test_main_rejects_missing_root_and_unknown_patient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COHORT_A_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", ["prepare_cohort_a_subset.py"])
    with pytest.raises(SystemExit, match="No Cohort A root provided"):
        subset.main()

    root = tmp_path / "cohort_a"
    _make_complete_patient(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_cohort_a_subset.py",
            "--root",
            str(root),
            "--patient-ids",
            "unknown",
        ],
    )
    with pytest.raises(SystemExit, match="Patient IDs not found"):
        subset.main()


def test_main_builds_copied_subset_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = tmp_path / "cohort_a"
    out_dir = tmp_path / "output"
    _make_complete_patient(root)
    monkeypatch.setenv("COHORT_A_ROOT", str(root))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_cohort_a_subset.py",
            "--out-dir",
            str(out_dir),
            "--max-patients",
            "1",
            "--copy-files",
            "--relative-paths",
        ],
    )

    subset.main()

    output = capsys.readouterr().out
    assert "Selected 1 patient(s): patient_alpha" in output
    assert "Path mode: relative-to-root" in output
    assert (out_dir / "cohort_a_subset_manifest.csv").exists()
    assert (out_dir / "cohort_a_subset_pairs.csv").exists()
    assert (out_dir / "cohort_a_subset_summary.json").exists()
    assert (out_dir / "files" / "inputsTr" / "patient_alpha_BL_img_00.nii.gz").exists()
