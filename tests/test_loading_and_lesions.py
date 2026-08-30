from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.cohort_a_loading import (
    MissingImagingFileError,
    UnreadableImagingFileError,
    load_patient_pair,
)
from src.lesion_components import EmptyLesionMaskError, LesionMaskError, extract_individual_lesions


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine if affine is not None else np.eye(4)), str(path))


def make_manifest(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    root = tmp_path / "cohort_a"
    manifest = tmp_path / "outputs" / "cohort_a_subset_pairs.csv"
    patient_ids = ["patient_alpha", "patient_beta"]
    affine = np.diag([2.0, 2.5, 3.0, 1.0])
    rows = []

    for idx, patient_id in enumerate(patient_ids):
        paths = {}
        for tp in ("BL", "FU"):
            shape = (10, 9, 8)
            ct = np.full(shape, idx, dtype=np.float32)
            pet = np.full(shape, idx + 1.5, dtype=np.float32)
            mask = np.zeros(shape, dtype=np.uint8)
            mask[1:3, 1:3, 1:3] = 1
            mask[6:8, 5:7, 4:6] = 1
            if patient_id == "patient_beta" and tp == "FU":
                mask[4:5, 2:4, 5:7] = 1

            ct_path = root / "inputsTr" / f"{patient_id}_{tp}_img_00.nii.gz"
            pet_path = root / "inputsTr" / f"{patient_id}_{tp}_pet_00.nii.gz"
            mask_path = root / "inputsTr" / f"{patient_id}_{tp}_mask_00.nii.gz"
            save_nifti(ct_path, ct, affine)
            save_nifti(pet_path, pet, affine)
            save_nifti(mask_path, mask, affine)
            paths[f"{tp.lower()}_ct_path"] = str(ct_path.relative_to(root))
            paths[f"{tp.lower()}_pet_path"] = str(pet_path.relative_to(root))
            paths[f"{tp.lower()}_lesion_mask_path"] = str(mask_path.relative_to(root))

        rows.append(
            {
                "patient_id": patient_id,
                "bl_scan_id": f"{patient_id}_BL_00",
                "fu_scan_id": f"{patient_id}_FU_00",
                **paths,
                "reference_csv_path": "",
                "bl_reference_json_path": "",
                "fu_reference_json_path": "",
                "missing_files": "",
                "is_complete": True,
            }
        )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return root, manifest, patient_ids


def test_loads_bl_fu_pet_ct_and_metadata_for_multiple_patients(tmp_path: Path) -> None:
    root, manifest, patient_ids = make_manifest(tmp_path)
    for patient_id in patient_ids:
        pair = load_patient_pair(manifest, patient_id, data_root=root, modalities=("ct", "pet"))
        for tp in (pair.baseline, pair.followup):
            assert tp.ct is not None
            assert tp.pet is not None
            assert tp.ct.metadata.shape == (10, 9, 8)
            assert tp.pet.metadata.shape == (10, 9, 8)
            assert tp.ct.metadata.spacing_mm == pytest.approx((2.0, 2.5, 3.0))
            assert tp.ct.metadata.orientation == ("R", "A", "S")
            assert tp.ct.metadata.affine.shape == (4, 4)


def test_extracts_individual_lesions_and_assigns_temporary_ids(tmp_path: Path) -> None:
    root, manifest, _ = make_manifest(tmp_path)
    pair = load_patient_pair(manifest, "patient_beta", data_root=root, modalities=("lesion_mask",))
    assert pair.baseline.lesion_mask is not None
    assert pair.followup.lesion_mask is not None

    bl = extract_individual_lesions(pair.baseline.lesion_mask, patient_id="patient_beta", timepoint="BL")
    fu = extract_individual_lesions(pair.followup.lesion_mask, patient_id="patient_beta", timepoint="FU")

    assert bl.lesion_count == 2
    assert fu.lesion_count == 3
    assert [x.temporary_id for x in bl.lesions] == ["patient_beta_BL_L001", "patient_beta_BL_L002"]
    assert [x.temporary_id for x in fu.lesions] == [
        "patient_beta_FU_L001",
        "patient_beta_FU_L002",
        "patient_beta_FU_L003",
    ]
    assert all(x.volume_ml > 0 for x in bl.lesions + fu.lesions)


def test_multilabel_mask_keeps_touching_source_labels_separate(tmp_path: Path) -> None:
    path = tmp_path / "multi.nii.gz"
    data = np.zeros((5, 5, 5), dtype=np.int16)
    data[1:3, 1:3, 1:3] = 1
    data[3:4, 1:3, 1:3] = 2  # touches label 1 but is a distinct source label
    save_nifti(path, data)

    from src.cohort_a_loading import load_nifti_volume

    volume = load_nifti_volume(path, preserve_dtype=True)
    result = extract_individual_lesions(volume, patient_id="p", timepoint="BL", connectivity=6)
    assert result.lesion_count == 2
    assert [x.source_label for x in result.lesions] == [1, 2]


def test_empty_and_invalid_masks_are_detected(tmp_path: Path) -> None:
    from src.cohort_a_loading import load_nifti_volume

    empty_path = tmp_path / "empty.nii.gz"
    save_nifti(empty_path, np.zeros((5, 5, 5), dtype=np.uint8))
    empty = load_nifti_volume(empty_path, preserve_dtype=True)
    with pytest.raises(EmptyLesionMaskError, match="empty"):
        extract_individual_lesions(empty, patient_id="p", timepoint="BL")

    invalid_path = tmp_path / "invalid.nii.gz"
    invalid = np.zeros((5, 5, 5), dtype=np.float32)
    invalid[1, 1, 1] = 0.4
    save_nifti(invalid_path, invalid)
    invalid_volume = load_nifti_volume(invalid_path, preserve_dtype=True)
    with pytest.raises(LesionMaskError, match="probability map"):
        extract_individual_lesions(invalid_volume, patient_id="p", timepoint="BL")


def test_missing_and_unreadable_files_produce_clear_errors(tmp_path: Path) -> None:
    root, manifest, _ = make_manifest(tmp_path)
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.replace("inputsTr/patient_alpha_BL_pet_00.nii.gz", ""), encoding="utf-8")
    with pytest.raises(MissingImagingFileError, match="bl_pet_path"):
        load_patient_pair(manifest, "patient_alpha", data_root=root, modalities=("pet",))

    # Restore, then replace one PET NIfTI with invalid text.
    root, manifest, _ = make_manifest(tmp_path / "second")
    broken = root / "inputsTr" / "patient_alpha_BL_pet_00.nii.gz"
    broken.write_text("not a nifti file", encoding="utf-8")
    with pytest.raises(UnreadableImagingFileError, match="Could not read"):
        load_patient_pair(manifest, "patient_alpha", data_root=root, modalities=("pet",))


def test_resolves_outputsTr_mask_and_cohort_a_root_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, manifest, _ = make_manifest(tmp_path)
    source = root / "inputsTr" / "patient_alpha_BL_mask_00.nii.gz"
    target = root / "outputsTr" / "patient_alpha_BL_mask_00.nii.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)

    text = manifest.read_text(encoding="utf-8")
    text = text.replace(
        "inputsTr/patient_alpha_BL_mask_00.nii.gz",
        "outputsTr/patient_alpha_BL_mask_00.nii.gz",
    )
    manifest.write_text(text, encoding="utf-8")

    monkeypatch.setenv("COHORT_A_ROOT", str(root))
    pair = load_patient_pair(manifest, "patient_alpha", modalities=("lesion_mask",))
    assert pair.baseline.lesion_mask is not None
    assert pair.baseline.lesion_mask.metadata.path == target.resolve()
