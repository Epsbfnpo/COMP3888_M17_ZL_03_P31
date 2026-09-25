"""P31-26 functional tests: patient data loading from the Cohort A pair manifest.

The fast tests build a deterministic Cohort A-shaped directory with BL/FU CT,
PET and lesion-mask NIfTI files.  Each modality/timepoint contains a distinct
sentinel value, so path or timepoint mix-ups are caught rather than merely
checking that *some* volume loaded.

The final test is an optional real-data check.  ``make test-p31-26`` enables it
after generating a small Cohort A manifest from ``DATA``.  The normal project
suite leaves that test skipped so CI/developer machines do not require the
private/large Cohort A dataset.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.cohort_a_loading import (
    ManifestError,
    MissingImagingFileError,
    UnreadableImagingFileError,
    load_pair_manifest,
    load_patient_pair,
    resolve_manifest_path,
)
from tools.prepare_cohort_a_subset import build_manifest, write_pair_manifest


SHAPE = (8, 7, 6)
AFFINE = np.diag([2.0, 2.5, 3.0, 1.0])


def _save_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, AFFINE), str(path))


def _build_synthetic_cohort_a(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """Create two representative Cohort A-style patients and a real pair manifest."""
    root = tmp_path / "cohort_a"
    out_dir = tmp_path / "outputs" / "cohort_a_subset"
    patient_ids = ["patient_alpha", "patient_beta"]

    # Every modality/timepoint gets a different value.  If BL/FU, CT/PET, or
    # mask paths are accidentally swapped, the assertions below detect it.
    sentinels = {
        "patient_alpha": {
            "BL": {"ct": 11.0, "pet": 21.0, "mask": 1},
            "FU": {"ct": 12.0, "pet": 22.0, "mask": 2},
        },
        "patient_beta": {
            "BL": {"ct": 31.0, "pet": 41.0, "mask": 3},
            "FU": {"ct": 32.0, "pet": 42.0, "mask": 4},
        },
    }

    for patient_id in patient_ids:
        inputs = root / "inputsTr"
        inputs.mkdir(parents=True, exist_ok=True)
        # Cohort A discovery uses the patient CSV as the patient identifier source.
        (inputs / f"{patient_id}.csv").write_text("reference\n", encoding="utf-8")

        for tp in ("BL", "FU"):
            values = sentinels[patient_id][tp]
            _save_nifti(
                inputs / f"{patient_id}_{tp}_img_00.nii.gz",
                np.full(SHAPE, values["ct"], dtype=np.float32),
            )
            _save_nifti(
                inputs / f"{patient_id}_{tp}_pet_00.nii.gz",
                np.full(SHAPE, values["pet"], dtype=np.float32),
            )

            mask = np.zeros(SHAPE, dtype=np.uint8)
            mask[1:3, 2:4, 1:3] = values["mask"]
            # Exercise the real preparer's supported mask locations.
            mask_dir = root / ("outputsTr" if patient_id == "patient_alpha" else "targetsTr")
            _save_nifti(mask_dir / f"{patient_id}_{tp}_mask_00.nii.gz", mask)

            (inputs / f"{patient_id}_{tp}_00.json").write_text("{}", encoding="utf-8")

    rows = build_manifest(root, patient_ids, out_dir, "relative-to-root")
    pair_manifest = out_dir / "cohort_a_subset_pairs.csv"
    write_pair_manifest(rows, pair_manifest)
    return root, pair_manifest, patient_ids


@pytest.fixture
def synthetic_case(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    return _build_synthetic_cohort_a(tmp_path)


def test_p31_26_valid_patient_id_loads_bl_fu_ct_pet_and_masks(synthetic_case) -> None:
    root, manifest, _ = synthetic_case

    pair = load_patient_pair(
        manifest,
        "patient_alpha",
        data_root=root,
        modalities=("ct", "pet", "lesion_mask"),
    )

    assert pair.patient_id == "patient_alpha"
    assert pair.baseline.timepoint == "BL"
    assert pair.followup.timepoint == "FU"
    assert pair.baseline.scan_id == "patient_alpha_BL_00"
    assert pair.followup.scan_id == "patient_alpha_FU_00"

    for timepoint in (pair.baseline, pair.followup):
        assert timepoint.ct is not None
        assert timepoint.pet is not None
        assert timepoint.lesion_mask is not None
        assert timepoint.ct.metadata.shape == SHAPE
        assert timepoint.pet.metadata.shape == SHAPE
        assert timepoint.lesion_mask.metadata.shape == SHAPE


def test_p31_26_manifest_paths_and_timepoint_associations_are_exact(synthetic_case) -> None:
    root, manifest, _ = synthetic_case
    table = load_pair_manifest(manifest)
    row = table.loc[table["patient_id"] == "patient_alpha"].iloc[0]

    pair = load_patient_pair(
        manifest,
        "patient_alpha",
        data_root=root,
        modalities=("ct", "pet", "lesion_mask"),
    )

    checks = [
        (pair.baseline.ct, "bl_ct_path", 11.0),
        (pair.baseline.pet, "bl_pet_path", 21.0),
        (pair.baseline.lesion_mask, "bl_lesion_mask_path", 1.0),
        (pair.followup.ct, "fu_ct_path", 12.0),
        (pair.followup.pet, "fu_pet_path", 22.0),
        (pair.followup.lesion_mask, "fu_lesion_mask_path", 2.0),
    ]

    for loaded, column, expected_value in checks:
        assert loaded is not None
        expected_path = resolve_manifest_path(row[column], manifest_path=manifest, data_root=root)
        assert loaded.metadata.path == expected_path.resolve()
        if "mask" in column:
            assert float(np.max(loaded.data)) == expected_value
        else:
            assert float(np.mean(loaded.data)) == pytest.approx(expected_value)


def test_p31_26_loaded_data_is_ready_for_downstream_processing(synthetic_case) -> None:
    root, manifest, _ = synthetic_case
    pair = load_patient_pair(
        manifest,
        "patient_beta",
        data_root=root,
        modalities=("ct", "pet", "lesion_mask"),
    )

    for tp in (pair.baseline, pair.followup):
        assert tp.ct is not None and tp.pet is not None and tp.lesion_mask is not None
        assert tp.ct.data.ndim == 3
        assert tp.pet.data.ndim == 3
        assert tp.lesion_mask.data.ndim == 3
        assert tp.ct.data.shape == tp.pet.data.shape == tp.lesion_mask.data.shape == SHAPE
        assert np.isfinite(tp.ct.data).all()
        assert np.isfinite(tp.pet.data).all()
        assert np.isfinite(tp.lesion_mask.metadata.affine).all()
        assert tp.ct.metadata.spacing_mm == pytest.approx((2.0, 2.5, 3.0))
        assert tp.ct.metadata.orientation == ("R", "A", "S")


def test_p31_26_unknown_patient_id_has_clear_error(synthetic_case) -> None:
    root, manifest, _ = synthetic_case
    with pytest.raises(ManifestError, match="Patient 'does_not_exist' was not found"):
        load_patient_pair(manifest, "does_not_exist", data_root=root, modalities=("ct",))


@pytest.mark.parametrize("problem", ["blank", "missing", "unreadable"])
def test_p31_26_missing_or_invalid_file_has_clear_error(synthetic_case, problem: str) -> None:
    root, manifest, _ = synthetic_case
    table = pd.read_csv(manifest).fillna("")
    row_index = table.index[table["patient_id"] == "patient_alpha"][0]
    raw_path = table.loc[row_index, "fu_pet_path"]
    pet_path = resolve_manifest_path(raw_path, manifest_path=manifest, data_root=root)

    if problem == "blank":
        table.loc[row_index, "fu_pet_path"] = ""
        table.to_csv(manifest, index=False)
        error_type = MissingImagingFileError
        message = "fu_pet_path"
    elif problem == "missing":
        pet_path.unlink()
        error_type = MissingImagingFileError
        message = "patient_alpha FU pet"
    else:
        pet_path.write_text("this is not a NIfTI file", encoding="utf-8")
        error_type = UnreadableImagingFileError
        message = "Could not read patient_alpha FU pet"

    with pytest.raises(error_type, match=message):
        load_patient_pair(
            manifest,
            "patient_alpha",
            data_root=root,
            modalities=("pet",),
        )


def test_p31_26_representative_real_cohort_a_patients_load_successfully() -> None:
    """Run only when explicitly enabled by ``make test-p31-26``.

    This test verifies the exact same loader against the generated manifest and
    representative real Cohort A patients on the developer/client machine.
    """
    if os.environ.get("P31_RUN_REAL_LOADING_TESTS", "").strip() != "1":
        pytest.skip("real Cohort A loading check is enabled by make test-p31-26")

    root_text = os.environ.get("DATA_ROOT", "").strip()
    manifest_text = os.environ.get("P31_PAIR_MANIFEST", "").strip()
    assert root_text, "DATA_ROOT is required for the real P31-26 loading test"
    assert manifest_text, "P31_PAIR_MANIFEST is required for the real P31-26 loading test"

    root = Path(root_text).expanduser().resolve()
    manifest = Path(manifest_text).expanduser().resolve()
    assert root.is_dir(), f"Real Cohort A data root does not exist: {root}"
    assert manifest.is_file(), f"Generated Cohort A pair manifest does not exist: {manifest}"

    table = load_pair_manifest(manifest)
    required_paths = [
        "bl_ct_path", "bl_pet_path", "bl_lesion_mask_path",
        "fu_ct_path", "fu_pet_path", "fu_lesion_mask_path",
    ]
    usable = table.loc[table[required_paths].astype(str).apply(lambda col: col.str.strip().ne("" )).all(axis=1)]
    assert not usable.empty, "No manifest row contains complete BL/FU CT, PET and lesion-mask paths"

    # Two patients are enough to demonstrate that loading is not tied to one hard-coded ID.
    representative = usable.head(2)
    for _, row in representative.iterrows():
        patient_id = str(row["patient_id"])
        pair = load_patient_pair(
            manifest,
            patient_id,
            data_root=root,
            modalities=("ct", "pet", "lesion_mask"),
        )
        assert pair.baseline.timepoint == "BL"
        assert pair.followup.timepoint == "FU"
        for tp in (pair.baseline, pair.followup):
            assert tp.ct is not None and tp.pet is not None and tp.lesion_mask is not None
            assert tp.ct.data.ndim >= 3
            assert tp.pet.data.ndim >= 3
            assert tp.lesion_mask.data.ndim >= 3

        # Verify every loaded file is exactly the file named by this patient's manifest row.
        associations = [
            (pair.baseline.ct, "bl_ct_path"),
            (pair.baseline.pet, "bl_pet_path"),
            (pair.baseline.lesion_mask, "bl_lesion_mask_path"),
            (pair.followup.ct, "fu_ct_path"),
            (pair.followup.pet, "fu_pet_path"),
            (pair.followup.lesion_mask, "fu_lesion_mask_path"),
        ]
        for loaded, column in associations:
            expected = resolve_manifest_path(row[column], manifest_path=manifest, data_root=root)
            assert loaded is not None
            assert loaded.metadata.path == expected.resolve()
