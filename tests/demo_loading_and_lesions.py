from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cohort_a_loading import load_patient_pair
from src.lesion_components import extract_individual_lesions


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def build_demo(root: Path, manifest_path: Path) -> list[str]:
    patient_ids = ["demo_patient_01", "demo_patient_02"]
    affine = np.diag([2.0, 2.5, 3.0, 1.0])
    rows = []

    for p_index, patient_id in enumerate(patient_ids, start=1):
        paths: dict[str, str] = {}
        for timepoint in ("BL", "FU"):
            shape = (12, 10, 8)
            ct = np.full(shape, -500 + 10 * p_index, dtype=np.float32)
            pet = np.zeros(shape, dtype=np.float32)
            mask = np.zeros(shape, dtype=np.uint8)

            #2 BL lesions for both patients. FU adds a third lesion for patient 02.
            mask[1:3, 1:3, 1:3] = 1
            mask[7:9, 6:8, 4:6] = 1
            if timepoint == "FU" and p_index == 2:
                mask[4:6, 3:5, 5:7] = 1
            pet[mask > 0] = 4.0 + p_index + (1.0 if timepoint == "FU" else 0.0)

            prefix = f"{patient_id}_{timepoint}"
            ct_path = root / "inputsTr" / f"{prefix}_img_00.nii.gz"
            pet_path = root / "inputsTr" / f"{prefix}_pet_00.nii.gz"
            mask_path = root / "inputsTr" / f"{prefix}_mask_00.nii.gz"
            save_nifti(ct_path, ct, affine)
            save_nifti(pet_path, pet, affine)
            save_nifti(mask_path, mask, affine)
            paths[f"{timepoint.lower()}_ct_path"] = str(ct_path.relative_to(root))
            paths[f"{timepoint.lower()}_pet_path"] = str(pet_path.relative_to(root))
            paths[f"{timepoint.lower()}_lesion_mask_path"] = str(mask_path.relative_to(root))

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

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return patient_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Create two synthetic Cohort A patients and demonstrate both stories.")
    parser.add_argument("--out-dir", default="outputs/loading_lesion_demo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    data_root = out_dir / "cohort_a"
    manifest_path = out_dir / "cohort_a_subset_pairs.csv"
    patient_ids = build_demo(data_root, manifest_path)

    for patient_id in patient_ids:
        pair = load_patient_pair(
            manifest_path,
            patient_id,
            data_root=data_root,
            modalities=("ct", "pet", "lesion_mask"),
        )
        assert pair.baseline.lesion_mask is not None
        assert pair.followup.lesion_mask is not None
        bl = extract_individual_lesions(pair.baseline.lesion_mask, patient_id=patient_id, timepoint="BL")
        fu = extract_individual_lesions(pair.followup.lesion_mask, patient_id=patient_id, timepoint="FU")
        print(
            f"{patient_id}: "
            f"BL CT {pair.baseline.ct.metadata.shape if pair.baseline.ct else None}, "
            f"BL PET {pair.baseline.pet.metadata.shape if pair.baseline.pet else None}, "
            f"FU CT {pair.followup.ct.metadata.shape if pair.followup.ct else None}, "
            f"FU PET {pair.followup.pet.metadata.shape if pair.followup.pet else None}, "
            f"lesions BL={bl.lesion_count}, FU={fu.lesion_count}"
        )
    print(f"Demo data and manifest written under: {out_dir}")


if __name__ == "__main__":
    main()
