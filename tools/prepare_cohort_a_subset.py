from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


TIMEPOINTS = ("BL", "FU")
NII_SUFFIXES = (".nii.gz", ".nii")


@dataclass(frozen=True)
class ScanRecord:
    patient_id: str
    timepoint: str
    image_id: str
    scan_id: str
    ct_path: str
    pet_path: str
    lesion_mask_path: str
    reference_csv_path: str
    reference_json_path: str
    missing_files: str
    is_complete: bool
    copied: bool


def nii_stem(path: Path) -> str:
    name = path.name
    for suffix in NII_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def format_path(path: Path | None, base: Path, path_mode: str) -> str:
    if path is None:
        return ""
    path = path.resolve()
    if path_mode == "absolute":
        return str(path)
    try:
        return str(path.relative_to(base.resolve()))
    except ValueError:
        if path_mode == "relative-to-manifest":
            try:
                return os.path.relpath(path, base.resolve())
            except ValueError:
                return str(path)
        return str(path)


def path_base(root: Path, manifest_dir: Path, path_mode: str) -> Path:
    if path_mode == "relative-to-manifest":
        return manifest_dir
    return root


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def discover_patient_ids(root: Path) -> list[str]:
    inputs = root / "inputsTr"
    if not inputs.exists():
        return []
    return sorted(path.stem for path in inputs.glob("*.csv") if path.is_file())


def image_ids_for_patient(root: Path, patient_id: str, timepoint: str) -> list[str]:
    inputs = root / "inputsTr"
    pattern = f"{patient_id}_{timepoint}_img_*.nii*"
    ids: set[str] = set()
    for path in inputs.glob(pattern):
        stem = nii_stem(path)
        match = re.search(r"_img_(\d+)$", stem)
        if match:
            ids.add(f"{int(match.group(1)):02d}")
    return sorted(ids)


def candidate_pet_paths(root: Path, patient_id: str, timepoint: str, image_id: str) -> list[Path]:
    inputs = root / "inputsTr"
    return [
        inputs / f"{patient_id}_{timepoint}_pet_{image_id}.nii.gz",
        inputs / f"{patient_id}_{timepoint}_PET_{image_id}.nii.gz",
        inputs / f"{patient_id}_{timepoint}_suv_{image_id}.nii.gz",
        inputs / f"{patient_id}_{timepoint}_SUV_{image_id}.nii.gz",
    ]


def candidate_mask_paths(root: Path, patient_id: str, timepoint: str, image_id: str) -> list[Path]:
    inputs = root / "inputsTr"
    targets = root / "targetsTr"
    return [
        inputs / f"{patient_id}_{timepoint}_mask_{image_id}.nii.gz",
        targets / f"{patient_id}_{timepoint}_mask_{image_id}.nii.gz",
        inputs / f"{patient_id}_{timepoint}_label_{image_id}.nii.gz",
        targets / f"{patient_id}_{timepoint}_label_{image_id}.nii.gz",
    ]


def scan_record(
    root: Path,
    patient_id: str,
    timepoint: str,
    image_id: str,
    base: Path,
    path_mode: str,
) -> ScanRecord:
    inputs = root / "inputsTr"
    ct = inputs / f"{patient_id}_{timepoint}_img_{image_id}.nii.gz"
    pet = first_existing(candidate_pet_paths(root, patient_id, timepoint, image_id))
    mask = first_existing(candidate_mask_paths(root, patient_id, timepoint, image_id))
    reference_csv = inputs / f"{patient_id}.csv"
    reference_json = inputs / f"{patient_id}_{timepoint}_{image_id}.json"

    missing = []
    if not ct.exists():
        missing.append("ct_path")
    if pet is None:
        missing.append("pet_path")
    if mask is None:
        missing.append("lesion_mask_path")
    if not reference_csv.exists():
        missing.append("reference_csv_path")
    if not reference_json.exists():
        missing.append("reference_json_path")

    return ScanRecord(
        patient_id=patient_id,
        timepoint=timepoint,
        image_id=image_id,
        scan_id=f"{patient_id}_{timepoint}_{image_id}",
        ct_path=format_path(ct if ct.exists() else None, base, path_mode),
        pet_path=format_path(pet, base, path_mode),
        lesion_mask_path=format_path(mask, base, path_mode),
        reference_csv_path=format_path(reference_csv if reference_csv.exists() else None, base, path_mode),
        reference_json_path=format_path(reference_json if reference_json.exists() else None, base, path_mode),
        missing_files=";".join(missing),
        is_complete=len(missing) == 0,
        copied=False,
    )


def build_manifest(root: Path, patient_ids: list[str], manifest_dir: Path, path_mode: str) -> list[ScanRecord]:
    rows: list[ScanRecord] = []
    base = path_base(root, manifest_dir, path_mode)
    for patient_id in patient_ids:
        for timepoint in TIMEPOINTS:
            ids = image_ids_for_patient(root, patient_id, timepoint)
            if not ids:
                rows.append(
                    ScanRecord(
                        patient_id=patient_id,
                        timepoint=timepoint,
                        image_id="",
                        scan_id=f"{patient_id}_{timepoint}_missing",
                        ct_path="",
                        pet_path="",
                        lesion_mask_path="",
                        reference_csv_path=format_path(root / "inputsTr" / f"{patient_id}.csv", base, path_mode)
                        if (root / "inputsTr" / f"{patient_id}.csv").exists()
                        else "",
                        reference_json_path="",
                        missing_files=f"{timepoint.lower()}_scan",
                        is_complete=False,
                        copied=False,
                    )
                )
                continue
            for image_id in ids:
                rows.append(scan_record(root, patient_id, timepoint, image_id, base, path_mode))
    return rows


def copy_manifest_files(
    rows: list[ScanRecord],
    root: Path,
    subset_dir: Path,
    manifest_dir: Path,
    path_mode: str,
) -> list[ScanRecord]:
    subset_dir.mkdir(parents=True, exist_ok=True)
    copied_rows: list[ScanRecord] = []
    output_base = manifest_dir if path_mode == "relative-to-manifest" else subset_dir
    for row in rows:
        values = row.__dict__.copy()
        copied_any = False
        for field in ["ct_path", "pet_path", "lesion_mask_path", "reference_csv_path", "reference_json_path"]:
            current = values[field]
            if not current:
                continue
            src = Path(current)
            if not src.is_absolute():
                src = root / current
            if not src.exists():
                continue
            dst = subset_dir / src.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            values[field] = format_path(dst, output_base, path_mode)
            copied_any = True
        values["copied"] = copied_any
        copied_rows.append(ScanRecord(**values))
    return copied_rows


def write_scan_manifest(rows: list[ScanRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ScanRecord.__dataclass_fields__.keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_pair_manifest(rows: list[ScanRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "patient_id",
        "bl_scan_id",
        "fu_scan_id",
        "bl_ct_path",
        "bl_pet_path",
        "bl_lesion_mask_path",
        "fu_ct_path",
        "fu_pet_path",
        "fu_lesion_mask_path",
        "reference_csv_path",
        "bl_reference_json_path",
        "fu_reference_json_path",
        "missing_files",
        "is_complete",
    ]
    by_patient: dict[str, dict[str, ScanRecord]] = {}
    for row in rows:
        if row.image_id == "":
            by_patient.setdefault(row.patient_id, {})[row.timepoint] = row
        else:
            by_patient.setdefault(row.patient_id, {}).setdefault(row.timepoint, row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for patient_id in sorted(by_patient):
            bl = by_patient[patient_id].get("BL")
            fu = by_patient[patient_id].get("FU")
            missing = []
            for label, record in [("BL", bl), ("FU", fu)]:
                if record is None:
                    missing.append(f"{label.lower()}_scan")
                elif record.missing_files:
                    missing.extend(f"{label.lower()}_{x}" for x in record.missing_files.split(";") if x)
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "bl_scan_id": bl.scan_id if bl else "",
                    "fu_scan_id": fu.scan_id if fu else "",
                    "bl_ct_path": bl.ct_path if bl else "",
                    "bl_pet_path": bl.pet_path if bl else "",
                    "bl_lesion_mask_path": bl.lesion_mask_path if bl else "",
                    "fu_ct_path": fu.ct_path if fu else "",
                    "fu_pet_path": fu.pet_path if fu else "",
                    "fu_lesion_mask_path": fu.lesion_mask_path if fu else "",
                    "reference_csv_path": (bl.reference_csv_path if bl else "") or (fu.reference_csv_path if fu else ""),
                    "bl_reference_json_path": bl.reference_json_path if bl else "",
                    "fu_reference_json_path": fu.reference_json_path if fu else "",
                    "missing_files": ";".join(missing),
                    "is_complete": len(missing) == 0,
                }
            )


def write_summary(rows: list[ScanRecord], out_json: Path, selected_ids: list[str]) -> None:
    patients = sorted({row.patient_id for row in rows})
    complete_patients = sorted(
        {
            pid
            for pid in patients
            if all(
                any(row.patient_id == pid and row.timepoint == tp and row.is_complete for row in rows)
                for tp in TIMEPOINTS
            )
        }
    )
    payload = {
        "selected_patient_ids": selected_ids,
        "patients_in_manifest": patients,
        "n_patients": len(patients),
        "n_scan_rows": len(rows),
        "n_complete_scan_rows": sum(1 for row in rows if row.is_complete),
        "n_complete_patients": len(complete_patients),
        "complete_patient_ids": complete_patients,
        "missing_by_field": {},
    }
    for row in rows:
        for field in row.missing_files.split(";"):
            if field:
                payload["missing_by_field"][field] = payload["missing_by_field"].get(field, 0) + 1
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a small Cohort A subset manifest with BL/FU imaging, mask, and reference paths."
    )
    parser.add_argument("--root", required=True, help="Cohort A root containing inputsTr/ and targetsTr/.")
    parser.add_argument("--out-dir", default="outputs/cohort_a_subset", help="Directory for manifest outputs.")
    parser.add_argument("--max-patients", type=int, default=5, help="Select the first N patients after sorting.")
    parser.add_argument("--patient-ids", default="", help="Optional comma-separated patient IDs. Overrides --max-patients.")
    parser.add_argument("--copy-files", action="store_true", help="Copy selected files into --subset-dir.")
    parser.add_argument("--subset-dir", default="", help="Destination for copied subset files.")
    parser.add_argument(
        "--path-mode",
        choices=["relative-to-root", "relative-to-manifest", "absolute"],
        default="relative-to-root",
        help="How paths are written in CSV outputs. Defaults to portable paths relative to --root.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Backward-compatible alias for --path-mode relative-to-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    discovered = discover_patient_ids(root)
    if args.patient_ids.strip():
        selected = [pid.strip() for pid in args.patient_ids.split(",") if pid.strip()]
    else:
        selected = discovered[: max(0, args.max_patients)]
    unknown = sorted(set(selected) - set(discovered))
    if unknown:
        raise SystemExit(f"Patient IDs not found under {root / 'inputsTr'}: {', '.join(unknown)}")

    path_mode = "relative-to-root" if args.relative_paths else args.path_mode
    rows = build_manifest(root, selected, out_dir, "absolute" if args.copy_files else path_mode)
    if args.copy_files:
        subset_dir = Path(args.subset_dir).resolve() if args.subset_dir else out_dir / "files"
        rows = copy_manifest_files(rows, root, subset_dir, out_dir, path_mode)

    scan_manifest = out_dir / "cohort_a_subset_manifest.csv"
    pair_manifest = out_dir / "cohort_a_subset_pairs.csv"
    summary_json = out_dir / "cohort_a_subset_summary.json"
    write_scan_manifest(rows, scan_manifest)
    write_pair_manifest(rows, pair_manifest)
    write_summary(rows, summary_json, selected)

    print(f"Selected {len(selected)} patient(s): {', '.join(selected) if selected else '(none)'}")
    print(f"Wrote scan manifest: {scan_manifest}")
    print(f"Wrote pair manifest: {pair_manifest}")
    print(f"Wrote summary: {summary_json}")
    print(f"Path mode: {path_mode}")
    missing_rows = [row for row in rows if row.missing_files]
    if missing_rows:
        print(f"Rows with missing files: {len(missing_rows)}")
        for row in missing_rows[:10]:
            print(f"  {row.patient_id} {row.timepoint} {row.image_id or '-'}: {row.missing_files}")


if __name__ == "__main__":
    main()
