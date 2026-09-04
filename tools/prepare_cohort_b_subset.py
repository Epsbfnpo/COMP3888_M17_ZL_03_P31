from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


TIMEPOINTS = ("BL", "FU")
NII_SUFFIXES = (".nii.gz", ".nii")

# Cohort B / autoPET III naming convention:
#   imagesTr/psma_<patient_hash>_<study_date>_0000.nii.gz  CT
#   imagesTr/psma_<patient_hash>_<study_date>_0001.nii.gz  PET (SUV)
#   labelsTr/psma_<patient_hash>_<study_date>.nii.gz       lesion mask
#
# The project/client notes describe <study_date> as the final date token.
# Support the common YYYYMMDD, YYYY-MM-DD and YYYY_MM_DD forms.
STUDY_RE = re.compile(
    r"^psma_(?P<patient>.+)_(?P<date>"
    r"\d{8}|\d{4}-\d{2}-\d{2}|\d{4}_\d{2}_\d{2})$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RawStudy:
    patient_id: str
    study_date: str
    study_key: str
    ct_path: Path
    pet_path: Path | None
    lesion_mask_path: Path | None

    @property
    def date_value(self) -> date:
        return parse_study_date(self.study_date)


@dataclass(frozen=True)
class ScanRecord:
    """
    Cohort-B scan row using the same downstream fields as Cohort A.

    reference_* fields are intentionally blank because Cohort B has no
    expert BL<->FU correspondence ground truth.
    """

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
    lower = name.lower()
    for suffix in NII_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def parse_study_date(text: str) -> date:
    compact = text.replace("-", "").replace("_", "")
    if len(compact) != 8 or not compact.isdigit():
        raise ValueError(f"Unsupported study date: {text!r}")
    return date(
        int(compact[0:4]),
        int(compact[4:6]),
        int(compact[6:8]),
    )


def parse_study_key(study_key: str) -> tuple[str, str] | None:
    """
    Parse:
        psma_<patient_hash>_<date>

    Returns:
        (patient_hash, date_text)
    """
    match = STUDY_RE.fullmatch(study_key)
    if match is None:
        return None

    patient_id = match.group("patient").strip()
    study_date = match.group("date").strip()

    # Reject malformed calendar dates even if the regex matched.
    try:
        parse_study_date(study_date)
    except ValueError:
        return None

    if not patient_id:
        return None

    return patient_id, study_date


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


def _with_nii_suffix(directory: Path, stem: str) -> Path | None:
    for suffix in NII_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def discover_studies(root: Path) -> tuple[list[RawStudy], list[str]]:
    """
    Discover PSMA CT scans and recover patient/date metadata from filenames.

    Only files beginning with `psma_` are considered. FDG studies are ignored.
    """
    images_dir = root / "imagesTr"
    labels_dir = root / "labelsTr"

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Expected Cohort B images directory not found: {images_dir}"
        )

    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected Cohort B labels directory not found: {labels_dir}"
        )

    ct_paths: list[Path] = []
    for suffix in NII_SUFFIXES:
        ct_paths.extend(images_dir.glob(f"psma_*_0000{suffix}"))

    studies: list[RawStudy] = []
    unparsed: list[str] = []

    for ct_path in sorted(set(path.resolve() for path in ct_paths)):
        ct_stem = nii_stem(ct_path)

        if not ct_stem.lower().endswith("_0000"):
            continue

        study_key = ct_stem[:-5]
        parsed = parse_study_key(study_key)

        if parsed is None:
            unparsed.append(ct_path.name)
            continue

        patient_id, study_date = parsed

        pet_path = _with_nii_suffix(
            images_dir,
            f"{study_key}_0001",
        )
        mask_path = _with_nii_suffix(
            labels_dir,
            study_key,
        )

        studies.append(
            RawStudy(
                patient_id=patient_id,
                study_date=study_date,
                study_key=study_key,
                ct_path=ct_path,
                pet_path=pet_path,
                lesion_mask_path=mask_path,
            )
        )

    return studies, unparsed


def group_studies(
    studies: Iterable[RawStudy],
) -> dict[str, list[RawStudy]]:
    grouped: dict[str, list[RawStudy]] = {}

    for study in studies:
        grouped.setdefault(study.patient_id, []).append(study)

    for patient_id in grouped:
        grouped[patient_id].sort(
            key=lambda study: (study.date_value, study.study_key)
        )

    return grouped


def exact_two_timepoint_patients(
    grouped: dict[str, list[RawStudy]],
) -> list[str]:
    return sorted(
        patient_id
        for patient_id, studies in grouped.items()
        if len(studies) == 2
    )


def _geometry(path: Path) -> dict[str, object]:
    """
    Read NIfTI header geometry without materialising the full image array.
    """
    image = nib.load(str(path))

    shape = tuple(int(v) for v in image.shape[:3])
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    affine = np.asarray(image.affine, dtype=float)

    return {
        "shape": shape,
        "spacing_mm": spacing,
        "affine": affine,
    }


def compare_ct_geometry(bl_path: Path, fu_path: Path) -> dict[str, object]:
    """
    Compare BL/FU geometry and compute a simple ranking score.

    The score is used only for choosing useful registration test cases:
      +4 shape differs
      +2 spacing differs
      +1 affine differs
    """
    bl = _geometry(bl_path)
    fu = _geometry(fu_path)

    shape_diff = bl["shape"] != fu["shape"]
    spacing_diff = not np.allclose(
        bl["spacing_mm"],
        fu["spacing_mm"],
        rtol=0.0,
        atol=1e-4,
    )
    affine_diff = not np.allclose(
        bl["affine"],
        fu["affine"],
        rtol=0.0,
        atol=1e-3,
    )

    score = (
        (4 if shape_diff else 0)
        + (2 if spacing_diff else 0)
        + (1 if affine_diff else 0)
    )

    return {
        "score": score,
        "shape_differs": bool(shape_diff),
        "spacing_differs": bool(spacing_diff),
        "affine_differs": bool(affine_diff),
        "bl_shape": list(bl["shape"]),
        "fu_shape": list(fu["shape"]),
        "bl_spacing_mm": list(bl["spacing_mm"]),
        "fu_spacing_mm": list(fu["spacing_mm"]),
    }


def rank_by_geometry_difference(
    patient_ids: list[str],
    grouped: dict[str, list[RawStudy]],
) -> tuple[list[str], dict[str, dict[str, object]]]:
    comparisons: dict[str, dict[str, object]] = {}

    for patient_id in patient_ids:
        studies = grouped[patient_id]
        if len(studies) != 2:
            continue

        bl, fu = studies
        try:
            comparisons[patient_id] = compare_ct_geometry(
                bl.ct_path,
                fu.ct_path,
            )
        except Exception as exc:
            comparisons[patient_id] = {
                "score": -1,
                "geometry_error": str(exc),
            }

    ranked = sorted(
        patient_ids,
        key=lambda patient_id: (
            -int(comparisons.get(patient_id, {}).get("score", -1)),
            patient_id,
        ),
    )

    return ranked, comparisons


def raw_study_to_record(
    study: RawStudy,
    *,
    timepoint: str,
    base: Path,
    path_mode: str,
) -> ScanRecord:
    missing: list[str] = []

    if not study.ct_path.exists():
        missing.append("ct_path")
    if study.pet_path is None:
        missing.append("pet_path")
    if study.lesion_mask_path is None:
        missing.append("lesion_mask_path")

    return ScanRecord(
        patient_id=study.patient_id,
        timepoint=timepoint,
        image_id=study.study_date,
        scan_id=study.study_key,
        ct_path=format_path(
            study.ct_path if study.ct_path.exists() else None,
            base,
            path_mode,
        ),
        pet_path=format_path(
            study.pet_path,
            base,
            path_mode,
        ),
        lesion_mask_path=format_path(
            study.lesion_mask_path,
            base,
            path_mode,
        ),
        reference_csv_path="",
        reference_json_path="",
        missing_files=";".join(missing),
        is_complete=len(missing) == 0,
        copied=False,
    )


def build_manifest(
    root: Path,
    patient_ids: list[str],
    grouped: dict[str, list[RawStudy]],
    manifest_dir: Path,
    path_mode: str,
) -> list[ScanRecord]:
    """
    Convert exactly-two-timepoint Cohort B studies to BL/FU scan records.
    """
    rows: list[ScanRecord] = []
    base = path_base(root, manifest_dir, path_mode)

    for patient_id in patient_ids:
        studies = grouped.get(patient_id, [])

        if len(studies) != 2:
            raise ValueError(
                f"Patient {patient_id!r} has {len(studies)} discovered PSMA "
                "timepoints; this tool currently requires exactly two."
            )

        bl_study, fu_study = studies

        rows.append(
            raw_study_to_record(
                bl_study,
                timepoint="BL",
                base=base,
                path_mode=path_mode,
            )
        )
        rows.append(
            raw_study_to_record(
                fu_study,
                timepoint="FU",
                base=base,
                path_mode=path_mode,
            )
        )

    return rows


def _resolve_record_path(
    value: str,
    *,
    root: Path,
) -> Path | None:
    if not value:
        return None

    path = Path(value)
    if not path.is_absolute():
        path = root / path

    return path.resolve()


def copy_manifest_files(
    rows: list[ScanRecord],
    root: Path,
    subset_dir: Path,
    manifest_dir: Path,
    path_mode: str,
) -> list[ScanRecord]:
    """
    Copy CT/PET/mask files for the selected patients into a portable subset.

    The copied directory keeps the original imagesTr/labelsTr layout.
    """
    subset_dir.mkdir(parents=True, exist_ok=True)

    copied_rows: list[ScanRecord] = []
    output_base = (
        manifest_dir
        if path_mode == "relative-to-manifest"
        else subset_dir
    )

    for row in rows:
        values = row.__dict__.copy()
        copied_any = False

        for field in ("ct_path", "pet_path", "lesion_mask_path"):
            src = _resolve_record_path(
                str(values[field]),
                root=root,
            )
            if src is None or not src.exists():
                continue

            try:
                relative = src.relative_to(root)
            except ValueError:
                relative = Path(src.name)

            dst = subset_dir / relative
            dst.parent.mkdir(parents=True, exist_ok=True)

            if not dst.exists():
                shutil.copy2(src, dst)

            values[field] = format_path(
                dst,
                output_base,
                path_mode,
            )
            copied_any = True

        values["copied"] = copied_any
        copied_rows.append(ScanRecord(**values))

    return copied_rows


def write_scan_manifest(
    rows: list[ScanRecord],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = list(ScanRecord.__dataclass_fields__.keys())

    with out_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(row.__dict__)


def write_pair_manifest(
    rows: list[ScanRecord],
    out_csv: Path,
) -> None:
    """
    Write the same wide columns consumed by the existing Cohort A loader.

    Cohort B does not have expert correspondence files, so the reference
    columns remain blank.
    """
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
        by_patient.setdefault(row.patient_id, {})[row.timepoint] = row

    with out_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for patient_id in sorted(by_patient):
            bl = by_patient[patient_id].get("BL")
            fu = by_patient[patient_id].get("FU")

            missing: list[str] = []

            for label, record in (("BL", bl), ("FU", fu)):
                if record is None:
                    missing.append(f"{label.lower()}_scan")
                    continue

                if record.missing_files:
                    missing.extend(
                        f"{label.lower()}_{field}"
                        for field in record.missing_files.split(";")
                        if field
                    )

            writer.writerow(
                {
                    "patient_id": patient_id,
                    "bl_scan_id": bl.scan_id if bl else "",
                    "fu_scan_id": fu.scan_id if fu else "",
                    "bl_ct_path": bl.ct_path if bl else "",
                    "bl_pet_path": bl.pet_path if bl else "",
                    "bl_lesion_mask_path": (
                        bl.lesion_mask_path if bl else ""
                    ),
                    "fu_ct_path": fu.ct_path if fu else "",
                    "fu_pet_path": fu.pet_path if fu else "",
                    "fu_lesion_mask_path": (
                        fu.lesion_mask_path if fu else ""
                    ),
                    "reference_csv_path": "",
                    "bl_reference_json_path": "",
                    "fu_reference_json_path": "",
                    "missing_files": ";".join(missing),
                    "is_complete": len(missing) == 0,
                }
            )


def write_summary(
    *,
    rows: list[ScanRecord],
    out_json: Path,
    selected_ids: list[str],
    all_grouped: dict[str, list[RawStudy]],
    unparsed_filenames: list[str],
    geometry_comparisons: dict[str, dict[str, object]],
) -> None:
    selected_patients = sorted({row.patient_id for row in rows})

    complete_patients = sorted(
        patient_id
        for patient_id in selected_patients
        if all(
            any(
                row.patient_id == patient_id
                and row.timepoint == timepoint
                and row.is_complete
                for row in rows
            )
            for timepoint in TIMEPOINTS
        )
    )

    timepoint_counts = {
        patient_id: len(studies)
        for patient_id, studies in sorted(all_grouped.items())
    }

    non_exact_two = {
        patient_id: count
        for patient_id, count in timepoint_counts.items()
        if count != 2
    }

    missing_by_field: dict[str, int] = {}
    for row in rows:
        for field in row.missing_files.split(";"):
            if field:
                missing_by_field[field] = (
                    missing_by_field.get(field, 0) + 1
                )

    payload = {
        "cohort": "B",
        "source": "autoPET III PSMA subset",
        "selected_patient_ids": selected_ids,
        "patients_in_manifest": selected_patients,
        "n_patients": len(selected_patients),
        "n_scan_rows": len(rows),
        "n_complete_scan_rows": sum(
            1 for row in rows if row.is_complete
        ),
        "n_complete_patients": len(complete_patients),
        "complete_patient_ids": complete_patients,
        "n_discovered_psma_patients": len(all_grouped),
        "n_exact_two_timepoint_patients": sum(
            1 for studies in all_grouped.values()
            if len(studies) == 2
        ),
        "non_exact_two_timepoint_patients": non_exact_two,
        "missing_by_field": missing_by_field,
        "unparsed_psma_ct_filenames": unparsed_filenames,
        "selected_geometry_comparison": {
            patient_id: geometry_comparisons.get(patient_id, {})
            for patient_id in selected_ids
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a small longitudinal Cohort B / autoPET III PSMA subset "
            "manifest by grouping scans by patient hash and sorting study dates."
        )
    )

    parser.add_argument(
        "--root",
        default="",
        help=(
            "Cohort B root containing imagesTr/ and labelsTr/. "
            "If omitted, COHORT_B_ROOT is used."
        ),
    )

    parser.add_argument(
        "--out-dir",
        default="outputs/cohort_b_subset",
        help="Directory for manifest outputs.",
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=5,
        help="Select N exactly-two-timepoint PSMA patients.",
    )

    parser.add_argument(
        "--patient-ids",
        default="",
        help=(
            "Optional comma-separated patient hashes. "
            "Overrides --max-patients."
        ),
    )

    parser.add_argument(
        "--prefer-geometry-differences",
        action="store_true",
        help=(
            "Rank patients with BL/FU shape, spacing or affine differences "
            "first. Useful for registration testing."
        ),
    )

    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy selected CT/PET/mask files into --subset-dir.",
    )

    parser.add_argument(
        "--subset-dir",
        default="",
        help="Destination for copied subset files.",
    )

    parser.add_argument(
        "--path-mode",
        choices=[
            "relative-to-root",
            "relative-to-manifest",
            "absolute",
        ],
        default="relative-to-root",
        help=(
            "How paths are written in CSV outputs. "
            "Defaults to paths relative to --root."
        ),
    )

    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help=(
            "Backward-compatible alias for "
            "--path-mode relative-to-root."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_text = (
        args.root.strip()
        or os.environ.get("COHORT_B_ROOT", "").strip()
    )

    if not root_text:
        raise SystemExit(
            "No Cohort B root provided. "
            "Use --root or set COHORT_B_ROOT."
        )

    root = Path(root_text).expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"Cohort B root does not exist: {root}")

    out_dir = Path(args.out_dir).expanduser().resolve()

    try:
        studies, unparsed = discover_studies(root)
    except (FileNotFoundError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    if not studies:
        raise SystemExit(
            f"No parseable PSMA CT scans were found under "
            f"{root / 'imagesTr'}."
        )

    grouped = group_studies(studies)
    exact_two = exact_two_timepoint_patients(grouped)

    if not exact_two:
        raise SystemExit(
            "No PSMA patients with exactly two discovered timepoints were found."
        )

    # Geometry is calculated for the selected patients by default, and for all
    # candidates only when it is needed for ranking.
    geometry_comparisons: dict[str, dict[str, object]] = {}

    if args.patient_ids.strip():
        selected = [
            patient_id.strip()
            for patient_id in args.patient_ids.split(",")
            if patient_id.strip()
        ]

        unknown = sorted(
            patient_id
            for patient_id in selected
            if patient_id not in grouped
        )
        if unknown:
            raise SystemExit(
                "Patient IDs not found in the PSMA subset: "
                + ", ".join(unknown)
            )

        wrong_count = {
            patient_id: len(grouped[patient_id])
            for patient_id in selected
            if len(grouped[patient_id]) != 2
        }
        if wrong_count:
            details = ", ".join(
                f"{patient_id} ({count} timepoints)"
                for patient_id, count in wrong_count.items()
            )
            raise SystemExit(
                "Selected patients must have exactly two timepoints: "
                + details
            )

    else:
        candidates = exact_two

        if args.prefer_geometry_differences:
            candidates, geometry_comparisons = (
                rank_by_geometry_difference(
                    candidates,
                    grouped,
                )
            )

        selected = candidates[: max(0, args.max_patients)]

    if not selected:
        raise SystemExit("No patients selected.")

    # Ensure geometry summary exists for every selected patient.
    for patient_id in selected:
        if patient_id not in geometry_comparisons:
            bl, fu = grouped[patient_id]
            try:
                geometry_comparisons[patient_id] = compare_ct_geometry(
                    bl.ct_path,
                    fu.ct_path,
                )
            except Exception as exc:
                geometry_comparisons[patient_id] = {
                    "score": -1,
                    "geometry_error": str(exc),
                }

    path_mode = (
        "relative-to-root"
        if args.relative_paths
        else args.path_mode
    )

    # Absolute source paths are easiest during the optional copy step.
    build_path_mode = (
        "absolute"
        if args.copy_files
        else path_mode
    )

    rows = build_manifest(
        root,
        selected,
        grouped,
        out_dir,
        build_path_mode,
    )

    if args.copy_files:
        subset_dir = (
            Path(args.subset_dir).expanduser().resolve()
            if args.subset_dir
            else out_dir / "files"
        )

        rows = copy_manifest_files(
            rows,
            root,
            subset_dir,
            out_dir,
            path_mode,
        )

    scan_manifest = out_dir / "cohort_b_subset_manifest.csv"
    pair_manifest = out_dir / "cohort_b_subset_pairs.csv"
    summary_json = out_dir / "cohort_b_subset_summary.json"

    write_scan_manifest(rows, scan_manifest)
    write_pair_manifest(rows, pair_manifest)

    write_summary(
        rows=rows,
        out_json=summary_json,
        selected_ids=selected,
        all_grouped=grouped,
        unparsed_filenames=unparsed,
        geometry_comparisons=geometry_comparisons,
    )

    print(
        f"Discovered {len(grouped)} PSMA patient(s); "
        f"{len(exact_two)} have exactly two timepoints."
    )
    print(
        f"Selected {len(selected)} patient(s): "
        + ", ".join(selected)
    )

    if args.prefer_geometry_differences:
        print("Selection preference: BL/FU geometry differences first.")

    print()
    print("Selected geometry checks:")
    for patient_id in selected:
        comparison = geometry_comparisons.get(patient_id, {})
        if "geometry_error" in comparison:
            print(
                f"  {patient_id}: geometry check failed: "
                f"{comparison['geometry_error']}"
            )
            continue

        print(
            f"  {patient_id}: "
            f"BL {tuple(comparison.get('bl_shape', []))} -> "
            f"FU {tuple(comparison.get('fu_shape', []))}; "
            f"shape_diff={comparison.get('shape_differs')}, "
            f"spacing_diff={comparison.get('spacing_differs')}, "
            f"affine_diff={comparison.get('affine_differs')}"
        )

    print()
    print(f"Wrote scan manifest: {scan_manifest}")
    print(f"Wrote pair manifest: {pair_manifest}")
    print(f"Wrote summary: {summary_json}")
    print(f"Path mode: {path_mode}")

    missing_rows = [
        row for row in rows
        if row.missing_files
    ]

    if missing_rows:
        print(f"Rows with missing files: {len(missing_rows)}")
        for row in missing_rows[:10]:
            print(
                f"  {row.patient_id} {row.timepoint} "
                f"{row.scan_id}: {row.missing_files}"
            )

    if unparsed:
        print(
            f"Warning: {len(unparsed)} psma_* CT filename(s) "
            "could not be parsed as psma_<hash>_<date>."
        )
        for filename in unparsed[:10]:
            print(f"  {filename}")


if __name__ == "__main__":
    main()
