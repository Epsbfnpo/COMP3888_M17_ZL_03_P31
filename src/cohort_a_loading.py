from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd


class CohortALoadError(RuntimeError):
    """Base class for Cohort A loading failures."""


class ManifestError(CohortALoadError):
    """Raised when the pair manifest is missing required information."""


class MissingImagingFileError(CohortALoadError):
    """Raised when a requested imaging path is blank or does not exist."""


class UnreadableImagingFileError(CohortALoadError):
    """Raised when a NIfTI file exists but cannot be read."""


@dataclass(frozen=True)
class VolumeMetadata:
    path: Path
    shape: tuple[int, ...]
    spacing_mm: tuple[float, float, float]
    affine: np.ndarray
    orientation: tuple[str, str, str]
    dtype: str


@dataclass(frozen=True)
class LoadedVolume:
    data: np.ndarray
    image: nib.spatialimages.SpatialImage
    metadata: VolumeMetadata


@dataclass(frozen=True)
class TimepointVolumes:
    timepoint: str
    scan_id: str
    ct: LoadedVolume | None
    pet: LoadedVolume | None
    lesion_mask: LoadedVolume | None


@dataclass(frozen=True)
class PatientPairVolumes:
    patient_id: str
    baseline: TimepointVolumes
    followup: TimepointVolumes


PAIR_COLUMNS = {
    "BL": {
        "scan_id": "bl_scan_id",
        "ct": "bl_ct_path",
        "pet": "bl_pet_path",
        "lesion_mask": "bl_lesion_mask_path",
    },
    "FU": {
        "scan_id": "fu_scan_id",
        "ct": "fu_ct_path",
        "pet": "fu_pet_path",
        "lesion_mask": "fu_lesion_mask_path",
    },
}


def load_pair_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ManifestError(f"Pair manifest does not exist: {manifest_path}")
    try:
        manifest = pd.read_csv(manifest_path).fillna("")
    except Exception as exc:
        raise ManifestError(f"Could not read pair manifest '{manifest_path}': {exc}") from exc

    required = {"patient_id"}
    for mapping in PAIR_COLUMNS.values():
        required.update(mapping.values())
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ManifestError(
            f"Pair manifest '{manifest_path}' is missing required column(s): {', '.join(missing)}"
        )
    return manifest


def get_patient_row(manifest: pd.DataFrame, patient_id: str) -> pd.Series:
    rows = manifest.loc[manifest["patient_id"].astype(str) == str(patient_id)]
    if rows.empty:
        raise ManifestError(f"Patient '{patient_id}' was not found in the pair manifest.")
    if len(rows) > 1:
        raise ManifestError(
            f"Patient '{patient_id}' appears {len(rows)} times in the pair manifest; expected one BL/FU pair row."
        )
    return rows.iloc[0]


def resolve_manifest_path(
    raw_path: str | Path,
    *,
    manifest_path: str | Path,
    data_root: str | Path | None = None,
) -> Path:
    text = str(raw_path).strip()
    if not text:
        raise MissingImagingFileError("The manifest path is blank.")

    path = Path(text).expanduser()
    if path.is_absolute():
        return path

    manifest_dir = Path(manifest_path).resolve().parent
    candidates: list[Path] = []
    root_text = str(data_root).strip() if data_root is not None else os.environ.get("COHORT_A_ROOT", "").strip()
    if root_text:
        candidates.append(Path(root_text).expanduser().resolve() / path)
    candidates.append(manifest_dir / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    #return the most likely candidate so the error message is useful
    return candidates[0].resolve()


def _metadata(img: nib.spatialimages.SpatialImage, path: Path) -> VolumeMetadata:
    shape = tuple(int(v) for v in img.shape)
    zooms = img.header.get_zooms()
    if len(zooms) < 3:
        raise UnreadableImagingFileError(
            f"NIfTI volume '{path}' does not expose three spatial voxel spacings."
        )
    orientation = tuple(str(v) for v in nib.aff2axcodes(img.affine))
    return VolumeMetadata(
        path=path,
        shape=shape,
        spacing_mm=tuple(float(v) for v in zooms[:3]),
        affine=np.asarray(img.affine, dtype=float).copy(),
        orientation=orientation,  # type: ignore[arg-type]
        dtype=str(img.get_data_dtype()),
    )


def load_nifti_volume(
    path: str | Path,
    *,
    role: str = "volume",
    preserve_dtype: bool = False,
) -> LoadedVolume:
    file_path = Path(path)
    if not file_path.exists():
        raise MissingImagingFileError(f"Missing {role} file: {file_path}")
    if not file_path.is_file():
        raise MissingImagingFileError(f"Expected {role} to be a file, got: {file_path}")

    try:
        img = nib.load(str(file_path))
        if preserve_dtype:
            data = np.asanyarray(img.dataobj)
        else:
            data = img.get_fdata(dtype=np.float32)
        metadata = _metadata(img, file_path.resolve())
    except CohortALoadError:
        raise
    except Exception as exc:
        raise UnreadableImagingFileError(
            f"Could not read {role} NIfTI file '{file_path}': {exc}"
        ) from exc

    if data.ndim < 3:
        raise UnreadableImagingFileError(
            f"{role.capitalize()} '{file_path}' has shape {data.shape}; expected at least 3 dimensions."
        )
    return LoadedVolume(data=np.asarray(data), image=img, metadata=metadata)


def load_timepoint_from_row(
    row: pd.Series,
    *,
    timepoint: str,
    manifest_path: str | Path,
    data_root: str | Path | None = None,
    modalities: Iterable[str] = ("ct", "pet", "lesion_mask"),
    allow_missing: bool = False,
) -> TimepointVolumes:
    tp = timepoint.upper()
    if tp not in PAIR_COLUMNS:
        raise ValueError("timepoint must be 'BL' or 'FU'")

    requested = tuple(modalities)
    unknown = sorted(set(requested) - {"ct", "pet", "lesion_mask"})
    if unknown:
        raise ValueError(f"Unknown modality name(s): {', '.join(unknown)}")

    mapping = PAIR_COLUMNS[tp]
    patient_id = str(row.get("patient_id", "")).strip() or "<unknown>"
    scan_id = str(row.get(mapping["scan_id"], "")).strip()
    loaded: dict[str, LoadedVolume | None] = {"ct": None, "pet": None, "lesion_mask": None}

    for modality in requested:
        column = mapping[modality]
        raw_path = str(row.get(column, "")).strip()
        role = f"{patient_id} {tp} {modality}"
        try:
            resolved = resolve_manifest_path(
                raw_path,
                manifest_path=manifest_path,
                data_root=data_root,
            )
            loaded[modality] = load_nifti_volume(
                resolved,
                role=role,
                preserve_dtype=(modality == "lesion_mask"),
            )
        except MissingImagingFileError as exc:
            if allow_missing:
                loaded[modality] = None
                continue
            if not raw_path:
                raise MissingImagingFileError(
                    f"Cannot load {role}: manifest column '{column}' is blank."
                ) from exc
            raise MissingImagingFileError(
                f"Cannot load {role}: manifest path '{raw_path}' could not be resolved. "
                f"Pass data_root or set COHORT_A_ROOT when the manifest uses paths relative to the Cohort A root."
            ) from exc

    return TimepointVolumes(
        timepoint=tp,
        scan_id=scan_id,
        ct=loaded["ct"],
        pet=loaded["pet"],
        lesion_mask=loaded["lesion_mask"],
    )


def load_patient_pair(
    manifest_path: str | Path,
    patient_id: str,
    *,
    data_root: str | Path | None = None,
    modalities: Iterable[str] = ("ct", "pet"),
    allow_missing: bool = False,
) -> PatientPairVolumes:
    manifest = load_pair_manifest(manifest_path)
    row = get_patient_row(manifest, patient_id)
    baseline = load_timepoint_from_row(
        row,
        timepoint="BL",
        manifest_path=manifest_path,
        data_root=data_root,
        modalities=modalities,
        allow_missing=allow_missing,
    )
    followup = load_timepoint_from_row(
        row,
        timepoint="FU",
        manifest_path=manifest_path,
        data_root=data_root,
        modalities=modalities,
        allow_missing=allow_missing,
    )
    return PatientPairVolumes(patient_id=str(patient_id), baseline=baseline, followup=followup)


def volume_summary(volume: LoadedVolume | None) -> dict[str, object] | None:
    if volume is None:
        return None
    meta = volume.metadata
    return {
        "path": str(meta.path),
        "shape": list(meta.shape),
        "spacing_mm": list(meta.spacing_mm),
        "orientation": list(meta.orientation),
        "dtype": meta.dtype,
        "affine": meta.affine.tolist(),
    }
