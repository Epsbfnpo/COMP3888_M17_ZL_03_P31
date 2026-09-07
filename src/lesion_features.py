from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
from .cohort_a_loading import LoadedVolume
from .lesion_components import (
    EmptyLesionMaskError,
    extract_individual_lesions,
    lesion_rows,
)
FEATURE_COLUMNS = [
    'patient_id', 'timepoint', 'lesion_id', 'component_index', 'source_label',
    'voxel_count', 'volume_ml', 'centroid_x_vox', 'centroid_y_vox',
    'centroid_z_vox', 'centroid_x_mm', 'centroid_y_mm', 'centroid_z_mm',
    'connectivity', 'coordinate_space', 'pet_units', 'pet_mean', 'pet_max',
    'suvmean', 'suvmax',
]

def _check_geometry(
        volume: LoadedVolume,
        reference: LoadedVolume,
        role: str,
) -> None:
    affine = np.asarray(volume.metadata.affine)
    if (
            volume.data.ndim != 3
            or affine.shape != (4, 4)
            or not np.all(np.isfinite(affine))
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12
            or not np.allclose(affine[3], [0, 0, 0, 1])
    ):
        raise ValueError(f'{role} must have valid 3-D spatial geometry.')
    #Matching shapes and affines confirm that input uses the FU reference grid.
    if (
            volume.data.shape != reference.data.shape
            or not np.allclose(
        affine, reference.metadata.affine, rtol=0, atol=1e-4,
    )
    ):
        raise ValueError(
            f'{role} must already be resampled onto the FU reference grid.'
        )
    #The pipeline treats unknown spatial units as mm.
    if volume.image.header.get_xyzt_units()[0] not in ('unknown', 'mm'):
        raise ValueError(f'{role} spatial units must be mm.')

def extract_aligned_lesion_features(
        baseline_mask: LoadedVolume,
        followup_mask: LoadedVolume,
        *,
        reference: LoadedVolume,
        patient_id: str,
        baseline_pet: LoadedVolume | None = None,
        followup_pet: LoadedVolume | None = None,
        baseline_pet_units: str = 'unknown',
        followup_pet_units: str = 'unknown',
        connectivity: int = 18,
) -> list[dict[str, object]]:
    """Extract features from registered BL and FU masks on the FU CT grid.
    Coordinates use NIfTI RAS in mm.
    PET inputs must share this grid, with BL PET using the same registration.
    Specify units='SUV' only when PET values are already converted to SUV.
    This function performs no registration, SUV conversion, lesion filtering, or matching.
    Lesion IDs refer to components on the aligned grid, not the native grid.
    """
    if not str(patient_id).strip():
        raise ValueError('patient_id is required.')
    _check_geometry(reference, reference, 'FU reference')
    rows = []
    for tp, mask, pet, units in (
            ('BL', baseline_mask, baseline_pet, baseline_pet_units),
            ('FU', followup_mask, followup_pet, followup_pet_units),
    ):
        _check_geometry(mask, reference, f'{tp} mask')
        if pet is not None:
            _check_geometry(pet, reference, f'{tp} PET')
        try:
            result = extract_individual_lesions(
                mask,
                patient_id=str(patient_id),
                timepoint=tp,
                connectivity=connectivity,
            )
        except EmptyLesionMaskError:
            continue
        for lesion, row in zip(result.lesions, lesion_rows([result])):
            row['lesion_id'] = row.pop('temporary_lesion_id')
            #The determinant gives voxel volume for oblique or sheared affines, division by 1000 converts mm³ to mL
            row['volume_ml'] = lesion.voxel_count * abs(float(
                np.linalg.det(mask.metadata.affine[:3, :3])
            )) / 1000.0
            row.update(
                coordinate_space='FU_RAS_mm',
                pet_units=units if pet is not None else None,
                pet_mean=None,
                pet_max=None,
                suvmean=None,
                suvmax=None,
            )
            if pet is not None:
                values = pet.data[
                    result.labelled_mask == lesion.component_index
                    ]
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f'{lesion.temporary_id}: '
                        'PET contains non-finite lesion values.'
                    )
                row['pet_mean'] = float(np.mean(values, dtype=np.float64))
                row['pet_max'] = float(np.max(values))
                #SUV fields are filled only when the supplied PET units are SUV
                if units.upper() == 'SUV':
                    row['suvmean'], row['suvmax'] = (
                        row['pet_mean'], row['pet_max']
                    )
            rows.append(row)
    return rows

def export_lesion_features(
        rows: list[dict[str, object]],
        path: str | Path,
) -> None:
    """Write one row per lesion; missing PET measurements remain blank."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
