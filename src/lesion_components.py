from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi

from .cohort_a_loading import LoadedVolume


class LesionMaskError(ValueError):
    """Base class for invalid lesion masks."""


class EmptyLesionMaskError(LesionMaskError):
    """Raised when a lesion mask contains no positive lesion voxels."""


@dataclass(frozen=True)
class LesionComponent:
    temporary_id: str
    component_index: int
    source_label: int
    voxel_count: int
    volume_ml: float
    centroid_voxel: tuple[float, float, float]
    centroid_world_mm: tuple[float, float, float]


@dataclass(frozen=True)
class LesionExtractionResult:
    patient_id: str
    timepoint: str
    labelled_mask: np.ndarray
    lesions: tuple[LesionComponent, ...]
    connectivity: int

    @property
    def lesion_count(self) -> int:
        return len(self.lesions)


def component_structure(connectivity: int = 18) -> np.ndarray:
    rank_map = {6: 1, 18: 2, 26: 3}
    if connectivity not in rank_map:
        raise ValueError("connectivity must be one of 6, 18, or 26")
    return ndi.generate_binary_structure(3, rank_map[connectivity])


def validate_lesion_mask(mask: np.ndarray) -> np.ndarray:
    data = np.asarray(mask)
    if data.ndim != 3:
        raise LesionMaskError(f"Lesion mask must be 3-D; got shape {data.shape}.")
    if not np.issubdtype(data.dtype, np.number) and data.dtype != np.bool_:
        raise LesionMaskError(f"Lesion mask must contain numeric labels; got dtype {data.dtype}.")
    if not np.all(np.isfinite(data)):
        raise LesionMaskError("Lesion mask contains NaN or infinite values.")
    if np.any(data < 0):
        raise LesionMaskError("Lesion mask contains negative labels.")

    if np.issubdtype(data.dtype, np.floating):
        rounded = np.rint(data)
        if not np.allclose(data, rounded, rtol=0.0, atol=1e-5):
            raise LesionMaskError(
                "Lesion mask contains non-integer floating values; this looks like a probability map rather than a segmentation mask."
            )
        data = rounded

    labels = data.astype(np.int32, copy=False)
    if not np.any(labels > 0):
        raise EmptyLesionMaskError("Lesion mask is empty: no positive lesion voxels were found.")
    return labels


def _temporary_components(mask_labels: np.ndarray, connectivity: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    #Split each positive source label into connected components.

    #Binary mask is processed as source label 1. Multi label mask preserves source labels boundary, 
        #therefore two contact lesions with different source labels will not merge. 
    #Return a new temporary label image and (temporary_1abel, source_1abel) pair.
    structure = component_structure(connectivity)
    output = np.zeros(mask_labels.shape, dtype=np.int32)
    mapping: list[tuple[int, int]] = []
    next_id = 1

    for source_label in sorted(int(v) for v in np.unique(mask_labels) if int(v) > 0):
        cc, n_components = ndi.label(mask_labels == source_label, structure=structure)
        for local_id in range(1, int(n_components) + 1):
            output[cc == local_id] = next_id
            mapping.append((next_id, source_label))
            next_id += 1
    return output, mapping



def extract_individual_lesions(
    mask_volume: LoadedVolume,
    *,
    patient_id: str,
    timepoint: str,
    connectivity: int = 18,
) -> LesionExtractionResult:
    labels = validate_lesion_mask(mask_volume.data)
    temporary_labels, mapping = _temporary_components(labels, connectivity)

    spacing = np.asarray(mask_volume.metadata.spacing_mm, dtype=float)
    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    affine = np.asarray(mask_volume.metadata.affine, dtype=float)
    lesions: list[LesionComponent] = []

    tp = timepoint.upper()
    for temp_id, source_label in mapping:
        component_mask = temporary_labels == temp_id
        coords = np.argwhere(component_mask)
        if len(coords) == 0:
            continue
        centroid_voxel_arr = coords.mean(axis=0)
        homogeneous = np.array(
            [centroid_voxel_arr[0], centroid_voxel_arr[1], centroid_voxel_arr[2], 1.0],
            dtype=float,
        )
        centroid_world = (affine @ homogeneous)[:3]
        voxel_count = int(component_mask.sum())
        lesions.append(
            LesionComponent(
                temporary_id=f"{patient_id}_{tp}_L{temp_id:03d}",
                component_index=temp_id,
                source_label=source_label,
                voxel_count=voxel_count,
                volume_ml=float(voxel_count * voxel_volume_ml),
                centroid_voxel=tuple(float(v) for v in centroid_voxel_arr),
                centroid_world_mm=tuple(float(v) for v in centroid_world),
            )
        )

    return LesionExtractionResult(
        patient_id=str(patient_id),
        timepoint=tp,
        labelled_mask=temporary_labels,
        lesions=tuple(lesions),
        connectivity=connectivity,
    )


def lesion_rows(results: Iterable[LesionExtractionResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        for lesion in result.lesions:
            rows.append(
                {
                    "patient_id": result.patient_id,
                    "timepoint": result.timepoint,
                    "temporary_lesion_id": lesion.temporary_id,
                    "component_index": lesion.component_index,
                    "source_label": lesion.source_label,
                    "voxel_count": lesion.voxel_count,
                    "volume_ml": lesion.volume_ml,
                    "centroid_x_vox": lesion.centroid_voxel[0],
                    "centroid_y_vox": lesion.centroid_voxel[1],
                    "centroid_z_vox": lesion.centroid_voxel[2],
                    "centroid_x_mm": lesion.centroid_world_mm[0],
                    "centroid_y_mm": lesion.centroid_world_mm[1],
                    "centroid_z_mm": lesion.centroid_world_mm[2],
                    "connectivity": result.connectivity,
                }
            )
    return rows
