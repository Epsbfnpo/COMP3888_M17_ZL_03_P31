from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .cohort_a_loading import LoadedVolume, PatientPairVolumes

class VisualizationError(ValueError):
    """raised when an image or mask pair cannot be visualised safely"""

@dataclass(frozen=True)
class TimepointDisplay:
    #BL(baseline)/FU(Follow-up)
    timepoint: str
    slice_index: int
    lesion_voxels_on_slice: int
    total_lesion_voxels: int

@dataclass(frozen=True)
class PatientDisplayResult:
    patient_id: str
    #CT/PET
    modality: str
    baseline: TimepointDisplay
    followup: TimepointDisplay
    output_path: Path | None

def validate_image_mask_alignment(image: LoadedVolume, mask: LoadedVolume) -> None:
    if image.data.ndim != 3 or mask.data.ndim != 3:
        raise VisualizationError(
            f"Image and mask must both be 3-D; got {image.data.shape} and {mask.data.shape}."
        )#check if the data is 3D data
    if image.data.shape != mask.data.shape:
        raise VisualizationError(
            f"Image/mask shape mismatch: {image.data.shape} versus {mask.data.shape}."
        )#check if the array shapes same
    if not np.allclose(image.metadata.affine, mask.metadata.affine, rtol=0.0, atol=1e-3):
        raise VisualizationError("Image and mask affines do not match; resample before overlaying.")
    #check if the spatial coordinates consistent

#select the slice with the most number lesions
def select_lesion_slice(mask: np.ndarray, *, axis: int = 2) -> int:
    data = np.asarray(mask)#make sure input is np array
    if data.ndim != 3:
        raise VisualizationError(f"Lesion mask must be 3-D; got {data.shape}.")
    if axis not in (0, 1, 2):
        raise VisualizationError("axis must be 0, 1, or 2")
    positive = np.isfinite(data) & (data > 0)
    #calculate number of lesion voxels in each slice.
    counts = positive.sum(axis=tuple(i for i in range(3) if i != axis))
    #check if the mask empty
    if not np.any(counts):
        raise VisualizationError("Lesion mask is empty; no lesion-containing slice exists.")
    return int(np.argmax(counts))

def _display_slice(volume: np.ndarray, index: int, axis: int) -> np.ndarray:
    return np.flipud(np.rot90(np.take(volume, index, axis=axis)))

def _intensity_limits(data: np.ndarray, modality: str) -> tuple[float, float]:
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        raise VisualizationError("Image contains no finite intensities.")
    if modality == "ct":#CT uses fixed window width and window level settings
        return -160.0, 240.0
    #PET calculated based on actual data
    lo, hi = np.percentile(finite, (1.0, 99.5))
    return float(lo), float(hi if hi > lo else lo + 1.0)

def _draw_image(ax, image: LoadedVolume, index: int, axis: int, modality: str, title: str) -> None:
    vmin, vmax = _intensity_limits(image.data, modality)
    ax.imshow(_display_slice(image.data, index, axis), cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")

def _draw_overlay(ax, image, mask, index, axis, modality, title) -> None:
    #draw the image, overlay lesion mask on same slice.
    _draw_image(ax, image, index, axis, modality, title)
    mask_slice = _display_slice(mask.data > 0, index, axis)
    #hide non-lesion pixels so the underlying image still visible
    ax.imshow(
        np.ma.masked_where(~mask_slice, mask_slice),
        cmap="autumn", alpha=0.55,
        interpolation="nearest", vmin=0, vmax=1,
    )
    ax.contour(mask_slice, levels=[0.5], colors=["lime"], linewidths=0.8)


def render_patient_comparison(
        pair: PatientPairVolumes, *, modality: str = "ct", axis: int = 2,
        baseline_slice: int | None = None, followup_slice: int | None = None,
        output_path: str | Path | None = None,
) -> tuple[Figure, PatientDisplayResult]:
    modality = modality.lower()
    if modality not in {"ct", "pet"}:
        raise VisualizationError("modality must be 'ct' or 'pet'")
    #load selected modality and lesion masks for both timepoints
    bl_image, fu_image = getattr(pair.baseline, modality), getattr(pair.followup, modality)
    bl_mask, fu_mask = pair.baseline.lesion_mask, pair.followup.lesion_mask
    if any(v is None for v in (bl_image, fu_image, bl_mask, fu_mask)):
        raise VisualizationError(
            f"Patient {pair.patient_id} needs BL/FU {modality.upper()} and lesion masks."
        )
    assert bl_image is not None and fu_image is not None and bl_mask is not None and fu_mask is not None
    #each image and its mask must share same shape and spatial geometry.
    validate_image_mask_alignment(bl_image, bl_mask)
    validate_image_mask_alignment(fu_image, fu_mask)
    #unless specified, select the slice with the most lesion voxels at each timepoint.
    bl_index = select_lesion_slice(bl_mask.data, axis=axis) if baseline_slice is None else baseline_slice
    fu_index = select_lesion_slice(fu_mask.data, axis=axis) if followup_slice is None else followup_slice
    #check if both slice indices are within bounds
    for name, index, size in (
            ("baseline", bl_index, bl_image.data.shape[axis]),
            ("follow-up", fu_index, fu_image.data.shape[axis]),
    ):
        if not 0 <= index < size:
            raise VisualizationError(
                f"{name} slice {index} is outside the valid range 0..{size - 1}."
            )
    #col(bl and fu), row(original image and lesion overlay)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    label = modality.upper()
    _draw_image(axes[0, 0], bl_image, bl_index, axis, modality, f"Baseline {label} — slice {bl_index}")
    _draw_overlay(axes[1, 0], bl_image, bl_mask, bl_index, axis, modality, f"Baseline {label} + lesion mask")
    _draw_image(axes[0, 1], fu_image, fu_index, axis, modality, f"Follow-up {label} — slice {fu_index}")
    _draw_overlay(axes[1, 1], fu_image, fu_mask, fu_index, axis, modality, f"Follow-up {label} + lesion mask")
    fig.suptitle(f"Cohort A patient {pair.patient_id}: baseline vs follow-up", fontsize=15)

    # save figure if output path provided
    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=160, bbox_inches="tight")
    # record lesion voxel counts for selected slice and full volume
    bl_positive, fu_positive = bl_mask.data > 0, fu_mask.data > 0
    result = PatientDisplayResult(
        pair.patient_id, modality,
        TimepointDisplay(
            "BL", bl_index,
            int(np.take(bl_positive, bl_index, axis=axis).sum()),
            int(bl_positive.sum()),
        ),
        TimepointDisplay(
            "FU", fu_index,
            int(np.take(fu_positive, fu_index, axis=axis).sum()),
            int(fu_positive.sum()),
        ),
        saved_path,
    )
    return fig, result