from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest
from src.cohort_a_loading import (
    LoadedVolume,
    PatientPairVolumes,
    TimepointVolumes,
    VolumeMetadata,
)
from src.lesion_visualization import (
    VisualizationError,
    _intensity_limits,
    render_patient_comparison,
    select_lesion_slice,
    validate_image_mask_alignment,
)

def make_volume(data: np.ndarray, affine: np.ndarray | None = None) -> LoadedVolume:
    """Create a small in-memory volume for unit tests."""
    affine = np.eye(4) if affine is None else affine
    image = nib.Nifti1Image(data, affine)
    metadata = VolumeMetadata(
        path=Path("memory.nii.gz"),
        shape=data.shape,
        spacing_mm=(1.0, 1.0, 1.0),
        affine=affine,
        orientation=("R", "A", "S"),
        dtype=str(data.dtype),
    )
    return LoadedVolume(data=data, image=image, metadata=metadata)

def make_patient_pair() -> PatientPairVolumes:
    bl_image = make_volume(np.arange(60, dtype=float).reshape(3, 4, 5))
    fu_image = make_volume(np.arange(60, dtype=float).reshape(3, 4, 5) + 10)
    bl_mask_data = np.zeros((3, 4, 5), dtype=np.uint8)
    bl_mask_data[0:2, 0:2, 3] = 1
    fu_mask_data = np.zeros((3, 4, 5), dtype=np.uint8)
    fu_mask_data[1:3, 1:4, 1] = 1

    baseline = TimepointVolumes(
        timepoint="BL",
        scan_id="scan_bl",
        ct=bl_image,
        pet=None,
        lesion_mask=make_volume(bl_mask_data),
    )
    followup = TimepointVolumes(
        timepoint="FU",
        scan_id="scan_fu",
        ct=fu_image,
        pet=None,
        lesion_mask=make_volume(fu_mask_data),
    )
    return PatientPairVolumes("patient_01", baseline, followup)

def test_select_lesion_slice_uses_the_slice_with_most_voxels() -> None:
    mask = np.zeros((3, 4, 5), dtype=float)
    mask[:, 0, 1] = 1
    mask[:, :3, 4] = 1
    mask[0, 0, 2] = np.nan
    assert select_lesion_slice(mask) == 4

def test_select_lesion_slice_supports_other_axes() -> None:
    mask = np.zeros((4, 3, 2), dtype=np.uint8)
    mask[2, :, :] = 1
    assert select_lesion_slice(mask, axis=0) == 2

@pytest.mark.parametrize(
    ("mask", "axis", "message"),
    [
        (np.zeros((3, 3)), 1, "3-D"),
        (np.zeros((3, 3, 3)), 3, "axis"),
        (np.zeros((3, 3, 3)), 2, "empty"),
    ],
)
def test_select_lesion_slice_rejects_invalid_input(mask, axis, message) -> None:
    with pytest.raises(VisualizationError, match=message):
        select_lesion_slice(mask, axis=axis)

def test_validate_alignment_rejects_shape_mismatch() -> None:
    image = make_volume(np.zeros((3, 4, 5)))
    mask = make_volume(np.zeros((3, 4, 6)))

    with pytest.raises(VisualizationError, match="shape mismatch"):
        validate_image_mask_alignment(image, mask)


def test_validate_alignment_rejects_different_affines() -> None:
    image = make_volume(np.zeros((3, 4, 5)))
    shifted_affine = np.eye(4)
    shifted_affine[0, 3] = 2.0
    mask = make_volume(np.zeros((3, 4, 5)), shifted_affine)

    with pytest.raises(VisualizationError, match="affines"):
        validate_image_mask_alignment(image, mask)

def test_intensity_limits_use_ct_window_and_pet_percentiles() -> None:
    data = np.arange(100, dtype=float)
    assert _intensity_limits(data, "ct") == (-160.0, 240.0)
    assert _intensity_limits(data, "pet") == pytest.approx(
        tuple(np.percentile(data, (1.0, 99.5)))
    )

def test_intensity_limits_reject_image_without_finite_values() -> None:
    with pytest.raises(VisualizationError, match="no finite"):
        _intensity_limits(np.array([np.nan, np.inf]), "pet")

def test_render_patient_comparison_saves_figure_and_summary(tmp_path) -> None:
    output_path = tmp_path / "figures" / "patient_01.png"

    fig, result = render_patient_comparison(
        make_patient_pair(),
        output_path=output_path,
    )
    try:
        #automatically selecting should find lesion-heavy slices
        assert result.baseline.slice_index == 3
        assert result.followup.slice_index == 1
        assert result.baseline.total_lesion_voxels == 4
        assert result.followup.total_lesion_voxels == 6
        assert result.output_path == output_path
        assert output_path.stat().st_size > 0
        assert len(fig.axes) == 4
    finally:
        plt.close(fig)

def test_render_rejects_missing_modality() -> None:
    with pytest.raises(VisualizationError, match="needs BL/FU PET"):
        render_patient_comparison(make_patient_pair(), modality="pet")


def test_render_rejects_slice_outside_volume() -> None:
    with pytest.raises(VisualizationError, match="outside the valid range"):
        render_patient_comparison(make_patient_pair(), baseline_slice=5)
















