from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
from src.cohort_a_loading import load_patient_pair
from src.lesion_visualization import (
    VisualizationError,
    render_patient_comparison,
    select_lesion_slice,
)
from tests.test_loading_and_lesions import make_manifest

def test_selects_slice_with_most_lesion_voxels():
    #create 2 lesion slices, slice 6 contains the larger lesion.
    mask = np.zeros((6, 7, 8), dtype=np.uint8)
    mask[1:3, 1:3, 2] = 1
    mask[1:5, 1:5, 6] = 1
    #function should select the slice with the most lesion voxels
    assert select_lesion_slice(mask) == 6

def test_empty_mask_has_clear_error():
    #an empty mask should raise a clear visualization error
    with pytest.raises(VisualizationError, match="empty"):
        select_lesion_slice(np.zeros((3, 3, 3)))

def test_renders_baseline_followup_overlay(tmp_path: Path):
    #create temporary test images and load 1 patient pair
    root, manifest, ids = make_manifest(tmp_path)
    pair = load_patient_pair(
        manifest,
        ids[0],
        data_root=root,
        modalities=("ct", "lesion_mask"),
    )
    output = tmp_path / "comparison.png"
    #render and save the bl/fu comparison
    fig, result = render_patient_comparison(
        pair,
        output_path=output,
    )
    #check the image saved and contains four panels
    assert output.exists()
    assert output.stat().st_size > 0
    assert len(fig.axes) == 4
    #both selected slices should contain lesion voxels
    assert result.baseline.lesion_voxels_on_slice > 0
    assert result.followup.lesion_voxels_on_slice > 0

def test_rejects_misaligned_mask(tmp_path: Path):
    #create temporary test data
    root, manifest, ids = make_manifest(tmp_path)
    #shift baseline mask affine to simulate spatial misalignment
    path = root / "inputsTr" / f"{ids[0]}_BL_mask_00.nii.gz"
    image = nib.load(path)
    affine = image.affine.copy()
    affine[0, 3] += 5
    #save the modified mask back to disk
    nib.save(
        nib.Nifti1Image(
            np.asanyarray(image.dataobj),
            affine,
        ),
        path,
    )
    pair = load_patient_pair(
        manifest,
        ids[0],
        data_root=root,
        modalities=("ct", "lesion_mask"),
    )
    #rendering should fail since the image and mask affines differ
    with pytest.raises(VisualizationError, match="affines"):
        render_patient_comparison(pair)