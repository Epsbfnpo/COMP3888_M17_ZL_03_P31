from dataclasses import replace
import numpy as np
import nibabel as nib
import pytest
from src.cohort_a_loading import load_nifti_volume
from src.matching_dashboard import LesionLocation
from src.correspondence_visualization import selected_lesion_mask, lesion_display_slice, OUTCOME_STYLES

def volume(tmp_path, data, name='mask', affine=None):
    path = tmp_path / (name + '.nii.gz')
    nib.save(nib.Nifti1Image(data.astype(np.int16), np.eye(4) if affine is None else affine), path)
    return load_nifti_volume(path, preserve_dtype=True)

def test_selects_component_not_all_mask_voxels(tmp_path):
    data = np.zeros((12, 12, 12))
    data[1:3, 1:3, 1:3] = 1
    data[7:10, 7:10, 7:10] = 1
    v = volume(tmp_path, data)
    location = LesionLocation('p_BL_L002', 'BL', (8, 8, 8), 2)
    result = selected_lesion_mask(location, v, v)
    assert result.sum() == 27
    assert not result[1, 1, 1]
    for axis in (0, 1, 2):
        image, index = lesion_display_slice(result, axis)
        assert index in (7, 8, 9)
        assert image.sum() == 9
    with pytest.raises(ValueError, match='centroid disagree'):
        selected_lesion_mask(replace(location, centroid_voxel=(1, 1, 1)), v, v)

def test_preserves_resampled_native_ids_even_when_disconnected(tmp_path):
    data = np.zeros((10, 10, 10))
    data[1, 1, 1] = data[7, 7, 7] = 9
    v = volume(tmp_path, data)
    location = LesionLocation('p_BL_L009', 'BL', (4, 4, 4), 9, 18,
                             str(v.metadata.path), 'BL_native_component_IDs_rigidly_resampled_to_FU')
    result = selected_lesion_mask(location, v)
    assert result.sum() == 2
    image, index = lesion_display_slice(result, 2)
    assert image.sum() == 1  #centroid slice would be empty
    with pytest.raises(ValueError, match='does not contain'):
        selected_lesion_mask(replace(location, component_index=3), v)
    wrong_grid = volume(tmp_path, data, 'other', np.diag([2, 2, 2, 1]))
    with pytest.raises(ValueError, match='affines'):
        selected_lesion_mask(location, wrong_grid)

def test_outcomes_have_distinct_colours_and_explicit_labels():
    assert len({OUTCOME_STYLES[k][1] for k in ('MATCHED', 'NEW', 'DISAPPEARING')}) == 3
    assert 'no BL' in OUTCOME_STYLES['NEW'][0]
    assert 'no FU' in OUTCOME_STYLES['DISAPPEARING'][0]

@pytest.mark.parametrize('outcome,has_bl,has_fu', [('MATCHED', True, True), ('NEW', False, True), ('DISAPPEARING', True, False)])
def test_focus_renders_masks_ids_and_missing_side(tmp_path, monkeypatch, outcome, has_bl, has_fu):
    from contextlib import nullcontext
    from tools import align_longitudinal_patient_v4_mapped as dashboard
    from src.matching_dashboard import MatchImageFocus
    data = np.zeros((8, 8, 8))
    data[2:5, 2:5, 2:5] = 1
    v = volume(tmp_path, data)
    bl = LesionLocation('p_BL_L001', 'BL', (3, 3, 3), 1)
    fu = replace(bl, lesion_id='p_FU_L001', timepoint='FU')
    images, messages, errors = [], [], []
    monkeypatch.setattr(dashboard.st, 'columns', lambda n: [nullcontext() for _ in range(n)])
    for method in ('divider', 'subheader', 'caption'):
        monkeypatch.setattr(dashboard.st, method, lambda *a, **k: None)
    for method in ('markdown', 'info'):
        monkeypatch.setattr(dashboard.st, method, lambda text: messages.append(text))
    monkeypatch.setattr(dashboard.st, 'error', errors.append)
    monkeypatch.setattr(dashboard.st, 'image', lambda image, **kw: images.append(image))
    dashboard._render_lesion_focus(
        MatchImageFocus(outcome, bl if has_bl else None, fu if has_fu else None),
        registered_display=np.zeros_like(data, dtype=np.uint8),
        fu_display=np.zeros_like(data, dtype=np.uint8), fu_ct=v,
        modality='CT', plane_name='Axial', baseline_mask=v, followup_mask=v,
    )
    assert not errors
    assert len(images) == int(has_bl) + int(has_fu)
    assert all(image.max() > 0 for image in images)
    if has_bl:
        assert any('p_BL_L001' in text for text in messages)
    if has_fu:
        assert any('p_FU_L001' in text for text in messages)
    if not (has_bl and has_fu):
        assert any(text.startswith('No ') for text in messages)

def test_resolve_selected_bl_preserves_mask_metadata():
    import pandas as pd
    from src.matching_dashboard import resolve_match_image_focus
    rows = pd.DataFrame([
        dict(patient_id='p', timepoint=tp, lesion_id=f'p_{tp}_L001',
             centroid_x_vox=3, centroid_y_vox=3, centroid_z_vox=3,
             component_index=1, connectivity=26, mask_path=f'{tp}.nii.gz',
             mask_role='') for tp in ('BL', 'FU')
    ])
    focus = resolve_match_image_focus(pd.Series(dict(
        bl_lesion_id='p_BL_L001', fu_lesion_id='p_FU_L001', match_type='MATCHED'
    )), rows)
    assert focus.baseline.mask_path == 'BL.nii.gz'
    assert focus.followup.lesion_id == 'p_FU_L001'
    assert focus.followup.component_index == 1
    assert focus.followup.connectivity == 26
