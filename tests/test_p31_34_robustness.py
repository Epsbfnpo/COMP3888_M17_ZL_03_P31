from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.cohort_a_loading import ManifestError, MissingImagingFileError, load_pair_manifest, load_patient_pair
from src.lesion_components import EmptyLesionMaskError, LesionMaskError, extract_individual_lesions
from src.lesion_features import extract_aligned_lesion_features
from src.lesion_pair_costs import LesionPairCostError, generate_lesion_pair_costs
from src.matching_dashboard import MatchingDashboardError, run_patient_match
from test_loading_and_lesions import make_manifest

@pytest.fixture
def case(tmp_path):
    root, manifest, patients = make_manifest(tmp_path)
    pair = load_patient_pair(manifest, patients[0], data_root=root, modalities=('ct', 'pet', 'lesion_mask'))
    return root, manifest, pair

@pytest.mark.parametrize('tp', ['bl', 'fu'])
@pytest.mark.parametrize('modality', ['ct', 'pet', 'lesion_mask'])
@pytest.mark.parametrize('missing', ['blank', 'absent'])
def test_missing_imaging_file(case, tp, modality, missing):
    root, manifest, pair = case
    table = pd.read_csv(manifest).fillna('')
    column = f'{tp}_{modality}_path'
    if missing == 'blank':
        table.loc[0, column] = ''
        table.to_csv(manifest, index=False)
    else:
        (root / table.loc[0, column]).unlink()
    with pytest.raises(MissingImagingFileError, match=f'{pair.patient_id} {tp.upper()} {modality}'):
        load_patient_pair(manifest, pair.patient_id, data_root=root, modalities=(modality,))

@pytest.mark.parametrize('column', ['patient_id', 'bl_scan_id', 'fu_scan_id', 'bl_ct_path', 'fu_ct_path',
                                   'bl_pet_path', 'fu_pet_path', 'bl_lesion_mask_path', 'fu_lesion_mask_path'])
def test_missing_manifest_column(case, column):
    _, manifest, _ = case
    pd.read_csv(manifest).drop(columns=column).to_csv(manifest, index=False)
    with pytest.raises(ManifestError, match=column):
        load_pair_manifest(manifest)

def test_empty_mask_is_detected_without_fake_lesions(case):
    _, _, pair = case
    empty = replace(pair.baseline.lesion_mask, data=np.zeros_like(pair.baseline.lesion_mask.data))
    with pytest.raises(EmptyLesionMaskError, match='no positive lesion voxels'):
        extract_individual_lesions(empty, patient_id=pair.patient_id, timepoint='BL')
    rows = extract_aligned_lesion_features(empty, pair.followup.lesion_mask,
                                         reference=pair.followup.ct, patient_id=pair.patient_id)
    assert len(rows) == 2
    assert {row['timepoint'] for row in rows} == {'FU'}


@pytest.mark.parametrize('value, message', [(np.nan, 'NaN'), (np.inf, 'infinite'),
                                          (-1, 'negative'), (0.4, 'non-integer'),
                                          (2**32 + 1, 'range')])
def test_invalid_mask(case, value, message):
    _, _, pair = case
    data = pair.baseline.lesion_mask.data.astype(float)
    data[0, 0, 0] = value
    with pytest.raises(LesionMaskError, match=message):
        extract_individual_lesions(replace(pair.baseline.lesion_mask, data=data),
                                   patient_id=pair.patient_id, timepoint='BL')


@pytest.mark.parametrize('role', ['mask', 'reference', 'pet'])
@pytest.mark.parametrize('problem', ['2d', '4d', 'shape', 'nan', 'singular', 'affine_row', 'shift'])
def test_invalid_geometry(case, role, problem):
    _, _, pair = case
    volume = {'mask': pair.baseline.lesion_mask, 'reference': pair.followup.ct, 'pet': pair.baseline.pet}[role]
    data, affine = volume.data.copy(), volume.metadata.affine.copy()
    if problem == '2d':
        data = data[:, :, 0]
    elif problem == '4d':
        data = data[..., None]
    elif problem == 'shape':
        data = data[:-1]
    elif problem == 'nan':
        affine[0, 0] = np.nan
    elif problem == 'singular':
        affine[0, 0] = 0
    elif problem == 'affine_row':
        affine[3, 0] = 1
    else:
        affine[0, 3] += 10
    invalid = replace(volume, data=data, metadata=replace(volume.metadata, affine=affine))
    with pytest.raises(ValueError, match='geometry|reference grid'):
        extract_aligned_lesion_features(
            invalid if role == 'mask' else pair.baseline.lesion_mask, pair.followup.lesion_mask,
            reference=invalid if role == 'reference' else pair.followup.ct,
            baseline_pet=invalid if role == 'pet' else pair.baseline.pet, patient_id=pair.patient_id,
        )


@pytest.mark.parametrize('column', ['patient_id', 'timepoint', 'lesion_id', 'volume_ml',
                                   'centroid_x_mm', 'centroid_y_mm', 'centroid_z_mm'])
@pytest.mark.parametrize('missing', ['column', 'value'])
def test_missing_lesion_features(case, column, missing):
    _, _, pair = case
    features = pd.DataFrame(extract_aligned_lesion_features(
        pair.baseline.lesion_mask, pair.followup.lesion_mask,
        reference=pair.followup.ct, patient_id=pair.patient_id,
    ))
    if missing == 'column':
        features = features.drop(columns=column)
    else:
        features.loc[0, column] = np.nan
    with pytest.raises(LesionPairCostError, match='column|finite|blank|BL/FU'):
        generate_lesion_pair_costs(features)


@pytest.mark.parametrize('csv, message', [
    ('wrong,FU1\nBL1,0.1\n', 'bl_lesion_id'),
    ('bl_lesion_id,FU1\nBL1,abc\n', 'not numeric'),
    ('bl_lesion_id,FU1\nBL1,\n', 'NaN'),
    ('bl_lesion_id,FU1\nBL1,-1\n', '>= 0'),
    ('bl_lesion_id,FU1\nBL1,-inf\n', '>= 0'),
    ('bl_lesion_id,FU1\n,0.1\n', 'blank'),
    ('bl_lesion_id,FU1\nBL1,0.1\nBL1,0.2\n', 'Duplicate'),
])
def test_invalid_matrix_preserves_results_and_allows_recovery(tmp_path, csv, message):
    matrix = tmp_path / 'matrix.csv'
    output = tmp_path / 'matches.csv'
    matrix.write_text('bl_lesion_id,FU1\nBL1,0.1\n')
    run_patient_match(matrix, 'good', output)
    before = output.read_bytes()
    matrix.write_text(csv)
    with pytest.raises(MatchingDashboardError, match=message):
        run_patient_match(matrix, 'bad', output)
    assert output.read_bytes() == before
    matrix.write_text('bl_lesion_id,FU1\nBL1,0.2\n')
    run_patient_match(matrix, 'bad', output)
    saved = pd.read_csv(output)
    assert set(saved.patient_id) == {'good', 'bad'}
    assert saved.loc[saved.patient_id == 'good', 'match_cost'].tolist() == [0.1]


@pytest.mark.parametrize('problem', ['missing_ct', 'unreadable_ct', 'manifest_column'])
def test_dashboard_reports_error_and_recovers(case, tmp_path, monkeypatch, problem):
    root, manifest, pair = case
    table = pd.read_csv(manifest)
    ct = root / table.loc[0, 'bl_ct_path']
    original = ct.read_bytes()
    original_manifest = manifest.read_bytes()
    if problem == 'missing_ct':
        ct.unlink()
    elif problem == 'unreadable_ct':
        ct.write_text('not a NIfTI volume')
    else:
        table.drop(columns='bl_ct_path').to_csv(manifest, index=False)
    monkeypatch.setenv('DATA_ROOT', str(root))
    monkeypatch.setenv('P31_PAIR_MANIFEST', str(manifest))
    monkeypatch.setenv('P31_PATIENT_OUTPUT_ROOT', str(tmp_path / 'patients'))
    st.cache_data.clear()
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'tools/align_longitudinal_patient_v5_mapped.py'))
        app.run(timeout=30)
        assert not app.exception
        expected = 'pair manifest' if problem == 'manifest_column' else 'CT' if problem == 'missing_ct' else 'NIfTI'
        assert any(expected in error.value for error in app.error)
        ct.write_bytes(original)
        # Refreshing the manifest invalidates the dashboard volume cache.
        manifest.write_bytes(original_manifest)
        st.cache_data.clear()
        app.run(timeout=30)
        assert not app.exception
        assert not app.error, [error.value for error in app.error]
        assert any(metric.value == pair.patient_id for metric in app.metric)
    finally:
        st.cache_data.clear()
        st.cache_resource.clear()
