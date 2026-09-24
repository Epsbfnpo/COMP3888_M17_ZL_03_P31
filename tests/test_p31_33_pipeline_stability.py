from __future__ import annotations
import hashlib
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.cohort_a_loading import load_pair_manifest
from src.lesion_pair_costs import generate_lesion_pair_costs, export_cost_matrices, export_pair_costs
from src.matching_dashboard import run_patient_match
from src.tracking_summary import build_tracking_summary_from_files
from tools import generate_aligned_lesion_features_batch as feature_batch
import itk
def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _check_volume(path, reference, *, labels=False):
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    assert image.shape == reference.shape, path
    np.testing.assert_allclose(image.affine, reference.affine, rtol=0, atol=1e-4)
    assert np.isfinite(data).all(), path
    if labels:
        assert (data >= 0).all(), path
        np.testing.assert_array_equal(data, np.rint(data))

def _compare_outputs(current, saved):
    for path in current:
        previous = saved / path.name
        if path.name.endswith(".nii.gz"):
            actual, expected = nib.load(path), nib.load(previous)
            np.testing.assert_allclose(actual.affine, expected.affine, rtol=0, atol=1e-6)
            if "labels" in path.name:
                np.testing.assert_array_equal(actual.dataobj, expected.dataobj)
            else:
                np.testing.assert_allclose(actual.dataobj, expected.dataobj, rtol=1e-6, atol=1e-4)
        elif path.suffix == ".csv":
            pd.testing.assert_frame_equal(
                pd.read_csv(path), pd.read_csv(previous), rtol=1e-6, atol=1e-6,
            )

def test_p31_33_pipeline_stability(tmp_path, monkeypatch):
    data_root = os.environ.get("P31_STABILITY_DATA_ROOT")
    if not data_root:
        pytest.skip("Set P31_STABILITY_DATA_ROOT, P31_STABILITY_MANIFEST and P31_STABILITY_PATIENTS")
    data_root = Path(data_root).expanduser().resolve()
    manifest_path = Path(os.environ["P31_STABILITY_MANIFEST"]).expanduser().resolve()
    patients = [value.strip() for value in os.environ["P31_STABILITY_PATIENTS"].split(",")]
    assert len(patients) >= 2 and len(set(patients)) == len(patients)
    assert all(value and Path(value).name == value and value not in (".", "..") for value in patients)
    manifest = load_pair_manifest(manifest_path)
    for patient in patients:
        selected = manifest[manifest.patient_id.astype(str) == patient]
        assert len(selected) == 1, f"Expected one manifest row for {patient}"
        for column in ("bl_ct_path", "fu_ct_path", "bl_lesion_mask_path", "fu_lesion_mask_path"):
            assert selected.iloc[0][column], f"{patient}: missing {column}"
            assert (data_root / selected.iloc[0][column]).is_file(), f"{patient}: {column}"

    previous_threads = itk.MultiThreaderBase.GetGlobalDefaultNumberOfThreads()
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(1)
    output_root = tmp_path / "patients"
    snapshots = tmp_path / "first_run"
    fingerprints = {}
    samples = []
    try:
        for run, order in enumerate((patients, list(reversed(patients))), start=1):
            for patient in order:
                started = perf_counter()
                sample = {"run": run, "patient_id": patient, "status": "FAILED", "elapsed_s": 0.0}
                samples.append(sample)
                patient_dir = output_root / patient
                features_path = patient_dir / "aligned_lesion_features.csv"
                status_path = patient_dir / "aligned_lesion_features_status.csv"
                monkeypatch.setattr(sys, "argv", [
                    "generate_aligned_lesion_features_batch.py",
                    "--pairs", str(manifest_path), "--data-root", str(data_root),
                    "--patient-ids", patient, "--expected-patients", "1",
                    "--registration-root", str(output_root),
                    "--aligned-mask-root", str(output_root / "aligned_masks"),
                    "--out", str(features_path), "--status-out", str(status_path),
                    "--rerun-registration", "--rerun-mask-transform",
                    "--disable-low-overlap-fallback",
                ])
                feature_batch.main()
                status = pd.read_csv(status_path)
                assert status.patient_id.tolist() == [patient]
                assert status.status.tolist() == ["OK"]
                assert not status.registration_reused.any(), "Registration must run again"

                features = pd.read_csv(features_path)
                assert set(features.patient_id) == {patient}
                assert set(features.timepoint) == {"BL", "FU"}, "Select patients with lesions at both timepoints"
                assert not features.duplicated(["timepoint", "lesion_id"]).any()
                assert np.isfinite(features[["volume_ml", "centroid_x_mm", "centroid_y_mm", "centroid_z_mm"]]).all().all()

                costs = generate_lesion_pair_costs(features)
                pair_cost_path = export_pair_costs(costs.pair_costs, patient_dir / "lesion_pair_costs.csv")
                matrix_paths = export_cost_matrices(costs.matrices, patient_dir / "lesion_pair_cost_matrices")
                matrix_path = patient_dir / "lesion_pair_cost_matrices" / f"{patient}_cost_matrix.csv"
                assert matrix_paths == [matrix_path]
                matrix = pd.read_csv(matrix_path, index_col=0)
                bl_ids = set(features.loc[features.timepoint == "BL", "lesion_id"])
                fu_ids = set(features.loc[features.timepoint == "FU", "lesion_id"])
                assert set(matrix.index) == bl_ids and set(matrix.columns) == fu_ids
                assert np.isfinite(matrix.to_numpy()).all()
                assert len(costs.pair_costs) == len(bl_ids) * len(fu_ids)

                matches_path = patient_dir / "lesion_matches.csv"
                run_patient_match(matrix_path, patient, matches_path)
                matches = pd.read_csv(matches_path)
                assert set(matches.patient_id) == {patient}
                assert set(matches.bl_lesion_id.dropna()) == bl_ids
                assert set(matches.fu_lesion_id.dropna()) == fu_ids
                assert not matches.duplicated(["bl_lesion_id", "fu_lesion_id", "match_type"]).any()
                assert np.isfinite(matches.match_cost).all()
                summary = build_tracking_summary_from_files(matches_path)
                summary_path = patient_dir / "tracking_patient_summary.csv"
                summary.to_csv(summary_path, index=False)
                assert summary.patient_id.tolist() == [patient, "ALL"]
                assert summary.bl_lesions.tolist() == [len(bl_ids)] * 2
                assert summary.fu_lesions.tolist() == [len(fu_ids)] * 2

                registered_ct = patient_dir / "registered_baseline_ct.nii.gz"
                transform = patient_dir / "TransformParameters.0.txt"
                aligned_mask = output_root / "aligned_masks" / patient / "baseline_component_labels_in_fu_space.nii.gz"
                row = manifest.loc[manifest.patient_id.astype(str) == patient].iloc[0]
                reference = nib.load(data_root / row.fu_ct_path)
                _check_volume(registered_ct, reference)
                _check_volume(aligned_mask, reference, labels=True)
                assert 'EulerTransform' in transform.read_text()
                paths = [registered_ct, transform, aligned_mask, features_path,
                         pair_cost_path, matrix_path, matches_path, summary_path]
                assert all(path.is_file() and path.stat().st_size for path in paths)

                for other, hashes in fingerprints.items():
                    if other != patient:
                        assert {path: _digest(path) for path in hashes} == hashes
                if run == 1:
                    saved = snapshots / patient
                    saved.mkdir(parents=True)
                    for path in paths:
                        shutil.copy2(path, saved / path.name)
                else:
                    _compare_outputs(paths, snapshots / patient)
                fingerprints[patient] = {path: _digest(path) for path in paths}
                sample.update(status="PASSED", elapsed_s=perf_counter() - started)
                print(f"P31-33 run {run}: {patient} PASSED ({sample['elapsed_s']:.1f}s)")
        assert len(samples) == 2 * len(patients)
    finally:
        itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(previous_threads)
        if samples and samples[-1]["status"] == "FAILED":
            samples[-1]["elapsed_s"] = perf_counter() - started
        report = Path(os.environ.get("P31_STABILITY_REPORT", tmp_path / "p31_33_runs.csv"))
        report.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(samples).to_csv(report, index=False)
        print(f"P31-33 run report: {report}")
