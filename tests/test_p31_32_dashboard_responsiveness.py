from __future__ import annotations
import csv
import os
from pathlib import Path
import re
import shutil
from time import perf_counter
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parents[1]

def test_p31_32_dashboard_responsiveness(tmp_path, monkeypatch, capsys):
    data_root = os.environ.get("P31_PERF_DATA_ROOT")
    if not data_root:
        pytest.skip("Set P31_PERF_DATA_ROOT to run the real-patient performance test")

    manifest = Path(os.environ.get(
        "P31_PERF_MANIFEST", ROOT / "outputs/cohort_b_subset/cohort_b_subset_pairs.csv"
    ))
    output_root = Path(os.environ.get(
        "P31_PERF_OUTPUT_ROOT", ROOT / "outputs/patients"
    ))
    pairs = pd.read_csv(manifest, dtype=str)
    patient = os.environ.get("P31_PERF_PATIENT", pairs.iloc[0]["patient_id"])
    selected = pairs[pairs["patient_id"] == patient]
    assert len(selected) == 1, f"Expected one manifest row for {patient}"
    for column in ("bl_ct_path", "fu_ct_path", "bl_lesion_mask_path", "fu_lesion_mask_path"):
        assert pd.notna(selected.iloc[0][column]), f"Missing {column}"
        assert (Path(data_root) / selected.iloc[0][column]).is_file(), column
    for name in ("registered_baseline_ct.nii.gz", "registered_baseline_lesion_mask.nii.gz",
                 "TransformParameters.0.txt"):
        assert (output_root / patient / name).is_file(), f"Prepare alignment first: {name}"

    patient_outputs = tmp_path / "patients"
    shutil.copytree(
        output_root / patient,
        patient_outputs / patient,
        ignore=shutil.ignore_patterns("lesion_matches.csv"),
    )
    test_manifest = tmp_path / "pairs.csv"
    selected.to_csv(test_manifest, index=False)
    monkeypatch.setenv("DATA_ROOT", str(Path(data_root).resolve()))
    monkeypatch.setenv("P31_PAIR_MANIFEST", str(test_manifest))
    monkeypatch.setenv("P31_PATIENT_OUTPUT_ROOT", str(patient_outputs))
    monkeypatch.setenv("P31_VIEWER_TIMING", "1")
    st.cache_data.clear()
    st.cache_resource.clear()
    app = AppTest.from_file(str(ROOT / "tools/align_longitudinal_patient_v5_mapped.py"))
    samples = []

    def run(action, plane, masks, index=None):
        capsys.readouterr()
        started = perf_counter()
        error = None
        try:
            app.run(timeout=120)
        except RuntimeError as exc:
            error = exc
        elapsed = perf_counter() - started
        output = capsys.readouterr().out
        timing = re.search(
            r"\[P31 viewer\].*?cache=(HIT|MISS) current_render=([\d.]+)ms.*?"
            r"fragment_python=([\d.]+)ms", output
        )
        samples.append({
            "action": action, "plane": plane, "masks": masks,
            "slice": index, "elapsed_s": elapsed,
            "cache": timing[1] if timing else "",
            "render_ms": float(timing[2]) if timing else "",
            "fragment_ms": float(timing[3]) if timing else "",
            "status": str(error) if error else "completed",
        })
        if error is not None:
            raise error
        assert not app.exception, [item.message for item in app.exception]
        assert not app.error, [item.value for item in app.error]
        assert len(app.slider) == 1, "Patient viewer did not load"
        slider = app.slider[0]
        assert slider.label == f"FU-space {plane.lower()} slice"
        if index is not None:
            assert slider.value == index
        assert app.checkbox[0].value == masks
        assert len(app.get("image")) >= 3, "Expected all three image panels"
        assert timing, "Missing viewer timing; rendering may have stopped early"
        samples[-1]["slice"] = slider.value

    try:
        run("load", "Axial", False)
        for plane in ("Axial", "Coronal", "Sagittal", "Axial"):
            plane_control = next(item for item in app.selectbox if item.label == "View plane")
            if plane_control.value != plane:
                plane_control.set_value(plane)
                run("plane", plane, app.checkbox[0].value)
            for masks in (True, False):
                app.checkbox[0].set_value(masks)
                run("overlay", plane, masks)
                maximum = app.slider[0].max
                centre = maximum // 2
                indices = [centre, centre + 1, centre + 2, centre + 3,
                           maximum, 0, maximum // 4, 3 * maximum // 4,
                           centre + 2, centre + 1, centre, 0]
                for index in indices:
                    index = min(maximum, index)
                    app.slider[0].set_value(index)
                    run("slice", plane, masks, index)

        slice_samples = [row for row in samples if row["action"] == "slice"]
        assert len(slice_samples) == 96
        assert {row["plane"] for row in slice_samples} == {
            "Axial", "Coronal", "Sagittal"
        }
        assert {row["masks"] for row in slice_samples} == {True, False}
        assert {row["cache"] for row in slice_samples} == {"HIT", "MISS"}
        assert all(row["status"] == "completed" for row in samples)
    finally:
        report = Path(os.environ.get("P31_PERF_REPORT", tmp_path / "p31_32_timings.csv"))
        if samples:
            report.parent.mkdir(parents=True, exist_ok=True)
            with report.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
                writer.writeheader()
                writer.writerows(samples)
            print(f"P31-32: {len(samples)} samples saved to {report}")
        st.cache_data.clear()
        st.cache_resource.clear()
