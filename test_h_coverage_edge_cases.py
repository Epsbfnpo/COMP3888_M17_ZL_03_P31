from __future__ import annotations

from pathlib import Path
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import cohort_a_loading as cal
from src import lesion_components as lc
from src import lesion_pair_costs as lpc
from src import multiplanar_view as mpv
from src import tracking_summary as ts


def _meta(path: str = "synthetic.nii.gz", shape=(2, 3, 4), spacing=(1.0, 1.0, 1.0), affine=None, orientation=("R", "A", "S")):
    affine = np.eye(4) if affine is None else np.asarray(affine, dtype=float)
    return cal.VolumeMetadata(
        path=Path(path),
        shape=tuple(shape),
        spacing_mm=tuple(spacing),
        affine=affine,
        orientation=tuple(orientation),
        dtype="float32",
    )


def _fake_header(zooms=(1.0, 1.0, 1.0), units="mm"):
    return SimpleNamespace(
        get_zooms=lambda: tuple(zooms),
        get_xyzt_units=lambda: (units, "unknown"),
    )


def _loaded(data=None, *, path="synthetic.nii.gz", spacing=(1.0, 1.0, 1.0), affine=None, orientation=("R", "A", "S"), image=None):
    if data is None:
        data = np.zeros((2, 3, 4), dtype=np.float32)
    data = np.asarray(data)
    affine = np.eye(4) if affine is None else np.asarray(affine, dtype=float)
    if image is None:
        image = SimpleNamespace(
            shape=data.shape,
            affine=affine,
            header=_fake_header(spacing),
            get_data_dtype=lambda: data.dtype,
            get_fdata=lambda dtype=np.float32: data.astype(dtype),
            dataobj=data,
        )
    return cal.LoadedVolume(
        data=data,
        image=image,
        metadata=_meta(path, data.shape, spacing, affine, orientation),
    )


def _manifest_row(**overrides):
    base = {
        "patient_id": "P1",
        "bl_scan_id": "P1_BL",
        "bl_ct_path": "bl_ct.nii.gz",
        "bl_pet_path": "bl_pet.nii.gz",
        "bl_lesion_mask_path": "bl_mask.nii.gz",
        "fu_scan_id": "P1_FU",
        "fu_ct_path": "fu_ct.nii.gz",
        "fu_pet_path": "fu_pet.nii.gz",
        "fu_lesion_mask_path": "fu_mask.nii.gz",
    }
    base.update(overrides)
    return pd.Series(base)


# ---------------------------------------------------------------------------
# cohort_a_loading edge/error branches
# ---------------------------------------------------------------------------

def test_loading_manifest_errors_and_patient_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(cal.ManifestError, match="does not exist"):
        cal.load_pair_manifest(tmp_path / "missing.csv")

    bad = tmp_path / "bad.csv"
    bad.write_text("x", encoding="utf-8")
    monkeypatch.setattr(pd, "read_csv", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(cal.ManifestError, match="Could not read"):
        cal.load_pair_manifest(bad)

    frame = pd.DataFrame({"patient_id": ["p"]})
    with pytest.raises(cal.ManifestError, match="missing required column"):
        # bypass the read_csv monkeypatch by writing through a one-shot replacement
        monkeypatch.setattr(pd, "read_csv", lambda *_a, **_k: frame)
        cal.load_pair_manifest(bad)

    manifest = pd.DataFrame({"patient_id": ["a", "a"]})
    with pytest.raises(cal.ManifestError, match="not found"):
        cal.get_patient_row(manifest, "missing")
    with pytest.raises(cal.ManifestError, match="appears 2 times"):
        cal.get_patient_row(manifest, "a")


def test_resolve_manifest_path_absolute_manifest_relative_env_and_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    absolute = tmp_path / "absolute.nii.gz"
    assert cal.resolve_manifest_path(absolute, manifest_path=tmp_path / "m.csv") == absolute

    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    rel = manifest_dir / "inside.nii.gz"
    rel.write_text("x")
    monkeypatch.delenv("COHORT_A_ROOT", raising=False)
    assert cal.resolve_manifest_path("inside.nii.gz", manifest_path=manifest_dir / "m.csv") == rel.resolve()

    root = tmp_path / "root"
    root.mkdir()
    rooted = root / "rooted.nii.gz"
    rooted.write_text("x")
    monkeypatch.setenv("COHORT_A_ROOT", str(root))
    assert cal.resolve_manifest_path("rooted.nii.gz", manifest_path=manifest_dir / "m.csv") == rooted.resolve()

    missing = cal.resolve_manifest_path("not_there.nii.gz", manifest_path=manifest_dir / "m.csv", data_root=root)
    assert missing == (root / "not_there.nii.gz").resolve()
    with pytest.raises(cal.MissingImagingFileError, match="blank"):
        cal.resolve_manifest_path("   ", manifest_path=manifest_dir / "m.csv")


def test_metadata_and_nifti_loading_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    original_metadata = cal._metadata
    image = SimpleNamespace(
        shape=(2, 2, 2),
        header=SimpleNamespace(get_zooms=lambda: (1.0, 1.0), get_data_dtype=lambda: np.dtype("float32")),
        affine=np.eye(4),
        get_data_dtype=lambda: np.dtype("float32"),
    )
    with pytest.raises(cal.UnreadableImagingFileError, match="three spatial"):
        cal._metadata(image, Path("x.nii.gz"))

    with pytest.raises(cal.MissingImagingFileError, match="Missing test"):
        cal.load_nifti_volume(tmp_path / "missing.nii.gz", role="test")

    directory = tmp_path / "dir"
    directory.mkdir()
    with pytest.raises(cal.MissingImagingFileError, match="Expected test to be a file"):
        cal.load_nifti_volume(directory, role="test")

    file_path = tmp_path / "x.nii.gz"
    file_path.write_text("placeholder")
    monkeypatch.setattr(cal.nib, "load", lambda *_: (_ for _ in ()).throw(ValueError("broken nifti")))
    with pytest.raises(cal.UnreadableImagingFileError, match="Could not read test"):
        cal.load_nifti_volume(file_path, role="test")

    # A CohortALoadError raised inside the try block is intentionally re-raised unchanged.
    fake = SimpleNamespace(
        dataobj=np.zeros((2, 2, 2)),
        get_fdata=lambda dtype=np.float32: np.zeros((2, 2, 2), dtype=dtype),
    )
    monkeypatch.setattr(cal.nib, "load", lambda *_: fake)
    monkeypatch.setattr(cal, "_metadata", lambda *_: (_ for _ in ()).throw(cal.UnreadableImagingFileError("metadata bad")))
    with pytest.raises(cal.UnreadableImagingFileError, match="metadata bad"):
        cal.load_nifti_volume(file_path)

    # Restore metadata and make a 2-D image to exercise the post-load dimensionality check.
    fake2 = SimpleNamespace(
        shape=(2, 2),
        affine=np.eye(4),
        header=SimpleNamespace(get_zooms=lambda: (1.0, 1.0, 1.0)),
        get_data_dtype=lambda: np.dtype("float32"),
        get_fdata=lambda dtype=np.float32: np.zeros((2, 2), dtype=dtype),
        dataobj=np.zeros((2, 2)),
    )
    monkeypatch.setattr(cal.nib, "load", lambda *_: fake2)
    monkeypatch.setattr(cal.nib, "aff2axcodes", lambda *_: ("R", "A", "S"))
    monkeypatch.setattr(cal, "_metadata", original_metadata)
    with pytest.raises(cal.UnreadableImagingFileError, match="expected at least 3 dimensions"):
        cal.load_nifti_volume(file_path)


def test_load_nifti_preserve_dtype_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "ok.nii.gz"
    path.write_text("placeholder")
    data = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
    fake = SimpleNamespace(
        shape=data.shape,
        affine=np.diag([2.0, 3.0, 4.0, 1.0]),
        header=SimpleNamespace(get_zooms=lambda: (2.0, 3.0, 4.0)),
        get_data_dtype=lambda: data.dtype,
        get_fdata=lambda dtype=np.float32: data.astype(dtype),
        dataobj=data,
    )
    monkeypatch.setattr(cal.nib, "load", lambda *_: fake)
    monkeypatch.setattr(cal.nib, "aff2axcodes", lambda *_: ("R", "A", "S"))
    vol = cal.load_nifti_volume(path, preserve_dtype=True)
    assert vol.data.dtype == np.int16
    assert cal.volume_summary(None) is None
    summary = cal.volume_summary(vol)
    assert summary["shape"] == [2, 2, 2]
    assert summary["spacing_mm"] == [2.0, 3.0, 4.0]
    assert summary["orientation"] == ["R", "A", "S"]
    assert summary["affine"][0][0] == 2.0


def test_load_timepoint_validates_inputs_and_missing_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    row = _manifest_row(bl_ct_path="")
    with pytest.raises(ValueError, match="timepoint"):
        cal.load_timepoint_from_row(row, timepoint="MID", manifest_path=tmp_path / "m.csv")
    with pytest.raises(ValueError, match="Unknown modality"):
        cal.load_timepoint_from_row(row, timepoint="BL", manifest_path=tmp_path / "m.csv", modalities=("ct", "x"))

    # blank required path
    with pytest.raises(cal.MissingImagingFileError, match="bl_ct_path.*blank"):
        cal.load_timepoint_from_row(row, timepoint="BL", manifest_path=tmp_path / "m.csv", modalities=("ct",))

    # blank optional path is tolerated
    tp = cal.load_timepoint_from_row(row, timepoint="BL", manifest_path=tmp_path / "m.csv", modalities=("ct",), allow_missing=True)
    assert tp.ct is None

    # non-blank unresolved path gets the detailed resolution error
    row2 = _manifest_row(bl_ct_path="missing.nii.gz")
    with pytest.raises(cal.MissingImagingFileError, match="could not be resolved"):
        cal.load_timepoint_from_row(row2, timepoint="BL", manifest_path=tmp_path / "m.csv", modalities=("ct",))

    # Successful path and preserve_dtype flag for lesion masks.
    seen = []
    monkeypatch.setattr(cal, "resolve_manifest_path", lambda raw_path, **_k: Path(raw_path))
    def fake_load(path, *, role, preserve_dtype=False):
        seen.append((str(path), role, preserve_dtype))
        return _loaded(path=str(path))
    monkeypatch.setattr(cal, "load_nifti_volume", fake_load)
    tp = cal.load_timepoint_from_row(_manifest_row(), timepoint="FU", manifest_path=tmp_path / "m.csv", modalities=("ct", "lesion_mask"))
    assert tp.timepoint == "FU" and tp.scan_id == "P1_FU"
    assert seen[-1][2] is True


def test_load_patient_pair_delegates_both_timepoints(monkeypatch: pytest.MonkeyPatch):
    manifest = pd.DataFrame([_manifest_row()])
    monkeypatch.setattr(cal, "load_pair_manifest", lambda *_: manifest)
    monkeypatch.setattr(cal, "get_patient_row", lambda *_: manifest.iloc[0])
    calls=[]
    def fake_tp(row, *, timepoint, **kwargs):
        calls.append(timepoint)
        return cal.TimepointVolumes(timepoint, f"{timepoint}_scan", None, None, None)
    monkeypatch.setattr(cal, "load_timepoint_from_row", fake_tp)
    pair = cal.load_patient_pair("m.csv", "P1", modalities=("ct",))
    assert calls == ["BL", "FU"]
    assert pair.patient_id == "P1"


# ---------------------------------------------------------------------------
# lesion_components one defensive branch that ordinary data cannot hit
# ---------------------------------------------------------------------------

def test_components_lesion_count_and_empty_component_defensive_branch(monkeypatch: pytest.MonkeyPatch):
    vol = _loaded(np.ones((2, 2, 2), dtype=np.uint8))
    monkeypatch.setattr(lc, "_temporary_components", lambda *_a, **_k: (np.zeros((2, 2, 2), dtype=np.int32), [(1, 1)]))
    result = lc.extract_individual_lesions(vol, patient_id="P", timepoint="bl")
    assert result.lesion_count == 0


# ---------------------------------------------------------------------------
# lesion_pair_costs edge/error branches
# ---------------------------------------------------------------------------

def _lesion_row(patient="P", tp="BL", lesion_id="L1", xyz=(0.0, 0.0, 0.0), **extra):
    row = {
        "patient_id": patient,
        "timepoint": tp,
        "lesion_id": lesion_id,
        "centroid_x_mm": xyz[0],
        "centroid_y_mm": xyz[1],
        "centroid_z_mm": xyz[2],
        "volume_ml": 1.0,
        "coordinate_space": "FU_RAS_mm",
    }
    row.update(extra)
    return row


def test_pair_cost_feature_table_validation_edge_cases():
    with pytest.raises(lpc.LesionPairCostError, match="lesion_id"):
        lpc._normalise_feature_table(pd.DataFrame([{"patient_id":"P","timepoint":"BL","centroid_x_mm":0,"centroid_y_mm":0,"centroid_z_mm":0}]))

    with pytest.raises(lpc.LesionPairCostError, match="missing required"):
        lpc._normalise_feature_table(pd.DataFrame([{"patient_id":"P","lesion_id":"L","timepoint":"BL"}]))

    empty = pd.DataFrame(columns=["patient_id","timepoint","lesion_id","centroid_x_mm","centroid_y_mm","centroid_z_mm"])
    assert lpc._normalise_feature_table(empty).empty

    for col in ("patient_id", "lesion_id", "timepoint"):
        row = _lesion_row()
        row[col] = np.nan
        with pytest.raises(lpc.LesionPairCostError, match=col):
            lpc._normalise_feature_table(pd.DataFrame([row]))

    for col in ("patient_id", "lesion_id"):
        row = _lesion_row()
        row[col] = "   "
        with pytest.raises(lpc.LesionPairCostError, match=col):
            lpc._normalise_feature_table(pd.DataFrame([row]))


def test_pair_cost_csv_loader_errors_and_concat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(lpc.LesionPairCostError, match="At least one"):
        lpc.load_aligned_feature_csvs([])
    with pytest.raises(FileNotFoundError, match="does not exist"):
        lpc.load_aligned_feature_csvs([tmp_path / "missing.csv"])

    path = tmp_path / "x.csv"; path.write_text("x")
    monkeypatch.setattr(pd, "read_csv", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad csv")))
    with pytest.raises(lpc.LesionPairCostError, match="Could not read"):
        lpc.load_aligned_feature_csvs([path])

    f1 = tmp_path / "a.csv"; f2 = tmp_path / "b.csv"
    pd.DataFrame([_lesion_row("P1","BL","B")]).to_csv(f1,index=False)
    pd.DataFrame([_lesion_row("P1","FU","F")]).to_csv(f2,index=False)
    monkeypatch.undo()
    table=lpc.load_aligned_feature_csvs([f1,f2])
    assert len(table)==2


def test_pair_cost_small_helpers_and_pet_comparability():
    assert lpc._fractional_difference(0,0) == 0.0
    assert lpc._coerce_optional_number("bad") is None
    assert lpc._coerce_optional_number(np.inf) is None
    assert lpc._coerce_optional_number("2.5") == 2.5

    bl=pd.Series({"suvmax":1.0}); fu=pd.Series({"suvmax":2.0})
    assert lpc._pet_values_are_comparable(bl,fu,"none") == (None,None,False)
    assert lpc._pet_values_are_comparable(bl,fu,"suvmean") == (None,None,False)
    bl2=pd.Series({"suvmax":"bad"}); assert lpc._pet_values_are_comparable(bl2,fu,"suvmax")[2] is False
    raw_bl=pd.Series({"pet_mean":1.0}); raw_fu=pd.Series({"pet_mean":2.0})
    assert lpc._pet_values_are_comparable(raw_bl,raw_fu,"pet_mean")[2] is False


def test_pair_cost_size_validation_and_empty_generation(tmp_path: Path):
    no_volume=pd.DataFrame([_lesion_row()]).drop(columns=["volume_ml"])
    lpc._validate_size_features(no_volume, lpc.PairCostConfig(size_weight=0))
    with pytest.raises(lpc.LesionPairCostError, match="requires.*volume_ml"):
        lpc._validate_size_features(no_volume, lpc.PairCostConfig(size_weight=1))
    bad=pd.DataFrame([_lesion_row(volume_ml=np.nan)])
    with pytest.raises(lpc.LesionPairCostError, match="finite values > 0"):
        lpc._validate_size_features(bad, lpc.PairCostConfig(size_weight=1))

    empty=pd.DataFrame(columns=["patient_id","timepoint","lesion_id","centroid_x_mm","centroid_y_mm","centroid_z_mm","volume_ml"])
    result=lpc.generate_lesion_pair_costs(empty, lpc.PairCostConfig(size_weight=0,pet_weight=0))
    assert result.pair_costs.empty and result.matrices=={}

    out=lpc.export_pair_costs(result.pair_costs,tmp_path/"nested"/"pairs.csv")
    assert out.exists()
    assert lpc._safe_filename("...///") == "patient"


def test_pair_cost_allows_missing_coordinate_space_metadata_when_not_provided():
    features = pd.DataFrame([
        _lesion_row("P", "BL", "B", (0, 0, 0)),
        _lesion_row("P", "FU", "F", (1, 0, 0)),
    ]).drop(columns=["coordinate_space"])
    result = lpc.generate_lesion_pair_costs(
        features,
        lpc.PairCostConfig(size_weight=0, pet_weight=0),
    )
    assert len(result.pair_costs) == 1


def test_pair_cost_matrix_empty_candidate_adds_missing_rank_column():
    features=pd.DataFrame([
        _lesion_row("P","BL","B",(0,0,0)),
        _lesion_row("P","FU","F",(100,0,0)),
    ])
    result=lpc.generate_lesion_pair_costs(features,lpc.PairCostConfig(size_weight=0,pet_weight=0,max_distance_mm=1))
    assert result.pair_costs.empty
    assert "candidate_rank_for_bl" in result.pair_costs.columns


# ---------------------------------------------------------------------------
# multiplanar_view branches including resampling
# ---------------------------------------------------------------------------

def test_multiplanar_invalid_axis_orientation_spacing_and_slice():
    vol=_loaded()
    with pytest.raises(mpv.MultiplanarViewError, match="unique"):
        mpv.anatomical_axis(("R","L","S"),"Sagittal")
    with pytest.raises(mpv.MultiplanarViewError, match="Expected a 3-D"):
        mpv.extract_display_slice(np.zeros((2,2)),axis=0,index=0)
    with pytest.raises(mpv.MultiplanarViewError, match="axis must"):
        mpv.extract_display_slice(np.zeros((2,2,2)),axis=9,index=0)
    with pytest.raises(IndexError):
        mpv.slice_centre_world_mm(vol,plane="Axial",index=999)

    bad_spacing=_loaded(spacing=(1.0,0.0,1.0))
    with pytest.raises(mpv.MultiplanarViewError, match="Invalid voxel spacing"):
        mpv.display_pixel_spacing_mm(bad_spacing,"Axial")


def test_physical_resize_validation_noop_and_float_fallback():
    with pytest.raises(mpv.MultiplanarViewError, match="2-D grayscale"):
        mpv.physical_aspect_resize(np.zeros((2,2,2,2)),row_spacing_mm=1,column_spacing_mm=1)
    with pytest.raises(mpv.MultiplanarViewError, match="RGB/RGBA"):
        mpv.physical_aspect_resize(np.zeros((2,2,2),dtype=np.uint8),row_spacing_mm=1,column_spacing_mm=1)
    with pytest.raises(mpv.MultiplanarViewError, match="positive finite"):
        mpv.physical_aspect_resize(np.zeros((2,2)),row_spacing_mm=np.nan,column_spacing_mm=1)
    with pytest.raises(mpv.MultiplanarViewError, match="max_side"):
        mpv.physical_aspect_resize(np.zeros((2,2)),row_spacing_mm=1,column_spacing_mm=1,max_side=0)

    empty=np.zeros((0,3),dtype=np.uint8)
    assert mpv.physical_aspect_resize(empty,row_spacing_mm=1,column_spacing_mm=1).shape==(0,3)

    # allow_upscale ratio < 1 branch
    wide=np.zeros((10,20,3),dtype=np.uint8)
    resized=mpv.physical_aspect_resize(wide,row_spacing_mm=1,column_spacing_mm=1,max_side=40,allow_upscale=True)
    assert resized.shape[:2]==(20,40)

    # allow_upscale=False, raw_ratio > ratio branch and exact no-op branch.
    arr=np.zeros((20,10,3),dtype=np.uint8)
    shrunk=mpv.physical_aspect_resize(arr,row_spacing_mm=1,column_spacing_mm=2,max_side=100,allow_upscale=False)
    assert shrunk.shape[0] <= 20 and shrunk.shape[1] <= 10
    same=np.zeros((10,10),dtype=np.uint8)
    returned=mpv.physical_aspect_resize(same,row_spacing_mm=1,column_spacing_mm=1,max_side=10,allow_upscale=False)
    assert returned.shape==(10,10) and returned.flags.c_contiguous

    # Non-uint8 path uses scipy.ndimage.zoom.
    flt=np.arange(12,dtype=np.float32).reshape(3,4)
    out=mpv.physical_aspect_resize(flt,row_spacing_mm=2,column_spacing_mm=1,max_side=12,allow_upscale=True)
    assert out.dtype==np.float32 and out.flags.c_contiguous


def test_geometry_matches_and_resample_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ref=_loaded(np.zeros((2,3,4),dtype=np.float32),path="ref.nii.gz")
    same=_loaded(np.ones((2,3,4),dtype=np.float32),path="same.nii.gz")
    assert mpv.geometry_matches(ref,same)
    shifted_aff=np.eye(4); shifted_aff[0,3]=2
    moving=_loaded(np.ones((2,3,4),dtype=np.float32),path="moving.nii.gz",affine=shifted_aff)
    assert not mpv.geometry_matches(moving,ref)
    assert mpv.resample_to_reference(same,ref) is same
    with pytest.raises(mpv.MultiplanarViewError, match="between 0 and 5"):
        mpv.resample_to_reference(moving,ref,order=9)

    monkeypatch.setattr(mpv,"resample_from_to",lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError("resample bad")))
    with pytest.raises(mpv.MultiplanarViewError, match="Could not resample"):
        mpv.resample_to_reference(moving,ref)

    result_data=np.ones((2,3,4),dtype=np.float32)*7
    result_img=SimpleNamespace(
        shape=result_data.shape,
        affine=np.diag([1.5,2.0,2.5,1.0]),
        header=SimpleNamespace(get_zooms=lambda:(1.5,2.0,2.5)),
        get_fdata=lambda dtype=np.float32:result_data.astype(dtype),
        get_data_dtype=lambda:result_data.dtype,
    )
    monkeypatch.setattr(mpv,"resample_from_to",lambda *_a,**_k:result_img)
    monkeypatch.setattr(mpv.nib,"aff2axcodes",lambda *_:("R","A","S"))
    inmem=mpv.resample_to_reference(moving,ref,role="PET")
    assert inmem.metadata.path==Path("<PET>")
    assert inmem.metadata.spacing_mm==(1.5,2.0,2.5)

    saved={}
    monkeypatch.setattr(mpv.nib,"save",lambda image,path:saved.update(path=path,image=image))
    expected=_loaded(path="saved.nii.gz")
    monkeypatch.setattr(mpv,"load_nifti_volume",lambda path,role: expected)
    out=mpv.resample_to_reference(moving,ref,output_path=tmp_path/"nested"/"saved.nii.gz",role="PET")
    assert out is expected and (tmp_path/"nested").is_dir() and saved["path"].endswith("saved.nii.gz")


# ---------------------------------------------------------------------------
# tracking_summary loaders, validation and aggregate fallbacks
# ---------------------------------------------------------------------------

def _matches_df():
    return pd.DataFrame([
        {"patient_id":"p1","bl_lesion_id":"b1","fu_lesion_id":"f1","match_type":"MATCHED"},
        {"patient_id":"p1","bl_lesion_id":"b2","fu_lesion_id":None,"match_type":"DISAPPEARING"},
        {"patient_id":"p1","bl_lesion_id":None,"fu_lesion_id":"f2","match_type":"NEW"},
    ])


def test_tracking_small_helpers_cover_invalid_and_null_values():
    assert ts._normalise_id(None) is None
    assert ts._normalise_id(pd.NA) is None
    assert ts._normalise_id(" nan ") is None
    assert ts._normalise_id(" id ")=="id"
    assert pd.isna(ts._as_optional_int(None))
    assert pd.isna(ts._as_optional_int("bad"))
    assert ts._as_optional_int("2.9")==2
    assert math.isnan(ts._as_optional_float(None))
    assert math.isnan(ts._as_optional_float("bad"))
    assert math.isnan(ts._as_optional_float(np.inf))
    assert ts._as_optional_float("1.5")==1.5


def test_tracking_loaders_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(FileNotFoundError,match="Matching results"):
        ts.load_tracking_matches(tmp_path/"missing.csv")
    with pytest.raises(FileNotFoundError,match="Evaluation summary"):
        ts.load_evaluation_summary(tmp_path/"missing_eval.csv")

    p=tmp_path/"x.csv"; p.write_text("x")
    monkeypatch.setattr(pd,"read_csv",lambda *_a,**_k: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(ts.TrackingSummaryError,match="Could not read matching"):
        ts.load_tracking_matches(p)
    with pytest.raises(ts.TrackingSummaryError,match="Could not read evaluation"):
        ts.load_evaluation_summary(p)
    monkeypatch.undo()

    pd.DataFrame([{"patient_id":"p"}]).to_csv(p,index=False)
    with pytest.raises(ts.TrackingSummaryError,match="missing required"):
        ts.load_tracking_matches(p)
    with pytest.raises(ts.TrackingSummaryError,match="missing required"):
        ts.load_evaluation_summary(p)

    pd.DataFrame([{"patient_id":" ","bl_lesion_id":"b","fu_lesion_id":"f","match_type":"MATCHED"}]).to_csv(p,index=False)
    with pytest.raises(ts.TrackingSummaryError,match="blank patient_id"):
        ts.load_tracking_matches(p)
    pd.DataFrame([{"patient_id":"p","bl_lesion_id":"b","fu_lesion_id":"f","match_type":" "}]).to_csv(p,index=False)
    with pytest.raises(ts.TrackingSummaryError,match="blank match_type"):
        ts.load_tracking_matches(p)

    evalp=tmp_path/"e.csv"
    required={"patient_id":" ","correct_matches":1,"incorrect_matches":0,"tracking_accuracy":1.0}
    pd.DataFrame([required]).to_csv(evalp,index=False)
    with pytest.raises(ts.TrackingSummaryError,match="blank patient_id"):
        ts.load_evaluation_summary(evalp)
    required["patient_id"]="p"
    pd.DataFrame([required,required]).to_csv(evalp,index=False)
    with pytest.raises(ts.TrackingSummaryError,match="duplicate"):
        ts.load_evaluation_summary(evalp)


def test_tracking_structural_and_evaluation_edge_paths():
    with pytest.raises(ts.TrackingSummaryError,match="No matching results"):
        ts._patient_structural_summary(pd.DataFrame(columns=list(ts.REQUIRED_MATCH_COLUMNS)),"p")

    blank=ts._evaluation_fields(None)
    assert blank["ground_truth_available"] is False
    row=pd.Series({"tracking_accuracy":0.5,"correct_matches":"2","incorrect_matches":"bad","error":np.nan,"evaluation_status":" "})
    out=ts._evaluation_fields(row)
    assert out["evaluation_status"]=="OK" and out["evaluation_error"]==""
    assert out["correct_matches"]==2 and pd.isna(out["incorrect_matches"])

    eval_df=pd.DataFrame([{"patient_id":"p","tracking_accuracy":1.0},{"patient_id":"p","tracking_accuracy":1.0}])
    with pytest.raises(ts.TrackingSummaryError,match="multiple rows"):
        ts._evaluation_row_for_patient(eval_df,"p")
    assert ts._evaluation_row_for_patient(None,"p") is None
    assert ts._evaluation_row_for_patient(pd.DataFrame(),"p") is None
    assert ts._evaluation_row_for_patient(pd.DataFrame([{"patient_id":"x"}]),"p") is None


def test_tracking_summary_validation_and_build_edge_paths(monkeypatch: pytest.MonkeyPatch):
    matches=_matches_df()
    with pytest.raises(ts.TrackingSummaryError,match="missing required"):
        ts.summarise_patient_tracking(matches.drop(columns=["match_type"]),"p1")
    with pytest.raises(ts.TrackingSummaryError,match="must not be blank"):
        ts.summarise_patient_tracking(matches,"   ")

    with pytest.raises(ts.TrackingSummaryError,match="missing required"):
        ts.build_tracking_summary(matches.drop(columns=["match_type"]))
    with pytest.raises(ts.TrackingSummaryError,match="empty"):
        ts.build_tracking_summary(matches.iloc[0:0])
    noids=matches.copy(); noids["patient_id"]=" "
    with pytest.raises(ts.TrackingSummaryError,match="no patient IDs"):
        ts.build_tracking_summary(noids)

    # include_overall=False branch
    frame=ts.build_tracking_summary(matches,include_overall=False)
    assert "ALL" not in set(frame["patient_id"])

    # Force a partial summary dict so the column-fill safety loop is exercised.
    monkeypatch.setattr(ts,"summarise_patient_tracking",lambda *_a,**_k:{"patient_id":"p1","bl_lesions":1})
    minimal=pd.DataFrame([{"patient_id":"p1","bl_lesion_id":"b","fu_lesion_id":"f","match_type":"MATCHED"}])
    filled=ts.build_tracking_summary(minimal,include_overall=False)
    assert set(ts.SUMMARY_COLUMNS)==set(filled.columns)
    assert pd.isna(filled.loc[0,"fu_lesions"])


def test_tracking_overall_evaluation_fallback_branches():
    ids=["p1","p2"]
    assert ts._overall_evaluation_fields(None,ids)["ground_truth_available"] is False

    dup_all=pd.DataFrame([
        {"patient_id":"ALL","correct_matches":1,"incorrect_matches":0,"tracking_accuracy":1.0},
        {"patient_id":"ALL","correct_matches":1,"incorrect_matches":0,"tracking_accuracy":1.0},
    ])
    with pytest.raises(ts.TrackingSummaryError,match="multiple ALL"):
        ts._overall_evaluation_fields(dup_all,ids)

    unrelated=pd.DataFrame([{"patient_id":"x","correct_matches":1,"incorrect_matches":0,"tracking_accuracy":1.0}])
    assert ts._overall_evaluation_fields(unrelated,ids)["ground_truth_available"] is False

    failed=pd.DataFrame([
        {"patient_id":"p1","correct_matches":None,"incorrect_matches":None,"tracking_accuracy":None,"evaluation_status":"FAILED"},
        {"patient_id":"p2","correct_matches":None,"incorrect_matches":None,"tracking_accuracy":None,"evaluation_status":"FAILED"},
    ])
    out=ts._overall_evaluation_fields(failed,ids)
    assert out["evaluation_status"]=="OK_PATIENTS=0/2"

    # Successful rows but no optional aggregate columns: numeric_sum missing-column path,
    # zero denominators and non-finite F1 path.
    minimal=pd.DataFrame([
        {"patient_id":"p1","correct_matches":0,"incorrect_matches":0,"tracking_accuracy":0.0,"evaluation_status":"OK"},
        {"patient_id":"p2","correct_matches":0,"incorrect_matches":0,"tracking_accuracy":0.0,"evaluation_status":"OK"},
    ])
    out=ts._overall_evaluation_fields(minimal,ids)
    assert pd.isna(out["ground_truth_links"])
    assert math.isnan(out["tracking_accuracy"]) and math.isnan(out["f1"])

    # Column exists but is all non-numeric -> numeric_sum returns NA.
    nonnumeric=minimal.copy(); nonnumeric["ground_truth_links"]=["bad","bad"]
    out=ts._overall_evaluation_fields(nonnumeric,ids)
    assert pd.isna(out["ground_truth_links"])


def test_tracking_build_from_files_without_optional_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    matchp=tmp_path/"m.csv"
    _matches_df().to_csv(matchp,index=False)
    out=ts.build_tracking_summary_from_files(matchp,evaluation_summary_path="",include_overall=False)
    assert list(out["patient_id"])==["p1"]

