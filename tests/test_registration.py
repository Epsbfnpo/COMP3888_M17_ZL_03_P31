"""Fast, isolated unit tests for src.registration (no real Elastix execution).

Run from the repository root:
    python -m pytest tests/test_registration.py -v
    python -m pytest tests/test_registration.py --cov=src.registration --cov-branch --cov-report=term-missing

Only synthetic CT data, tmp_path and mocked ITK/Elastix are used.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src import registration as reg


def volume(tmp_path, name="ct.nii.gz", shape=(12, 12, 24), *, affine=None,
           fill=100.0, spacing=(1.0, 1.0, 1.0), data=None, create=True):
    """Tiny LoadedVolume-compatible fixture; real NIfTI decoding is unnecessary."""
    path = tmp_path / name
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    if data is None:
        data = np.full(shape, fill, dtype=np.float32)
    if affine is None:
        affine = np.eye(4)
    return SimpleNamespace(
        data=np.asarray(data),
        metadata=SimpleNamespace(
            path=path, spacing_mm=spacing, affine=np.asarray(affine, dtype=float)
        ),
    )


@pytest.fixture
def pair(tmp_path):
    return volume(tmp_path, "bl.nii.gz"), volume(tmp_path, "fu.nii.gz")


class FakeParameterObject:
    def __init__(self):
        self.maps = []
        self.files = []
        self.changed = []

    def GetDefaultParameterMap(self, stage):
        return {"Transform": [stage]}

    def AddParameterMap(self, parameter_map):
        self.maps.append(parameter_map)

    def AddParameterFile(self, path):
        self.files.append(path)

    def SetParameter(self, index, name, value):
        self.changed.append((index, name, value))


class FakeItk:
    F = object()
    US = object()
    D = object()

    def __init__(self, *, transform_outputs=True, registration_outputs=True,
                 registration_error=None):
        self.parameters = []
        self.reads = []
        self.writes = []
        self.transform_outputs = transform_outputs
        self.registration_outputs = registration_outputs
        self.registration_error = registration_error
        self.transformix = None
        self.registration = None
        self.last_image = None
        outer = self

        class ParameterObject:
            @staticmethod
            def New():
                result = FakeParameterObject()
                outer.parameters.append(result)
                return result

        class TransformixFilter:
            @staticmethod
            def New(image):
                obj = MagicMock()
                obj.GetOutput.return_value = object()
                outer.transformix = obj
                return obj

        class ElastixRegistrationMethod:
            @staticmethod
            def New(fixed, moving):
                obj = MagicMock()
                obj.GetOutput.return_value = object()
                if outer.registration_error is not None:
                    obj.UpdateLargestPossibleRegion.side_effect = outer.registration_error
                else:
                    def update():
                        if outer.registration_outputs:
                            for i, _ in enumerate(outer.parameters[0].maps):
                                (outer.output_dir / f"TransformParameters.{i}.txt").write_text(
                                    '(CenterOfRotationPoint 0 0 0)\n', encoding="utf8"
                                )
                    obj.UpdateLargestPossibleRegion.side_effect = update
                obj.SetOutputDirectory.side_effect = lambda p: setattr(outer, "output_dir", Path(p))
                outer.registration = obj
                return obj

        class TranslationTransform:
            def __class_getitem__(cls, key):
                return cls

            @staticmethod
            def New():
                return MagicMock()

        self.ParameterObject = ParameterObject
        self.TransformixFilter = TransformixFilter
        self.ElastixRegistrationMethod = ElastixRegistrationMethod
        self.TranslationTransform = TranslationTransform

    def imread(self, path, dtype):
        self.reads.append((path, dtype))
        return (path, dtype)

    def imwrite(self, image, path):
        self.writes.append(path)
        if self.transformix is None or self.transform_outputs:
            if self.registration is None or self.registration_outputs:
                Path(path).write_bytes(b"fake image")


@pytest.fixture
def itk(monkeypatch):
    fake = FakeItk()
    monkeypatch.setitem(sys.modules, "itk", fake)
    return fake


# ---------------------------------------------------------------------------
# Input validation, stage ordering and initialization selection
# ---------------------------------------------------------------------------

def test_validate_ct_accepts_valid_volume(tmp_path):
    reg._validate_ct_volume(volume(tmp_path), role="Baseline")


@pytest.mark.parametrize("change,match", [
    ({"data": np.zeros((3, 3))}, "3-D"),
    ({"data": np.full((3, 3, 3), np.nan)}, "finite"),
    ({"spacing_mm": (1, 0, 1)}, "spacing"),
    ({"spacing_mm": (1, np.nan, 1)}, "spacing"),
    ({"spacing_mm": (1, 2)}, "spacing"),
    ({"affine": np.ones((3, 3))}, "affine"),
    ({"affine": np.full((4, 4), np.inf)}, "affine"),
])
def test_validate_ct_rejects_invalid_inputs(tmp_path, change, match):
    v = volume(tmp_path)
    for key, val in change.items():
        if key in ("spacing_mm", "affine"):
            setattr(v.metadata, key, val)
        else:
            setattr(v, key, val)
    with pytest.raises(reg.RegistrationError, match=match):
        reg._validate_ct_volume(v, role="Baseline")


def test_validate_ct_requires_real_source(tmp_path):
    with pytest.raises(reg.RegistrationError, match="source file"):
        reg._validate_ct_volume(volume(tmp_path, create=False), role="Follow-up")


@pytest.mark.parametrize("stages,expected", [
    ([" RIGID "], ("rigid",)),
    (["rigid", " AFFINE "], ("rigid", "affine")),
    (["affine"], ("affine",)),
])
def test_normalise_stages(stages, expected):
    assert reg._normalise_stages(stages) == expected


@pytest.mark.parametrize("stages,match", [
    ([], "At least one"),
    (["bspline"], "Unsupported"),
    (["rigid", "rigid"], "duplicates"),
    (["affine", "rigid"], "rigid must come before affine"),
])
def test_normalise_stages_rejects_invalid(stages, match):
    with pytest.raises(ValueError, match=match):
        reg._normalise_stages(stages)


@pytest.mark.parametrize("name,expected", [
    (" Adaptive ", "adaptive"),
    ("Anatomy", "anatomical"),
    ("anatomy-driven", "anatomical"),
    ("physical", "none"),
    ("physical_coordinates", "none"),
    ("geometric_center", "geometrical_center"),
    ("geometricalcenter", "geometrical_center"),
    ("centerofgravity", "center_of_gravity"),
])
def test_normalise_initialization_aliases(name, expected):
    assert reg._normalise_initialization_method(name) == expected


def test_normalise_initialization_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported initialization_method"):
        reg._normalise_initialization_method("guess")


def test_physical_extent_uses_affine_axis_lengths(tmp_path):
    affine = np.diag([2, 3, 4, 1])
    v = volume(tmp_path, shape=(5, 6, 7), affine=affine)
    np.testing.assert_allclose(reg._physical_extent_mm(v), (8, 15, 24))


@pytest.mark.parametrize("ratio,difference", [(0, 0), (1.1, 1), (0.5, -1), (0.5, np.inf)])
def test_extent_thresholds_must_be_valid(pair, ratio, difference):
    with pytest.raises(ValueError):
        reg._has_materially_different_scan_extents(
            *pair, ratio_threshold=ratio, difference_threshold_mm=difference
        )


def test_extent_mismatch_and_adaptive_selection(tmp_path):
    bl = volume(tmp_path, "long.nii.gz", shape=(10, 10, 100))
    fu = volume(tmp_path, "short.nii.gz", shape=(10, 10, 25))
    kw = dict(ratio_threshold=.75, difference_threshold_mm=30)
    assert reg._has_materially_different_scan_extents(bl, fu, **kw)
    assert reg._resolve_initialization_method(
        bl, fu, initialization_method="adaptive",
        extent_ratio_threshold=.75, extent_difference_threshold_mm=30
    ) == "anatomical"
    assert reg._resolve_initialization_method(
        bl, fu, initialization_method="none",
        extent_ratio_threshold=.75, extent_difference_threshold_mm=30
    ) == "none"
    assert reg._resolve_initialization_method(
        fu, fu, initialization_method="adaptive",
        extent_ratio_threshold=.75, extent_difference_threshold_mm=30
    ) == "geometrical_center"
    assert not reg._has_materially_different_scan_extents(
        fu, fu, ratio_threshold=.75, difference_threshold_mm=30
    )


def test_extent_singleton_axis(pair):
    for v in pair:
        v.data = v.data[:1]
    assert not reg._has_materially_different_scan_extents(
        *pair, ratio_threshold=.75, difference_threshold_mm=30
    )


# ---------------------------------------------------------------------------
# Content-based anatomical initialization: real numpy, no ITK
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window,expected", [
    (2, [1., 2.]),
    (1, [1., 2., 3., 4., 5.]),
    (4, [4/3, 2., 3., 4., 14/3]),
])
def test_smooth_profile_shape_and_edges(window, expected):
    data = np.array([1., 2.]) if window == 2 else np.arange(1., 6.)
    if window == 2:
        np.testing.assert_array_equal(reg._smooth_profile(data), expected)
    else:
        np.testing.assert_allclose(reg._smooth_profile(data, window), expected)


def test_superior_inferior_profile_selects_slice_axis_and_sorts(tmp_path):
    data = np.full((15, 8, 9), -1000., dtype=np.float32)
    data[:, 2:6, 2:7] = 350.
    data[0, 0, 0] = np.nan
    affine = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-2, 0, 0, 12], [0, 0, 0, 1.]])
    v = volume(tmp_path, data=data, affine=affine)
    z, features = reg._superior_inferior_anatomy_profile(v, inplane_stride=1)
    assert z.shape == (15,)
    assert features.shape == (15, 3)
    assert np.all(np.diff(z) > 0)
    assert np.all(np.isfinite(features))
    assert np.all((features >= 0) & (features <= 1))


@pytest.mark.parametrize("stride", [0, -2])
def test_superior_inferior_invalid_stride(tmp_path, stride):
    with pytest.raises(ValueError, match="inplane_stride"):
        reg._superior_inferior_anatomy_profile(volume(tmp_path), inplane_stride=stride)


def test_superior_inferior_invalid_affine_axis(tmp_path):
    v = volume(tmp_path)
    v.metadata.affine[2, :3] = 0
    with pytest.raises(reg.RegistrationError, match="superior-inferior"):
        reg._superior_inferior_anatomy_profile(v)


@pytest.mark.parametrize("first,second,expected", [
    ([1., 2.], [1., 2.], -1.),
    ([1., 2., 3.], [1., 2.], -1.),
    ([1., 1., 1.], [1., 2., 3.], 0.),
    ([1., 2., 3.], [8., 8., 8.], 0.),
    ([1., 2., 3.], [2., 4., 6.], 1.),
    ([1., 2., 3.], [6., 4., 2.], -1.),
])
def test_profile_correlation(first, second, expected):
    assert reg._profile_correlation(first, second) == pytest.approx(expected)


def test_body_centroid_uses_anatomical_region(tmp_path):
    data = np.full((18, 18, 20), -1000., dtype=np.float32)
    data[4:14, 5:15, 3:17] = 100.
    v = volume(tmp_path, data=data)
    centre = reg._body_centroid_ras(
        v, z_min=4, z_max=15, inplane_stride=1
    )
    np.testing.assert_allclose(centre, [8.5, 9.5, 9.5], atol=.6)


@pytest.mark.parametrize("z_min,z_max", [(50, 60), (0, 15)])
def test_body_centroid_raises_without_valid_tissue(tmp_path, z_min, z_max):
    v = volume(tmp_path, fill=-1000.)
    with pytest.raises(reg.RegistrationError, match="body centroid"):
        reg._body_centroid_ras(v, z_min=z_min, z_max=z_max, inplane_stride=1)


def test_anatomical_prealignment_from_shifted_profiles(tmp_path, monkeypatch):
    bl, fu = volume(tmp_path, "bl.nii.gz"), volume(tmp_path, "fu.nii.gz")
    z = np.arange(40, dtype=float)
    def features(zpos):
        return np.column_stack([
            np.exp(-((zpos - 13.) / 3.) ** 2),
            np.exp(-((zpos - 27.) / 2.) ** 2),
            np.sin(zpos / 4.),
        ])
    def profile(v, **_):
        return (z, features(z)) if v is bl else (z, features(z + 5))
    monkeypatch.setattr(reg, "_superior_inferior_anatomy_profile", profile)
    def centroid(v, **_):
        return np.array([12., 17., 0.]) if v is bl else np.array([8., 9., 0.])
    monkeypatch.setattr(reg, "_body_centroid_ras", centroid)
    result = reg._estimate_anatomical_prealignment(bl, fu, inplane_stride=1)
    assert result.translation_ras_mm == pytest.approx((4., 8., 5.), abs=1.1)
    assert result.translation_lps_mm == pytest.approx((-4., -8., 5.), abs=1.1)
    assert result.overlap_fraction >= .65
    assert np.isfinite(result.score)


def test_anatomical_prealignment_no_distinct_slices(pair, monkeypatch):
    monkeypatch.setattr(reg, "_superior_inferior_anatomy_profile",
                        lambda *_a, **_kw: (np.zeros(20), np.ones((20, 3))))
    with pytest.raises(reg.RegistrationError, match="not physically distinct"):
        reg._estimate_anatomical_prealignment(*pair)


def test_anatomical_prealignment_insufficient_overlap(pair, monkeypatch):
    counter = iter([0., 1000.])
    def fake_profile(*_a, **_kw):
        offset = next(counter)
        z = np.arange(10, dtype=float) + offset
        return z, np.ones((10, 3))
    monkeypatch.setattr(reg, "_superior_inferior_anatomy_profile", fake_profile)
    with pytest.raises(reg.RegistrationError, match="enough anatomical Z overlap"):
        reg._estimate_anatomical_prealignment(*pair)


@pytest.mark.parametrize("overlap", [.1, 1.1])
def test_anatomical_prealignment_rejects_bad_overlap(pair, overlap):
    with pytest.raises(ValueError, match="min_overlap_fraction"):
        reg._estimate_anatomical_prealignment(*pair, min_overlap_fraction=overlap)


# ---------------------------------------------------------------------------
# Parameter configuration and tiny helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,match", [
    ({"number_of_resolutions": 0}, "number_of_resolutions"),
    ({"maximum_iterations": 0}, "maximum_iterations"),
    ({"number_of_spatial_samples": 127}, "number_of_spatial_samples"),
    ({"required_ratio_of_valid_samples": 0}, "required_ratio"),
    ({"required_ratio_of_valid_samples": 1.1}, "required_ratio"),
    ({"required_ratio_of_valid_samples": np.nan}, "required_ratio"),
])
def test_parameter_validation_before_itk_import(kwargs, match):
    config = dict(number_of_resolutions=2, maximum_iterations=8,
                  number_of_spatial_samples=128, initialization_method="none")
    config.update(kwargs)
    with pytest.raises(ValueError, match=match):
        reg._build_parameter_object(("rigid",), **config)


def test_optional_valid_ratio_is_accepted():
    for value in (None, .01, 1.):
        reg._validate_required_ratio_of_valid_samples(value)


def test_missing_itk_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "itk", None)
    with pytest.raises(reg.RegistrationError, match="ITKElastix is not installed"):
        reg._build_parameter_object(
            ("rigid",), number_of_resolutions=1, maximum_iterations=1,
            number_of_spatial_samples=128, initialization_method="none"
        )


@pytest.mark.parametrize("method,expected", [
    ("none", "false"),
    ("anatomical", "false"),
    ("geometrical_center", "true"),
    ("center_of_gravity", "true"),
])
def test_parameter_maps_are_built_correctly(itk, method, expected):
    p = reg._build_parameter_object(
        ("rigid", "affine"), number_of_resolutions=2, maximum_iterations=16,
        number_of_spatial_samples=256, initialization_method=method,
        required_ratio_of_valid_samples=.25,
    )
    assert len(p.maps) == 2
    first, second = p.maps
    assert first["AutomaticTransformInitialization"] == [expected]
    assert second["AutomaticTransformInitialization"] == ["false"]
    if expected == "true":
        assert "AutomaticTransformInitializationMethod" in first
    else:
        assert "AutomaticTransformInitializationMethod" not in first
    assert first["Metric"] == ["AdvancedMattesMutualInformation"]
    assert first["MaximumNumberOfIterations"] == ["16"]
    assert first["NumberOfResolutions"] == ["2"]
    assert first["NumberOfSpatialSamples"] == ["256"]
    assert first["RequiredRatioOfValidSamples"] == ["0.25"]
    assert second["WriteResultImage"] == ["true"]


def test_parameter_map_leaves_default_sampling_ratio(itk):
    p = reg._build_parameter_object(
        ("rigid",), number_of_resolutions=1, maximum_iterations=1,
        number_of_spatial_samples=128, initialization_method="none"
    )
    assert "RequiredRatioOfValidSamples" not in p.maps[0]


def test_parameter_map_rejects_unresolved_adaptive(itk):
    with pytest.raises(ValueError, match="must be resolved"):
        reg._build_parameter_object(
            ("rigid",), number_of_resolutions=1, maximum_iterations=1,
            number_of_spatial_samples=128, initialization_method="adaptive"
        )


def test_remove_previous_outputs_preserves_unrelated_files(tmp_path):
    filenames = ["registered_baseline_ct.nii.gz", "TransformParameters.0.txt",
                 "result.0.nii", "result.ignore", "elastix.log", "notes.txt"]
    for name in filenames:
        (tmp_path / name).write_text("test")
    (tmp_path / "result.folder").mkdir()
    reg._remove_previous_outputs(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt", "result.folder"]


def test_remove_previous_outputs_can_run_on_empty_folder(tmp_path):
    reg._remove_previous_outputs(tmp_path)


def test_elastix_log_tail_filters_blanks(tmp_path):
    assert reg._elastix_log_tail(tmp_path) == ""
    (tmp_path / "elastix.log").write_text("\n one \n\n two \n three \n")
    assert reg._elastix_log_tail(tmp_path, line_count=2) == "\nElastix log tail:\ntwo\nthree"
    (tmp_path / "elastix.log").write_text("  \n  \n")
    assert reg._elastix_log_tail(tmp_path) == ""


def test_elastix_log_tail_io_failure(tmp_path, monkeypatch):
    (tmp_path / "elastix.log").write_text("error")
    original_read = Path.read_text
    def bad_read(self, *args, **kwargs):
        if self.name == "elastix.log":
            raise OSError("test failure")
        return original_read(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", bad_read)
    assert reg._elastix_log_tail(tmp_path) == ""


def test_parameter_text_parse_and_replace():
    original = '(CenterOfRotationPoint 1 2 3)\n(Transform "EulerTransform")\n'
    assert reg._parameter_values_from_text(original, "CenterOfRotationPoint") == ("1", "2", "3")
    updated = reg._set_parameter_line(original, "CenterOfRotationPoint", "4 5 6")
    assert reg._parameter_values_from_text(updated, "CenterOfRotationPoint") == ("4", "5", "6")
    assert updated.count("CenterOfRotationPoint") == 1
    assert '(Missing "new")' in reg._set_parameter_line("original", "Missing", '"new"')
    assert '(Missing "new")' in reg._set_parameter_line("original\n", "Missing", '"new"')
    with pytest.raises(reg.RegistrationError, match="missing parameter"):
        reg._parameter_values_from_text(original, "Missing")


def test_ct_body_dice_edge_cases():
    full = np.ones((3, 3, 3))
    empty = np.full((3, 3, 3), -1000.)
    assert reg._ct_body_dice(full, full, threshold_hu=-600) == 1.
    assert reg._ct_body_dice(full, empty, threshold_hu=-600) == 0.
    assert reg._ct_body_dice(empty, empty, threshold_hu=-600) == 0.
    assert reg._ct_body_dice(full, full[:2], threshold_hu=-600) == 0.
    partly_nan = full.copy()
    partly_nan[0, 0, 0] = np.nan
    assert 0 < reg._ct_body_dice(partly_nan, full, threshold_hu=-600) < 1


def test_ct_body_dice_large_input_is_downsampled():
    image = np.full((193, 3, 3), 100, dtype=np.float32)
    assert reg._ct_body_dice(image, image, threshold_hu=-600) == 1.


def test_output_cache_mtime_rules(tmp_path, monkeypatch):
    source, transform, output = (tmp_path / n for n in ("source", "transform", "output"))
    source.touch()
    transform.touch()
    assert reg._transformed_output_needs_update(output, source, transform)
    output.touch()
    assert not reg._transformed_output_needs_update(output, source, transform)
    source.touch()
    source_ns = output.stat().st_mtime_ns + 1_000_000
    import os
    os.utime(source, ns=(source_ns, source_ns))
    assert reg._transformed_output_needs_update(output, source, transform)
    output_ns = source_ns + 1_000_000
    os.utime(output, ns=(output_ns, output_ns))
    transform_ns = output_ns + 1_000_000
    os.utime(transform, ns=(transform_ns, transform_ns))
    assert reg._transformed_output_needs_update(output, source, transform)
    transform.unlink()
    assert reg._transformed_output_needs_update(output, source, transform)


# ---------------------------------------------------------------------------
# Transformix wrapper: all validations and both interpolation choices
# ---------------------------------------------------------------------------

def test_warp_mask_uses_nearest_neighbor(pair, tmp_path, itk):
    source = pair[0]
    path = tmp_path / "TransformParameters.0.txt"
    path.touch()
    target = tmp_path / "warped" / "mask.nii.gz"
    output = reg.warp_baseline_mask_to_fu(
        source, transform_path=path, registered_mask_path=target
    )
    assert output == target and output.is_file()
    params = itk.parameters[0]
    assert (0, "ResampleInterpolator", "FinalNearestNeighborInterpolator") in params.changed
    assert (0, "FinalBSplineInterpolationOrder", "0") in params.changed
    assert (0, "ResultImagePixelType", "unsigned short") in params.changed
    assert itk.reads[0][1] is itk.US
    itk.transformix.SetLogToConsole.assert_called_once_with(False)
    itk.transformix.SetLogToFile.assert_called_once_with(False)


def test_warp_pet_uses_linear_sampling(pair, tmp_path, itk):
    transform = tmp_path / "TransformParameters.0.txt"
    transform.touch()
    output = reg.warp_baseline_pet_to_fu(
        pair[0], transform_path=transform, registered_pet_path=tmp_path / "pet.nii.gz",
        force=True,
    )
    assert output.is_file()
    assert (0, "FinalBSplineInterpolationOrder", "1") in itk.parameters[0].changed
    assert (0, "ResampleInterpolator", "FinalBSplineInterpolator") in itk.parameters[0].changed
    assert itk.reads[0][1] is itk.F


def test_warp_reuses_cache_without_itk(pair, tmp_path, monkeypatch):
    transform, out = tmp_path / "TransformParameters.0.txt", tmp_path / "cached.nii.gz"
    transform.touch()
    out.touch()
    source = pair[0].metadata.path
    old_ns = source.stat().st_mtime_ns - 1_000_000
    import os
    os.utime(source, ns=(old_ns, old_ns))
    os.utime(transform, ns=(old_ns, old_ns))
    monkeypatch.setitem(sys.modules, "itk", None)
    assert reg.warp_baseline_mask_to_fu(
        pair[0], transform_path=transform, registered_mask_path=out
    ) == out


@pytest.mark.parametrize("missing,match", [("source", "source file not found"),
                                             ("transform", "Rigid transform not found")])
def test_transform_rejects_missing_input(pair, tmp_path, missing, match):
    transform = tmp_path / "TransformParameters.0.txt"
    transform.touch()
    if missing == "source":
        pair[0].metadata.path.unlink()
    else:
        transform.unlink()
    with pytest.raises(reg.RegistrationError, match=match):
        reg.warp_baseline_mask_to_fu(
            pair[0], transform_path=transform,
            registered_mask_path=tmp_path / "result.nii.gz"
        )


def test_transform_requires_itk_when_uncached(pair, tmp_path, monkeypatch):
    transform = tmp_path / "transform.txt"
    transform.touch()
    monkeypatch.setitem(sys.modules, "itk", None)
    with pytest.raises(reg.RegistrationError, match="ITKElastix is required"):
        reg.warp_baseline_mask_to_fu(
            pair[0], transform_path=transform, registered_mask_path=tmp_path / "out.nii.gz"
        )


def test_transform_rejects_bad_pixel_kind(pair, tmp_path, itk):
    transform = tmp_path / "transform.txt"
    transform.touch()
    with pytest.raises(ValueError, match="pixel kind"):
        reg._transform_volume_to_followup(
            pair[0], transform_path=transform, output_path=tmp_path / "out.nii.gz",
            source_pixel_kind="complex", interpolation_order=2,
            result_pixel_type="complex", role="mock",
        )


def test_transform_reports_execution_failure(pair, tmp_path, itk):
    transform = tmp_path / "transform.txt"
    transform.touch()
    def fail(*args, **kwargs):
        raise RuntimeError("mocked Transformix failure")
    itk.imread = fail
    with pytest.raises(reg.RegistrationError, match="mocked Transformix failure"):
        reg.warp_baseline_pet_to_fu(
            pair[0], transform_path=transform, registered_pet_path=tmp_path / "out.nii.gz"
        )


def test_transform_reports_missing_result(pair, tmp_path, monkeypatch):
    fake = FakeItk(transform_outputs=False)
    monkeypatch.setitem(sys.modules, "itk", fake)
    transform = tmp_path / "transform.txt"
    transform.touch()
    with pytest.raises(reg.RegistrationError, match="did not write"):
        reg.warp_baseline_mask_to_fu(
            pair[0], transform_path=transform, registered_mask_path=tmp_path / "out.nii.gz"
        )


# ---------------------------------------------------------------------------
# Main entry points with a small stand-in for the slow C++ Elastix pipeline
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_registration(monkeypatch, pair, itk):
    bl, fu = pair
    monkeypatch.setattr(reg, "load_nifti_volume", lambda *_args, **_kw: fu)
    return itk


def test_register_ct_pair_rigid_success(pair, tmp_path, mock_registration):
    bl, fu = pair
    result = reg.register_ct_pair(
        bl, fu, patient_id="P001", output_dir=tmp_path / "outputs",
        stages=("rigid",), initialization_method="none",
    )
    assert isinstance(result, reg.RegistrationResult)
    assert result.patient_id == "P001"
    assert result.stages == ("rigid",)
    assert result.initialization_method == "none"
    assert result.initial_translation_lps_mm is None
    assert result.anatomical_alignment_score is None
    assert result.body_dice == pytest.approx(1.)
    assert result.registered_ct_path.exists()
    assert len(result.transform_parameter_paths) == 1
    assert mock_registration.reads[0][0] == str(fu.metadata.path)
    assert mock_registration.reads[1][0] == str(bl.metadata.path)
    mock_registration.registration.SetLogToConsole.assert_called_once_with(False)
    mock_registration.registration.SetLogToFile.assert_called_once_with(True)


def test_register_ct_pair_rigid_affine_success(pair, tmp_path, mock_registration):
    result = reg.register_ct_pair(
        *pair, output_dir=tmp_path / "multi", stages=("rigid", "affine"),
        initialization_method="center_of_gravity", log_to_console=True,
    )
    assert len(result.transform_parameter_paths) == 2
    assert result.initialization_method == "center_of_gravity"
    mock_registration.registration.SetLogToConsole.assert_called_once_with(True)


def test_register_ct_pair_anatomical_success(pair, tmp_path, mock_registration, monkeypatch):
    pre = reg.AnatomicalPrealignment(
        translation_ras_mm=(2., 3., 4.), translation_lps_mm=(-2., -3., 4.),
        score=.9, overlap_fraction=.75,
    )
    monkeypatch.setattr(reg, "_estimate_anatomical_prealignment", lambda *_a, **_kw: pre)
    mask = object()
    make_mask = MagicMock(return_value=mask)
    monkeypatch.setattr(reg, "_make_itk_body_mask", make_mask)
    flatten = MagicMock()
    monkeypatch.setattr(reg, "_flatten_rigid_combination_transform", flatten)
    result = reg.register_ct_pair(
        *pair, output_dir=tmp_path / "anatomy", stages=("rigid",),
        initialization_method="anatomical",
    )
    assert result.initial_translation_lps_mm == pre.translation_lps_mm
    assert result.anatomical_alignment_score == pre.score
    assert result.initialization_method == "anatomical"
    mock_registration.registration.SetExternalInitialTransform.assert_called_once()
    mock_registration.registration.SetFixedMask.assert_called_once_with(mask)
    mock_registration.registration.SetMovingMask.assert_called_once_with(mask)
    assert make_mask.call_count == 2
    flatten.assert_called_once()


@pytest.mark.parametrize("overrides,match", [
    ({"body_mask_threshold_hu": np.nan}, "body_mask_threshold_hu"),
    ({"minimum_body_dice": -0.1}, "minimum_body_dice"),
    ({"minimum_body_dice": 1.1}, "minimum_body_dice"),
    ({"stages": ()}, "At least one"),
    ({"stages": ("affine", "rigid")}, "rigid must come before affine"),
    ({"initialization_method": "not_a_method"}, "Unsupported initialization"),
])
def test_register_ct_pair_rejects_bad_options(pair, tmp_path, overrides, match):
    kwargs = dict(output_dir=tmp_path / "bad", initialization_method="none")
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        reg.register_ct_pair(*pair, **kwargs)


def test_anatomical_initializer_requires_rigid_only(pair, tmp_path):
    with pytest.raises(ValueError, match="rigid-only"):
        reg.register_ct_pair(
            *pair, output_dir=tmp_path / "invalid", stages=("rigid", "affine"),
            initialization_method="anatomical",
        )


def test_register_ct_pair_overwrite_policy(pair, tmp_path, mock_registration):
    target = tmp_path / "out"
    target.mkdir()
    (target / "TransformParameters.0.txt").touch()
    with pytest.raises(reg.RegistrationError, match="already exist"):
        reg.register_ct_pair(
            *pair, output_dir=target, stages=("rigid",),
            initialization_method="none", overwrite=False,
        )
    result = reg.register_ct_pair(
        *pair, output_dir=target, stages=("rigid",),
        initialization_method="none", overwrite=True,
    )
    assert result.body_dice == pytest.approx(1.)


def test_register_ct_pair_overwrite_policy_registered_ct(pair, tmp_path, mock_registration):
    target = tmp_path / "out"
    target.mkdir()
    (target / "registered_baseline_ct.nii.gz").touch()
    with pytest.raises(reg.RegistrationError, match="already exist"):
        reg.register_ct_pair(
            *pair, output_dir=target, stages=("rigid",),
            initialization_method="none", overwrite=False,
        )


def test_register_ct_pair_overwrite_false_empty_dir(pair, tmp_path, mock_registration):
    result = reg.register_ct_pair(
        *pair, output_dir=tmp_path / "clean", stages=("rigid",),
        initialization_method="none", overwrite=False,
    )
    assert result.registered_ct_path.exists()


def test_registration_failure_includes_log_tail(pair, tmp_path, monkeypatch):
    fake = FakeItk(registration_error=RuntimeError("mock low overlap"))
    monkeypatch.setitem(sys.modules, "itk", fake)
    target = tmp_path / "bad"
    target.mkdir()
    (target / "elastix.log").write_text("failed to find correspondences\n")
    with pytest.raises(reg.RegistrationError, match="mock low overlap") as exc:
        reg.register_ct_pair(
            *pair, output_dir=target, initialization_method="none",
            overwrite=False, stages=("rigid",),
        )
    assert "Elastix log tail" in str(exc.value)
    assert "failed to find correspondences" in str(exc.value)


def test_registration_failure_includes_anatomy_metadata(pair, tmp_path, monkeypatch):
    fake = FakeItk(registration_error=RuntimeError("fail"))
    monkeypatch.setitem(sys.modules, "itk", fake)
    pre = reg.AnatomicalPrealignment(
        translation_ras_mm=(0., 0., 3.), translation_lps_mm=(0., 0., 3.),
        score=.8, overlap_fraction=.9,
    )
    monkeypatch.setattr(reg, "_estimate_anatomical_prealignment", lambda *_a, **_kw: pre)
    monkeypatch.setattr(reg, "_make_itk_body_mask", lambda *_a, **_kw: object())
    with pytest.raises(reg.RegistrationError, match="anatomy initial translation"):
        reg.register_ct_pair(
            *pair, output_dir=tmp_path / "bad", initialization_method="anatomical",
            stages=("rigid",),
        )


def test_registration_missing_registered_ct(pair, tmp_path, monkeypatch):
    fake = FakeItk(registration_outputs=False)
    monkeypatch.setitem(sys.modules, "itk", fake)
    with pytest.raises(reg.RegistrationError, match="no registered CT"):
        reg.register_ct_pair(*pair, output_dir=tmp_path / "missing", stages=("rigid",),
                             initialization_method="none")


def test_registration_missing_transform_files(pair, tmp_path, mock_registration, monkeypatch):
    original = mock_registration.registration
    old_write = mock_registration.imwrite
    def remove_transforms_after_write(image, path):
        old_write(image, path)
        for f in Path(path).parent.glob("TransformParameters.*.txt"):
            f.unlink()
    monkeypatch.setattr(mock_registration, "imwrite", remove_transforms_after_write)
    with pytest.raises(reg.RegistrationError, match="expected transform parameter files"):
        reg.register_ct_pair(*pair, output_dir=tmp_path / "missing", stages=("rigid",),
                             initialization_method="none")


@pytest.mark.parametrize("kind,match", [
    ("shape", "does not match FU CT shape"),
    ("affine", "affine geometry"),
    ("body", "insufficient overlap"),
])
def test_registration_postcondition_guards(pair, tmp_path, mock_registration, monkeypatch,
                                            kind, match):
    bl, fu = pair
    if kind == "shape":
        wrong = volume(tmp_path, "wrong.nii.gz", shape=(3, 3, 3))
    elif kind == "affine":
        bad_affine = np.eye(4)
        bad_affine[0, 3] = 100.
        wrong = volume(tmp_path, "wrong.nii.gz", affine=bad_affine)
    else:
        wrong = volume(tmp_path, "wrong.nii.gz", fill=-1000.)
    monkeypatch.setattr(reg, "load_nifti_volume", lambda *_a, **_kw: wrong)
    with pytest.raises(reg.RegistrationError, match=match):
        reg.register_ct_pair(*pair, output_dir=tmp_path / "postconditions",
                             stages=("rigid",), initialization_method="none")


def test_registration_anatomical_missing_transform_is_wrapped(pair, tmp_path,
                                                              mock_registration, monkeypatch):
    pre = reg.AnatomicalPrealignment((0., 0., 0.), (0., 0., 0.), .9, .9)
    monkeypatch.setattr(reg, "_estimate_anatomical_prealignment", lambda *_a, **_kw: pre)
    monkeypatch.setattr(reg, "_make_itk_body_mask", lambda *_a, **_kw: object())
    original = mock_registration.ElastixRegistrationMethod.New
    def missing_transform(fixed, moving):
        obj = original(fixed, moving)
        obj.UpdateLargestPossibleRegion.side_effect = lambda: None
        return obj
    monkeypatch.setattr(mock_registration.ElastixRegistrationMethod, "New", missing_transform)
    with pytest.raises(reg.RegistrationError, match="exactly one transform"):
        reg.register_ct_pair(*pair, output_dir=tmp_path / "invalid",
                             stages=("rigid",), initialization_method="anatomical")


def test_register_patient_ct_validates_missing_scans(tmp_path):
    valid = volume(tmp_path)
    pair = SimpleNamespace(patient_id="P009",
                           baseline=SimpleNamespace(ct=None),
                           followup=SimpleNamespace(ct=valid))
    with pytest.raises(reg.RegistrationError, match="no loaded baseline CT"):
        reg.register_patient_ct(pair, output_dir=tmp_path / "out")
    pair.baseline.ct = valid
    pair.followup.ct = None
    with pytest.raises(reg.RegistrationError, match="no loaded follow-up CT"):
        reg.register_patient_ct(pair, output_dir=tmp_path / "out")


def test_register_patient_ct_passes_all_options(pair, tmp_path, monkeypatch):
    bl, fu = pair
    p = SimpleNamespace(patient_id="P010", baseline=SimpleNamespace(ct=bl),
                        followup=SimpleNamespace(ct=fu))
    sentinel = object()
    fn = MagicMock(return_value=sentinel)
    monkeypatch.setattr(reg, "register_ct_pair", fn)
    options = dict(
        stages=("rigid",), number_of_resolutions=1, maximum_iterations=10,
        number_of_spatial_samples=128, initialization_method="none",
        extent_ratio_threshold=.7, extent_difference_threshold_mm=30.,
        anatomical_min_overlap_fraction=.8, anatomical_profile_stride=2,
        body_mask_threshold_hu=-500., minimum_body_dice=.3,
        required_ratio_of_valid_samples=.4, log_to_console=True, overwrite=False,
    )
    result = reg.register_patient_ct(p, output_dir=tmp_path / "out", **options)
    assert result is sentinel
    fn.assert_called_once_with(
        bl, fu, patient_id="P010", output_dir=tmp_path / "out", **options
    )

# ---------------------------------------------------------------------------
# ITK mask construction and rigid-transform chain flattening
# ---------------------------------------------------------------------------

def test_make_itk_body_mask_sets_thresholds():
    threshold_filter = MagicMock()
    output = object()
    threshold_filter.GetOutput.return_value = output
    class Image:
        def __class_getitem__(cls, key):
            return ("image", key)
    class BinaryThresholdImageFilter:
        def __class_getitem__(cls, key):
            return cls
        @staticmethod
        def New():
            return threshold_filter
    itk = SimpleNamespace(Image=Image, UC=object(),
                          BinaryThresholdImageFilter=BinaryThresholdImageFilter)
    image = object()
    assert reg._make_itk_body_mask(itk, image, threshold_hu=-400.) is output
    threshold_filter.SetInput.assert_called_once_with(image)
    threshold_filter.SetLowerThreshold.assert_called_once_with(-400.)
    threshold_filter.SetUpperThreshold.assert_called_once_with(1.0e6)
    threshold_filter.SetInsideValue.assert_called_once_with(1)
    threshold_filter.SetOutsideValue.assert_called_once_with(0)
    threshold_filter.UpdateLargestPossibleRegion.assert_called_once()


class FakeCombination:
    def __init__(self, matrix=None, translation=(5, -3, 2)):
        self.matrix = np.asarray(np.eye(3) if matrix is None else matrix, dtype=float)
        self.translation = np.asarray(translation, dtype=float)

    def TransformPoint(self, point):
        return tuple(self.matrix @ np.asarray(point, dtype=float) + self.translation)


class FakeEuler:
    def __init__(self, wrong_output=False):
        self.rotation = np.eye(3)
        self.offset = np.zeros(3)
        self.wrong_output = wrong_output
        self.computez = None
        self.center = None

    def SetComputeZYX(self, val):
        self.computez = val

    def SetCenter(self, val):
        self.center = val

    def SetMatrix(self, val):
        self.rotation = np.asarray(val)

    def SetOffset(self, val):
        self.offset = np.asarray(val)

    def GetParameters(self):
        return (0., 0., 0., *self.offset)

    def TransformPoint(self, point):
        value = self.rotation @ np.asarray(point) + self.offset
        if self.wrong_output:
            value[0] += 2.
        return tuple(value)


def flatten_itk(combination, *, use_fallback_matrix=False, wrong_output=False,
                fail_euler=False):
    euler = FakeEuler(wrong_output=wrong_output)
    class Euler3DTransform:
        def __class_getitem__(cls, key):
            return cls
        @staticmethod
        def New():
            if fail_euler:
                raise RuntimeError("mock Euler constructor error")
            return euler
    itk = SimpleNamespace(D=object(), Euler3DTransform=Euler3DTransform)
    if use_fallback_matrix:
        itk.GetMatrixFromArray = lambda v: v
    else:
        itk.matrix_from_array = lambda v: v
    registration = SimpleNamespace(GetCombinationTransform=lambda: combination)
    return itk, registration, euler


@pytest.mark.parametrize("use_fallback", [False, True])
def test_flatten_rigid_transform_preserves_mapping(tmp_path, use_fallback):
    path = tmp_path / "TransformParameters.0.txt"
    path.write_text(
        '(CenterOfRotationPoint 0 0 0)\n'
        '(TransformParameters 0 0 0 0 0 0)\n'
        '(InitialTransformParameterFileName "initial.txt")\n', encoding="utf8"
    )
    itk, registration, euler = flatten_itk(
        FakeCombination(), use_fallback_matrix=use_fallback
    )
    parameters = reg._flatten_rigid_combination_transform(itk, registration, path)
    assert parameters == pytest.approx((0, 0, 0, 5, -3, 2))
    assert euler.computez is False
    assert euler.center == (0., 0., 0.)
    text = path.read_text()
    assert '(InitialTransformParameterFileName "NoInitialTransform")' in text
    assert '(ComputeZYX "false")' in text
    assert reg._parameter_values_from_text(text, "TransformParameters") == (
        "0", "0", "0", "5", "-3", "2"
    )
    assert not path.with_name(path.name + ".tmp").exists()


@pytest.mark.parametrize("matrix", [
    np.diag([2., 1., 1.]),
    np.diag([-1., 1., 1.]),
])
def test_flatten_rejects_nonlinear_or_reflected_transform(tmp_path, matrix):
    path = tmp_path / "transform.txt"
    path.write_text('(CenterOfRotationPoint 0 0 0)\n')
    itk, registration, _ = flatten_itk(FakeCombination(matrix=matrix))
    with pytest.raises(reg.RegistrationError, match="not rigid"):
        reg._flatten_rigid_combination_transform(itk, registration, path)


@pytest.mark.parametrize("text,match", [
    ('(TransformParameters 0 0 0 0 0 0)\n', "missing parameter"),
    ('(CenterOfRotationPoint 0 0)\n', "invalid centre of rotation"),
])
def test_flatten_rejects_invalid_rotation_centre(tmp_path, text, match):
    path = tmp_path / "transform.txt"
    path.write_text(text)
    itk, registration, _ = flatten_itk(FakeCombination())
    with pytest.raises(reg.RegistrationError, match=match):
        reg._flatten_rigid_combination_transform(itk, registration, path)


def test_flatten_detects_transform_disagreement(tmp_path):
    path = tmp_path / "transform.txt"
    path.write_text('(CenterOfRotationPoint 0 0 0)\n')
    itk, registration, _ = flatten_itk(FakeCombination(), wrong_output=True)
    with pytest.raises(reg.RegistrationError, match="does not reproduce"):
        reg._flatten_rigid_combination_transform(itk, registration, path)
    assert "NoInitialTransform" not in path.read_text()


def test_flatten_wraps_unexpected_errors(tmp_path):
    path = tmp_path / "transform.txt"
    path.write_text('(CenterOfRotationPoint 0 0 0)\n')
    itk, registration, _ = flatten_itk(FakeCombination(), fail_euler=True)
    with pytest.raises(reg.RegistrationError, match="mock Euler constructor error"):
        reg._flatten_rigid_combination_transform(itk, registration, path)


def test_transformix_without_optional_file_logging(pair, tmp_path, itk, monkeypatch):
    transform = tmp_path / "transform.txt"
    transform.touch()
    original_new = itk.TransformixFilter.New
    def no_file_logging(image):
        obj = original_new(image)
        del obj.SetLogToFile
        return obj
    monkeypatch.setattr(itk.TransformixFilter, "New", no_file_logging)
    result = reg.warp_baseline_pet_to_fu(
        pair[0], transform_path=transform, registered_pet_path=tmp_path / "pet.nii.gz"
    )
    assert result.is_file()


def test_registration_without_optional_file_logging(pair, tmp_path,
                                                     mock_registration, monkeypatch):
    original_new = mock_registration.ElastixRegistrationMethod.New
    def no_file_logging(fixed, moving):
        obj = original_new(fixed, moving)
        del obj.SetLogToFile
        return obj
    monkeypatch.setattr(mock_registration.ElastixRegistrationMethod, "New", no_file_logging)
    result = reg.register_ct_pair(
        *pair, output_dir=tmp_path / "no_optional_logging", stages=("rigid",),
        initialization_method="none"
    )
    assert result.body_dice == pytest.approx(1.)
