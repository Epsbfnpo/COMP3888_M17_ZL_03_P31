"""Fast, isolated unit tests for src.registration_mapping.

Run from the repository root:
    python -m pytest tests/test_registration_mapping.py -v
    python -m pytest tests/test_registration_mapping.py \
        --cov=src.registration_mapping --cov-branch --cov-report=term-missing

No CT data, Streamlit application, actual Elastix registration, or ITK installation
is required. Tiny synthetic transforms and a mock ITK module exercise both the
success paths and the failures. This measures *code* coverage, not anatomical
registration or coordinate-mapping accuracy on real patient images.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src import registration_mapping as mapping


@pytest.fixture(autouse=True)
def clear_slice_map_cache():
    """Prevent lru_cache from hiding a branch or leaking between test cases."""
    mapping.fu_to_bl_slice_map.cache_clear()
    yield
    mapping.fu_to_bl_slice_map.cache_clear()


def affine_flat(*, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag(spacing)
    matrix[:3, 3] = origin
    return tuple(matrix.ravel())


def mapping_args(**overrides):
    defaults = dict(
        bl_shape=(6, 7, 8),
        bl_affine_flat=affine_flat(),
        bl_orientation=("R", "A", "S"),
        fu_shape=(6, 7, 8),
        fu_affine_flat=affine_flat(),
        fu_orientation=("R", "A", "S"),
        plane_name="axial",
        transform_path="synthetic-rigid-transform.txt",
        transform_modified_ns=123,
    )
    defaults.update(overrides)
    return defaults


class FakeRigid:
    """Enough of ITK's Euler3DTransform to check setters and mapped points."""

    def __init__(self):
        self.center = (0.0, 0.0, 0.0)
        self.compute_zyx = False
        self.rotation = (0.0, 0.0, 0.0)
        self.translation = (0.0, 0.0, 0.0)
        self.points = []

    def SetCenter(self, center):
        self.center = tuple(center)

    def SetComputeZYX(self, enabled):
        self.compute_zyx = enabled

    def SetRotation(self, rx, ry, rz):
        self.rotation = (rx, ry, rz)

    def SetTranslation(self, displacement):
        self.translation = tuple(displacement)

    def TransformPoint(self, point):
        self.points.append(tuple(point))
        # All transform-building tests check setters; the synthetic geometry
        # tests below use 0 rotation and may use a known LPS translation.
        assert not any(self.rotation), "FakeRigid only supports zero rotation"
        p = np.asarray(point, dtype=float)
        return tuple(p + np.asarray(self.translation, dtype=float))


class FakeParameterObject:
    def __init__(self, parameters):
        self.parameters = parameters
        self.paths = []
        self.requested_indices = []

    def AddParameterFile(self, path):
        self.paths.append(path)

    def GetParameterMap(self, index):
        self.requested_indices.append(index)
        return self.parameters


@pytest.fixture
def fake_itk(monkeypatch):
    """Install a complete minimal fake ITK interface for Euler reconstruction."""
    parameter_map = {
        "Transform": ['"EulerTransform"'],
        "InitialTransformParameterFileName": ['"NoInitialTransform"'],
        "TransformParameters": ["0", "0", "0", "1.5", "-2", "3"],
        "CenterOfRotationPoint": ["4", "5", "6"],
        "ComputeZYX": ['"false"'],
    }
    records = SimpleNamespace(parameter_map=parameter_map, objects=[], rigids=[])

    class ParameterObject:
        @staticmethod
        def New():
            obj = FakeParameterObject(records.parameter_map)
            records.objects.append(obj)
            return obj

    class Euler3DTransform:
        def __class_getitem__(cls, index):
            return cls

        @staticmethod
        def New():
            rigid = FakeRigid()
            records.rigids.append(rigid)
            return rigid

    monkeypatch.setitem(
        sys.modules,
        "itk",
        SimpleNamespace(ParameterObject=ParameterObject,
                        Euler3DTransform=Euler3DTransform, D=object()),
    )
    return records


@pytest.fixture
def transform_file(tmp_path):
    """Existence matters; mock ParameterObject supplies all parameters."""
    path = tmp_path / "TransformParameters.0.txt"
    path.write_text('(Transform "EulerTransform")\n', encoding="utf-8")
    return path


@pytest.fixture
def identity_rigid(monkeypatch):
    """Return translated 3-D LPS points while recording construction calls."""
    built = []

    def construct(_):
        obj = FakeRigid()
        built.append(obj)
        return obj

    monkeypatch.setattr(mapping, "_build_elastix_euler_transform", construct)
    return built


# ---------------------------------------------------------------------------
# RAS <-> LPS coordinate conversion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("point,expected", [
    ((1, 2, 3), (-1, -2, 3)),
    ((-4, 5, -6), (4, -5, -6)),
    ((0, 0, 0), (0, 0, 0)),
    ((1.25, -2.5, 3.75), (-1.25, 2.5, 3.75)),
])
def test_ras_to_lps_coordinates(point, expected):
    result = mapping._ras_to_lps(point)
    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.dtype("float64")


@pytest.mark.parametrize("point,expected", [
    ((1, 2, 3), (-1, -2, 3)),
    ((-4, 5, -6), (4, -5, -6)),
    ((0, 0, 0), (0, 0, 0)),
    ((1.25, -2.5, 3.75), (-1.25, 2.5, 3.75)),
])
def test_lps_to_ras_coordinates(point, expected):
    result = mapping._lps_to_ras(point)
    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.dtype("float64")


@pytest.mark.parametrize("point", [
    (), (1,), (1, 2), (1, 2, 3, 4), ((1, 2, 3),),
    [[1], [2], [3]], np.zeros((2, 3)),
])
@pytest.mark.parametrize("fn,space", [
    (mapping._ras_to_lps, "RAS"), (mapping._lps_to_ras, "LPS")
])
def test_coordinate_helpers_reject_wrong_shape(fn, space, point):
    with pytest.raises(ValueError, match=f"3-D {space} point"):
        fn(point)


def test_ras_lps_round_trip():
    point = np.asarray([13.5, -4.25, 200.0])
    np.testing.assert_array_equal(mapping._lps_to_ras(mapping._ras_to_lps(point)), point)


# ---------------------------------------------------------------------------
# Elastix parameter parsing and rigid Euler reconstruction
# ---------------------------------------------------------------------------

def test_parameter_value_normalizes_values_to_strings():
    assert mapping._parameter_value({"Metric": (1, 2.5, "four")}, "Metric") == (
        "1", "2.5", "four"
    )


def test_parameter_value_preserves_empty_values():
    assert mapping._parameter_value({"Metric": []}, "Metric") == ()


def test_parameter_value_missing_with_default():
    assert mapping._parameter_value({}, "Metric", default=("fallback",)) == (
        "fallback",
    )


def test_parameter_value_missing_without_default():
    with pytest.raises(ValueError, match="parameter 'Missing' is missing") as exc:
        mapping._parameter_value({}, "Missing")
    assert isinstance(exc.value.__cause__, KeyError)


def test_parameter_value_unreadable_map_uses_default():
    bad = Mock()
    bad.__getitem__ = Mock(side_effect=RuntimeError("not readable"))
    assert mapping._parameter_value(bad, "Param", default=("ok",)) == ("ok",)


def test_parameter_value_unreadable_map_raises_value_error():
    bad = Mock()
    bad.__getitem__ = Mock(side_effect=RuntimeError("not readable"))
    with pytest.raises(ValueError, match="parameter 'Param' is missing") as exc:
        mapping._parameter_value(bad, "Param")
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_euler_transform_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Rigid transform not found"):
        mapping._build_elastix_euler_transform(tmp_path / "missing.txt")


def test_euler_transform_requires_itk(monkeypatch, transform_file):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "itk":
            raise ImportError("mock: itk not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="ITKElastix is required") as exc:
        mapping._build_elastix_euler_transform(transform_file)
    assert isinstance(exc.value.__cause__, ImportError)


def test_euler_transform_builds_correct_parameters(fake_itk, transform_file):
    rigid = mapping._build_elastix_euler_transform(str(transform_file))
    assert rigid is fake_itk.rigids[-1]
    assert rigid.center == (4.0, 5.0, 6.0)
    assert rigid.rotation == (0.0, 0.0, 0.0)
    assert rigid.translation == (1.5, -2.0, 3.0)
    assert rigid.compute_zyx is False
    assert fake_itk.objects[-1].paths == [str(transform_file)]
    assert fake_itk.objects[-1].requested_indices == [0]
    np.testing.assert_allclose(rigid.TransformPoint((1, 2, 3)), (2.5, 0, 6))


def test_euler_transform_accepts_path_objects_and_home_expansion(
    fake_itk, tmp_path, monkeypatch
):
    # POSIX expands ~ from HOME; Windows uses USERPROFILE instead.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    path = tmp_path / "rigid.txt"
    path.touch()
    rigid = mapping._build_elastix_euler_transform(Path("~/rigid.txt"))
    assert rigid is fake_itk.rigids[-1]
    assert fake_itk.objects[-1].paths == [str(path)]


@pytest.mark.parametrize("valid_initial", [
    "NoInitialTransform", '"NoInitialTransform"',
    "NoInitialTransformParameterFileName", '"NoInitialTransformParameterFileName"',
])
def test_euler_transform_accepts_no_initial_chain_values(
    fake_itk, transform_file, valid_initial
):
    fake_itk.parameter_map["InitialTransformParameterFileName"] = [valid_initial]
    assert isinstance(mapping._build_elastix_euler_transform(transform_file), FakeRigid)


def test_euler_transform_defaults_optional_parameters(fake_itk, transform_file):
    del fake_itk.parameter_map["InitialTransformParameterFileName"]
    del fake_itk.parameter_map["ComputeZYX"]
    rigid = mapping._build_elastix_euler_transform(transform_file)
    assert rigid.compute_zyx is False


@pytest.mark.parametrize("compute_zyx,expected", [
    ('"TRUE"', True), ("true", True), ("TrUe", True),
    ('"false"', False), ("0", False), ("anything", False),
])
def test_euler_transform_compute_zyx(fake_itk, transform_file, compute_zyx, expected):
    fake_itk.parameter_map["ComputeZYX"] = [compute_zyx]
    rigid = mapping._build_elastix_euler_transform(transform_file)
    assert rigid.compute_zyx is expected


@pytest.mark.parametrize("invalid_transform", [
    "AffineTransform", "BSplineTransform", "TranslationTransform", "CompositeTransform"
])
def test_euler_transform_rejects_non_rigid(
    fake_itk, transform_file, invalid_transform
):
    fake_itk.parameter_map["Transform"] = [invalid_transform]
    with pytest.raises(ValueError, match="supports only the rigid EulerTransform"):
        mapping._build_elastix_euler_transform(transform_file)
    assert not fake_itk.rigids


def test_euler_transform_rejects_initial_transform_chain(fake_itk, transform_file):
    fake_itk.parameter_map["InitialTransformParameterFileName"] = ['"earlier.txt"']
    with pytest.raises(ValueError, match="refuses to ignore a transform chain: earlier.txt"):
        mapping._build_elastix_euler_transform(transform_file)
    assert not fake_itk.rigids


@pytest.mark.parametrize("bad_length", [0, 3, 5, 7])
def test_euler_transform_rejects_incorrect_parameter_count(
    fake_itk, transform_file, bad_length
):
    fake_itk.parameter_map["TransformParameters"] = ["0"] * bad_length
    with pytest.raises(ValueError, match=f"found {bad_length}"):
        mapping._build_elastix_euler_transform(transform_file)


@pytest.mark.parametrize("bad_length", [0, 2, 4])
def test_euler_transform_rejects_incorrect_center_count(
    fake_itk, transform_file, bad_length
):
    fake_itk.parameter_map["CenterOfRotationPoint"] = ["0"] * bad_length
    with pytest.raises(ValueError, match="Expected a 3-D CenterOfRotationPoint"):
        mapping._build_elastix_euler_transform(transform_file)


@pytest.mark.parametrize("parameter", [
    "Transform", "TransformParameters", "CenterOfRotationPoint"
])
def test_euler_transform_rejects_missing_required_parameter(
    fake_itk, transform_file, parameter
):
    del fake_itk.parameter_map[parameter]
    with pytest.raises(ValueError, match="is missing"):
        mapping._build_elastix_euler_transform(transform_file)


@pytest.mark.parametrize("parameter", ["TransformParameters", "CenterOfRotationPoint"])
def test_euler_transform_rejects_non_numeric_parameters(
    fake_itk, transform_file, parameter
):
    fake_itk.parameter_map[parameter][0] = "not-a-number"
    with pytest.raises(ValueError, match="could not convert string to float"):
        mapping._build_elastix_euler_transform(transform_file)


# ---------------------------------------------------------------------------
# FU -> native BL continuous slice mapping using synthetic spatial geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plane,expected", [
    ("sagittal", (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)),
    ("coronal", (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
    ("axial", (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)),
])
def test_fu_to_bl_identity_mapping_for_all_planes(identity_rigid, monkeypatch, plane, expected):
    monkeypatch.setattr(mapping, "anatomical_axis", {
        "sagittal": lambda orientation, name: 0,
        "coronal": lambda orientation, name: 1,
        "axial": lambda orientation, name: 2,
    }[plane])
    result = mapping.fu_to_bl_slice_map(**mapping_args(plane_name=plane))
    assert result == expected
    assert len(identity_rigid) == 1
    assert len(identity_rigid[0].points) == len(expected)


def test_fu_to_bl_uses_FU_fixed_to_BL_moving_transform(identity_rigid, monkeypatch):
    monkeypatch.setattr(mapping, "anatomical_axis", lambda orientation, plane: 2)
    # The rigid transform is FU(LPS)->BL(LPS): +3 mm LPS Z == +3 RAS Z.
    # The BL voxel size is 2 mm, so 0,1,2,3 mm FU Z maps to 1.5,2,2.5,3 BL slices.
    identity_rigid.clear()
    result_args = mapping_args(
        bl_affine_flat=affine_flat(spacing=(1, 1, 2)),
        fu_shape=(4, 5, 4),
    )
    def build(_):
        obj = FakeRigid()
        obj.translation = (0, 0, 3)
        identity_rigid.append(obj)
        return obj
    monkeypatch.setattr(mapping, "_build_elastix_euler_transform", build)
    assert mapping.fu_to_bl_slice_map(**result_args) == (1.5, 2.0, 2.5, 3.0)


def test_fu_to_bl_uses_both_ras_lps_sign_flips(identity_rigid, monkeypatch):
    monkeypatch.setattr(mapping, "anatomical_axis", lambda orient, name: 0)
    def build(_):
        obj = FakeRigid()
        obj.translation = (2.0, 0.0, 0.0)  # LPS +2 X -> RAS -2 X
        identity_rigid.append(obj)
        return obj
    monkeypatch.setattr(mapping, "_build_elastix_euler_transform", build)
    result = mapping.fu_to_bl_slice_map(**mapping_args())
    assert result == (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    assert identity_rigid[0].points[0][0] == 0
    assert identity_rigid[0].points[1][0] == -1


def test_fu_to_bl_respects_affine_origins(identity_rigid, monkeypatch):
    monkeypatch.setattr(mapping, "anatomical_axis", lambda orient, name: 2)
    result = mapping.fu_to_bl_slice_map(**mapping_args(
        bl_affine_flat=affine_flat(origin=(0, 0, 10)),
        fu_affine_flat=affine_flat(origin=(0, 0, 12)),
        fu_shape=(3, 3, 4),
    ))
    assert result == (2.0, 3.0, 4.0, 5.0)


def test_fu_to_bl_respects_opposite_orientation_axes(identity_rigid, monkeypatch):
    seen = []
    def axis(orient, plane):
        seen.append((orient, plane))
        return 0 if orient == ("L", "P", "S") else 2
    monkeypatch.setattr(mapping, "anatomical_axis", axis)
    args = mapping_args(
        bl_orientation=("L", "P", "S"), fu_orientation=("R", "A", "S"),
        fu_shape=(3, 5, 4),
    )
    # The map is of BL axis 0, holding FU axes 0 and 1 at the volume centre.
    assert mapping.fu_to_bl_slice_map(**args) == (1.0, 1.0, 1.0, 1.0)
    assert seen == [(args["bl_orientation"], "axial"),
                    (args["fu_orientation"], "axial")]


def test_fu_to_bl_respects_nontrivial_affines(identity_rigid, monkeypatch):
    monkeypatch.setattr(mapping, "anatomical_axis", lambda orientation, plane: 2)
    bl = np.eye(4)
    bl[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 2, 0]]
    fu = np.eye(4)
    fu[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 2, 0]]
    fu[:3, 3] = [2, 0, 0]
    # Movement by 2 mm in RAS X is +2 BL native axis 2 slices.
    result = mapping.fu_to_bl_slice_map(**mapping_args(
        fu_shape=(4, 5, 3),
        bl_affine_flat=tuple(bl.ravel()), fu_affine_flat=tuple(fu.ravel()),
    ))
    assert result == (2.0, 3.0, 4.0)


@pytest.mark.parametrize("shape_field,shape", [
    ("bl_shape", (1, 2)), ("bl_shape", (1, 2, 3, 4)),
    ("bl_shape", (1, 0, 3)), ("bl_shape", (-1, 2, 3)),
    ("fu_shape", (1, 2)), ("fu_shape", (1, 2, 3, 4)),
    ("fu_shape", (1, 0, 3)), ("fu_shape", (1, 2, -3)),
])
def test_fu_to_bl_rejects_bad_shapes(shape_field, shape):
    with pytest.raises(ValueError, match="Expected a positive 3-D"):
        mapping.fu_to_bl_slice_map(**mapping_args(**{shape_field: shape}))


@pytest.mark.parametrize("field", ["bl_affine_flat", "fu_affine_flat"])
@pytest.mark.parametrize("size", [0, 9, 15, 17, 25])
def test_fu_to_bl_rejects_wrong_affine_element_count(field, size):
    with pytest.raises(ValueError, match="must each contain 16 values"):
        mapping.fu_to_bl_slice_map(**mapping_args(**{field: (1.0,) * size}))


def test_fu_to_bl_rejects_noninvertible_BL_affine(identity_rigid, monkeypatch):
    monkeypatch.setattr(mapping, "anatomical_axis", lambda orient, plane: 2)
    singular = np.eye(4)
    singular[2] = singular[1]
    with pytest.raises(ValueError, match="Original BL affine is not invertible") as exc:
        mapping.fu_to_bl_slice_map(**mapping_args(bl_affine_flat=tuple(singular.ravel())))
    assert isinstance(exc.value.__cause__, np.linalg.LinAlgError)
    assert not identity_rigid  # validation happens before loading rigid transform


def test_fu_to_bl_propagates_transform_loading_error(monkeypatch):
    monkeypatch.setattr(mapping, "anatomical_axis", lambda orient, plane: 2)
    def cannot_load(_):
        raise FileNotFoundError("missing simulated transform")
    monkeypatch.setattr(mapping, "_build_elastix_euler_transform", cannot_load)
    with pytest.raises(FileNotFoundError, match="missing simulated transform"):
        mapping.fu_to_bl_slice_map(**mapping_args())


def test_fu_to_bl_end_to_end_with_mocked_itk(
    fake_itk, transform_file, monkeypatch
):
    """Connect parameter-file loading, Euler construction, and slice mapping."""
    monkeypatch.setattr(mapping, "anatomical_axis", lambda orient, plane: 2)
    fake_itk.parameter_map["TransformParameters"] = ["0", "0", "0", "0", "0", "2"]
    result = mapping.fu_to_bl_slice_map(**mapping_args(
        fu_shape=(3, 3, 4),
        transform_path=str(transform_file),
    ))
    assert result == (2.0, 3.0, 4.0, 5.0)
    assert len(fake_itk.objects) == len(fake_itk.rigids) == 1
    assert fake_itk.objects[0].paths == [str(transform_file)]
    assert len(fake_itk.rigids[0].points) == 4


def test_fu_to_bl_euler_center_is_configured(fake_itk, transform_file, monkeypatch):
    """Ensure center/translation are both handed to the ITK rigid transform."""
    monkeypatch.setattr(mapping, "anatomical_axis", lambda orient, plane: 0)
    fake_itk.parameter_map["CenterOfRotationPoint"] = ["10", "20", "30"]
    fake_itk.parameter_map["TransformParameters"] = ["0", "0", "0", "1", "0", "0"]
    result = mapping.fu_to_bl_slice_map(**mapping_args(
        fu_shape=(4, 3, 3), transform_path=str(transform_file)
    ))
    assert result == (-1.0, 0.0, 1.0, 2.0)
    assert fake_itk.rigids[0].center == (10.0, 20.0, 30.0)
    assert fake_itk.rigids[0].translation == (1.0, 0.0, 0.0)


def test_fu_to_bl_caches_identical_calls(identity_rigid, monkeypatch):
    monkeypatch.setattr(mapping, "anatomical_axis", lambda orient, plane: 2)
    args = mapping_args()
    first = mapping.fu_to_bl_slice_map(**args)
    second = mapping.fu_to_bl_slice_map(**args)
    assert first is second
    assert len(identity_rigid) == 1
    assert mapping.fu_to_bl_slice_map.cache_info().hits == 1
    assert mapping.fu_to_bl_slice_map.cache_info().misses == 1


def test_fu_to_bl_refreshes_cache_when_transform_mtime_changes(identity_rigid, monkeypatch):
    monkeypatch.setattr(mapping, "anatomical_axis", lambda orient, plane: 2)
    a = mapping_args(transform_modified_ns=1)
    b = mapping_args(transform_modified_ns=2)
    assert mapping.fu_to_bl_slice_map(**a) == mapping.fu_to_bl_slice_map(**b)
    assert len(identity_rigid) == 2
    assert mapping.fu_to_bl_slice_map.cache_info().misses == 2


def test_fu_to_bl_returns_tuple_even_for_one_fu_slice(identity_rigid, monkeypatch):
    monkeypatch.setattr(mapping, "anatomical_axis", lambda orient, plane: 2)
    out = mapping.fu_to_bl_slice_map(**mapping_args(fu_shape=(3, 3, 1)))
    assert out == (0.0,)
    assert isinstance(out, tuple)


# ---------------------------------------------------------------------------
# Continuous BL index -> nearest slice and out-of-field-of-view flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("continuous,count,expected_index,is_outside", [
    (0.0, 10, 0, False),
    (1.0, 10, 1, False),
    (1.25, 10, 1, False),
    (1.75, 10, 2, False),
    (2.5, 10, 2, False),  # np.rint round-to-even
    (3.5, 10, 4, False),
    (-0.5, 10, 0, False), # half slice outside remains in FOV
    (9.5, 10, 9, False),
    (-0.51, 10, 0, True),
    (9.51, 10, 9, True),
    (-50.0, 10, 0, True),
    (50.0, 10, 9, True),
    (0.0, 1, 0, False),
    (-0.5, 1, 0, False),
    (0.5, 1, 0, False),
    (0.6, 1, 0, True),
])
def test_mapped_native_bl_slice_rounding_clipping_and_fov(
    continuous, count, expected_index, is_outside
):
    index, raw, outside = mapping.mapped_native_bl_slice(
        (continuous,), fu_slice_index=0, bl_slice_count=count
    )
    assert (index, raw, outside) == (expected_index, continuous, is_outside)
    assert isinstance(index, int)
    assert isinstance(raw, float)
    assert isinstance(outside, bool)


def test_mapped_native_bl_slice_selects_requested_fu_slice():
    assert mapping.mapped_native_bl_slice(
        (2.9, 4.2, 5.7), fu_slice_index=1, bl_slice_count=10
    ) == (4, 4.2, False)


@pytest.mark.parametrize("bad_count", [0, -1, -10])
def test_mapped_native_bl_slice_requires_positive_bl_slice_count(bad_count):
    with pytest.raises(ValueError, match="BL slice count must be positive"):
        mapping.mapped_native_bl_slice((0.0,), fu_slice_index=0, bl_slice_count=bad_count)


@pytest.mark.parametrize("index", [-2, -1, 1, 100])
def test_mapped_native_bl_slice_rejects_fu_index_out_of_range(index):
    with pytest.raises(IndexError, match="outside the precomputed mapping"):
        mapping.mapped_native_bl_slice((0.0,), fu_slice_index=index, bl_slice_count=4)


def test_mapped_native_bl_slice_rejects_empty_precomputed_mapping():
    with pytest.raises(IndexError, match="outside the precomputed mapping"):
        mapping.mapped_native_bl_slice((), fu_slice_index=0, bl_slice_count=4)


@pytest.mark.parametrize("not_finite", [float("nan"), float("inf"), float("-inf")])
def test_mapped_native_bl_slice_rejects_nonfinite_continuous_position(not_finite):
    with pytest.raises(ValueError, match="not finite"):
        mapping.mapped_native_bl_slice((not_finite,), fu_slice_index=0, bl_slice_count=4)


def test_public_api_explicit_exports():
    assert set(mapping.__all__) == {"fu_to_bl_slice_map", "mapped_native_bl_slice"}
