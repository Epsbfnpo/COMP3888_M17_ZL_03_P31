from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from .cohort_a_loading import LoadedVolume, PatientPairVolumes, load_nifti_volume


class RegistrationError(RuntimeError):
    """Raised when baseline-to-follow-up image registration fails."""


@dataclass(frozen=True)
class RegistrationResult:
    """Result of registering baseline CT into follow-up CT space."""

    patient_id: str
    stages: tuple[str, ...]
    registered_baseline_ct: LoadedVolume
    registered_ct_path: Path
    transform_parameter_paths: tuple[Path, ...]
    output_dir: Path
    initialization_method: str = "unknown"
    initial_translation_lps_mm: tuple[float, float, float] | None = None
    anatomical_alignment_score: float | None = None
    body_dice: float | None = None


@dataclass(frozen=True)
class AnatomicalPrealignment:
    """Content-derived fixed-FU to moving-BL translation estimate."""

    translation_ras_mm: tuple[float, float, float]
    translation_lps_mm: tuple[float, float, float]
    score: float
    overlap_fraction: float


SUPPORTED_STAGES = {"rigid", "affine"}
SUPPORTED_INITIALIZATION_METHODS = {
    "adaptive",
    "anatomical",
    "none",
    "geometrical_center",
    "center_of_gravity",
}

_ELASTIX_INITIALIZATION_NAMES = {
    "geometrical_center": "GeometricalCenter",
    "center_of_gravity": "CenterOfGravity",
}


def _validate_ct_volume(volume: LoadedVolume, *, role: str) -> None:
    """Validate basic properties needed by the registration pipeline."""
    if volume.data.ndim != 3:
        raise RegistrationError(
            f"{role} CT must be 3-D; got shape {volume.data.shape}."
        )

    if not volume.metadata.path.exists():
        raise RegistrationError(
            f"{role} CT source file does not exist: {volume.metadata.path}"
        )

    spacing = np.asarray(volume.metadata.spacing_mm, dtype=float)
    if (
        spacing.shape != (3,)
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0)
    ):
        raise RegistrationError(
            f"{role} CT has invalid voxel spacing: {volume.metadata.spacing_mm}"
        )

    affine = np.asarray(volume.metadata.affine, dtype=float)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise RegistrationError(f"{role} CT has an invalid affine matrix.")

    if not np.any(np.isfinite(volume.data)):
        raise RegistrationError(f"{role} CT contains no finite voxel intensities.")


def _normalise_stages(stages: Iterable[str]) -> tuple[str, ...]:
    normalised = tuple(str(stage).strip().lower() for stage in stages)

    if not normalised:
        raise ValueError("At least one registration stage is required.")

    unknown = sorted(set(normalised) - SUPPORTED_STAGES)
    if unknown:
        raise ValueError(
            "Unsupported registration stage(s): "
            + ", ".join(unknown)
            + ". Supported stages are: rigid, affine."
        )

    if len(set(normalised)) != len(normalised):
        raise ValueError("Registration stages must not contain duplicates.")

    if "rigid" in normalised and "affine" in normalised:
        if normalised.index("rigid") > normalised.index("affine"):
            raise ValueError("When both are used, rigid must come before affine.")

    return normalised


def _normalise_initialization_method(initialization_method: str) -> str:
    method = str(initialization_method).strip().lower().replace("-", "_")
    aliases = {
        "anatomy": "anatomical",
        "anatomy_driven": "anatomical",
        "physical": "none",
        "physical_coordinates": "none",
        "geometric_center": "geometrical_center",
        "geometricalcenter": "geometrical_center",
        "centerofgravity": "center_of_gravity",
    }
    method = aliases.get(method, method)
    if method not in SUPPORTED_INITIALIZATION_METHODS:
        raise ValueError(
            f"Unsupported initialization_method {initialization_method!r}. "
            "Supported values are: adaptive, anatomical, none, "
            "geometrical_center, center_of_gravity."
        )
    return method


def _physical_extent_mm(volume: LoadedVolume) -> np.ndarray:
    """Return the physical length of each voxel axis in millimetres."""
    shape = np.asarray(volume.data.shape, dtype=float)
    affine = np.asarray(volume.metadata.affine, dtype=float)
    voxel_axis_lengths = np.linalg.norm(affine[:3, :3], axis=0)
    return np.maximum(shape - 1.0, 0.0) * voxel_axis_lengths


def _has_materially_different_scan_extents(
    baseline_ct: LoadedVolume,
    followup_ct: LoadedVolume,
    *,
    ratio_threshold: float,
    difference_threshold_mm: float,
) -> bool:
    """Detect a field-of-view mismatch that makes centre alignment unsafe."""
    if not 0.0 < ratio_threshold <= 1.0:
        raise ValueError("extent_ratio_threshold must be in the interval (0, 1].")
    if not np.isfinite(difference_threshold_mm) or difference_threshold_mm < 0.0:
        raise ValueError("extent_difference_threshold_mm must be >= 0.")

    baseline_extent = _physical_extent_mm(baseline_ct)
    followup_extent = _physical_extent_mm(followup_ct)
    larger = np.maximum(baseline_extent, followup_extent)
    smaller = np.minimum(baseline_extent, followup_extent)
    ratios = np.divide(
        smaller,
        larger,
        out=np.ones_like(smaller),
        where=larger > 0.0,
    )
    differences = larger - smaller
    return bool(
        np.any(
            (ratios < float(ratio_threshold))
            & (differences > float(difference_threshold_mm))
        )
    )


def _resolve_initialization_method(
    baseline_ct: LoadedVolume,
    followup_ct: LoadedVolume,
    *,
    initialization_method: str,
    extent_ratio_threshold: float,
    extent_difference_threshold_mm: float,
) -> str:
    """Choose a safe initializer for the supplied pair of CT volumes."""
    method = _normalise_initialization_method(initialization_method)
    if method != "adaptive":
        return method

    if _has_materially_different_scan_extents(
        baseline_ct,
        followup_ct,
        ratio_threshold=extent_ratio_threshold,
        difference_threshold_mm=extent_difference_threshold_mm,
    ):
        return "anatomical"

    return "geometrical_center"


def _smooth_profile(values: np.ndarray, window: int = 7) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return values.copy()
    width = min(int(window), int(values.size))
    if width % 2 == 0:
        width -= 1
    if width < 3:
        return values.copy()
    kernel = np.ones(width, dtype=float) / float(width)
    padded = np.pad(values, width // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _superior_inferior_anatomy_profile(
    volume: LoadedVolume,
    *,
    inplane_stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted RAS-Z locations and content features for every CT slice."""
    if inplane_stride < 1:
        raise ValueError("inplane_stride must be >= 1.")

    affine = np.asarray(volume.metadata.affine, dtype=float)
    slice_axis = int(np.argmax(np.abs(affine[2, :3])))
    if abs(float(affine[2, slice_axis])) < 1e-6:
        raise RegistrationError(
            "Could not identify a superior-inferior CT axis from the affine."
        )

    data = np.moveaxis(np.asarray(volume.data), slice_axis, 0)
    sampled = np.asarray(
        data[:, ::inplane_stride, ::inplane_stride],
        dtype=np.float32,
    )
    finite = np.isfinite(sampled)

    # Air is near -1000 HU. These complementary features describe how much
    # soft tissue and bone occurs at each anatomical height without allowing
    # the air/background intensity to define a centre of gravity.
    tissue_fraction = np.mean(finite & (sampled > -600.0), axis=(1, 2))
    bone_fraction = np.mean(finite & (sampled > 250.0), axis=(1, 2))
    density = np.clip((sampled + 600.0) / 1600.0, 0.0, 1.0)
    density[~finite] = 0.0
    density_signal = np.mean(density, axis=(1, 2))

    center_voxel = (np.asarray(volume.data.shape, dtype=float) - 1.0) / 2.0
    z_locations = np.empty(data.shape[0], dtype=float)
    for index in range(data.shape[0]):
        point = center_voxel.copy()
        point[slice_axis] = float(index)
        z_locations[index] = float(
            (affine @ np.asarray((*point, 1.0), dtype=float))[2]
        )

    order = np.argsort(z_locations)
    z_locations = z_locations[order]
    features = np.column_stack(
        (
            _smooth_profile(tissue_fraction[order]),
            _smooth_profile(bone_fraction[order]),
            _smooth_profile(density_signal[order]),
        )
    )
    return z_locations, features


def _profile_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.size < 3 or second.size != first.size:
        return -1.0
    first_std = float(np.std(first))
    second_std = float(np.std(second))
    if first_std < 1e-8 or second_std < 1e-8:
        return 0.0
    return float(
        np.mean(
            ((first - float(np.mean(first))) / first_std)
            * ((second - float(np.mean(second))) / second_std)
        )
    )


def _body_centroid_ras(
    volume: LoadedVolume,
    *,
    z_min: float,
    z_max: float,
    inplane_stride: int,
    threshold_hu: float = -600.0,
) -> np.ndarray:
    """Return a robust body centroid over a selected RAS-Z interval."""
    affine = np.asarray(volume.metadata.affine, dtype=float)
    slice_axis = int(np.argmax(np.abs(affine[2, :3])))
    other_axes = [axis for axis in range(3) if axis != slice_axis]
    center_voxel = (np.asarray(volume.data.shape, dtype=float) - 1.0) / 2.0
    centroids: list[np.ndarray] = []

    for index in range(volume.data.shape[slice_axis]):
        voxel = center_voxel.copy()
        voxel[slice_axis] = float(index)
        center_world = affine @ np.asarray((*voxel, 1.0), dtype=float)
        if not z_min <= float(center_world[2]) <= z_max:
            continue

        image_slice = np.take(volume.data, index, axis=slice_axis)
        sampled = np.asarray(
            image_slice[::inplane_stride, ::inplane_stride]
        )
        mask = np.isfinite(sampled) & (sampled > threshold_hu)
        locations = np.argwhere(mask)
        if locations.shape[0] < 16:
            continue

        # The median is less sensitive than a mean to the CT table and arms.
        inplane_center = np.median(locations, axis=0) * inplane_stride
        voxel[other_axes[0]] = float(inplane_center[0])
        voxel[other_axes[1]] = float(inplane_center[1])
        centroids.append(
            (affine @ np.asarray((*voxel, 1.0), dtype=float))[:3]
        )

    if not centroids:
        raise RegistrationError(
            "Could not determine a CT body centroid in the matched anatomy."
        )
    return np.median(np.vstack(centroids), axis=0)


def _estimate_anatomical_prealignment(
    baseline_ct: LoadedVolume,
    followup_ct: LoadedVolume,
    *,
    min_overlap_fraction: float = 0.65,
    inplane_stride: int = 4,
) -> AnatomicalPrealignment:
    """Estimate fixed-FU to moving-BL translation from axial anatomy profiles."""
    if not 0.25 <= min_overlap_fraction <= 1.0:
        raise ValueError("min_overlap_fraction must be in [0.25, 1].")

    moving_z, moving_features = _superior_inferior_anatomy_profile(
        baseline_ct,
        inplane_stride=inplane_stride,
    )
    fixed_z, fixed_features = _superior_inferior_anatomy_profile(
        followup_ct,
        inplane_stride=inplane_stride,
    )

    moving_steps = np.diff(moving_z)
    fixed_steps = np.diff(fixed_z)
    positive_steps = np.concatenate(
        (moving_steps[moving_steps > 1e-6], fixed_steps[fixed_steps > 1e-6])
    )
    if positive_steps.size == 0:
        raise RegistrationError("CT slice locations are not physically distinct.")
    search_step_mm = max(1.0, float(np.median(positive_steps)))

    delta_min = float(moving_z[0] - fixed_z[-1])
    delta_max = float(moving_z[-1] - fixed_z[0])
    candidates = np.arange(
        delta_min,
        delta_max + 0.5 * search_step_mm,
        search_step_mm,
        dtype=float,
    )

    minimum_samples = max(
        12,
        int(
            np.ceil(
                min_overlap_fraction
                * min(int(fixed_z.size), int(moving_z.size))
            )
        ),
    )
    feature_weights = np.asarray((0.45, 0.35, 0.20), dtype=float)
    best: tuple[float, float, float] | None = None

    for delta in candidates:
        mapped_z = fixed_z + float(delta)
        valid = (mapped_z >= moving_z[0]) & (mapped_z <= moving_z[-1])
        sample_count = int(np.count_nonzero(valid))
        if sample_count < minimum_samples:
            continue

        fixed_segment = fixed_features[valid]
        moving_segment = np.column_stack(
            [
                np.interp(mapped_z[valid], moving_z, moving_features[:, index])
                for index in range(moving_features.shape[1])
            ]
        )
        correlations = np.asarray(
            [
                _profile_correlation(
                    fixed_segment[:, index],
                    moving_segment[:, index],
                )
                for index in range(fixed_segment.shape[1])
            ],
            dtype=float,
        )
        derivative_score = 0.5 * (
            _profile_correlation(
                np.gradient(fixed_segment[:, 0]),
                np.gradient(moving_segment[:, 0]),
            )
            + _profile_correlation(
                np.gradient(fixed_segment[:, 1]),
                np.gradient(moving_segment[:, 1]),
            )
        )
        overlap_fraction = sample_count / float(
            min(int(fixed_z.size), int(moving_z.size))
        )
        score = float(
            np.dot(feature_weights, correlations)
            + 0.25 * derivative_score
            + 0.05 * min(overlap_fraction, 1.0)
        )
        if best is None or score > best[0]:
            best = (score, float(delta), overlap_fraction)

    if best is None:
        raise RegistrationError(
            "Could not find enough anatomical Z overlap between BL and FU CT."
        )

    mapped_fixed_z = fixed_z + best[1]
    fixed_valid = (mapped_fixed_z >= moving_z[0]) & (
        mapped_fixed_z <= moving_z[-1]
    )
    fixed_center = _body_centroid_ras(
        followup_ct,
        z_min=float(fixed_z[fixed_valid][0]),
        z_max=float(fixed_z[fixed_valid][-1]),
        inplane_stride=inplane_stride,
    )
    moving_center = _body_centroid_ras(
        baseline_ct,
        z_min=float(mapped_fixed_z[fixed_valid][0]),
        z_max=float(mapped_fixed_z[fixed_valid][-1]),
        inplane_stride=inplane_stride,
    )
    translation_ras = moving_center - fixed_center
    translation_ras[2] = best[1]
    translation_lps = np.asarray(
        (-translation_ras[0], -translation_ras[1], translation_ras[2]),
        dtype=float,
    )
    return AnatomicalPrealignment(
        translation_ras_mm=tuple(float(value) for value in translation_ras),
        translation_lps_mm=tuple(float(value) for value in translation_lps),
        score=float(best[0]),
        overlap_fraction=float(best[2]),
    )


def _validate_required_ratio_of_valid_samples(
    required_ratio_of_valid_samples: float | None,
) -> None:
    if required_ratio_of_valid_samples is None:
        return

    value = float(required_ratio_of_valid_samples)
    if not np.isfinite(value) or not (0.0 < value <= 1.0):
        raise ValueError(
            "required_ratio_of_valid_samples must be in the interval (0, 1]."
        )


def _build_parameter_object(
    stages: tuple[str, ...],
    *,
    number_of_resolutions: int,
    maximum_iterations: int,
    number_of_spatial_samples: int,
    initialization_method: str,
    required_ratio_of_valid_samples: float | None = None,
):
    """
    Build conservative ITKElastix parameter maps.

    required_ratio_of_valid_samples is normally left as None, which preserves
    the Elastix default.
    """
    if number_of_resolutions < 1:
        raise ValueError("number_of_resolutions must be >= 1")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be >= 1")
    if number_of_spatial_samples < 128:
        raise ValueError("number_of_spatial_samples must be >= 128")

    _validate_required_ratio_of_valid_samples(required_ratio_of_valid_samples)

    try:
        import itk
    except ImportError as exc:
        raise RegistrationError(
            "ITKElastix is not installed. Install project requirements or run "
            "'python -m pip install itk-elastix'."
        ) from exc

    parameter_object = itk.ParameterObject.New()

    method = _normalise_initialization_method(initialization_method)
    if method == "adaptive":
        raise ValueError(
            "Adaptive initialization must be resolved before building the "
            "Elastix parameter object."
        )

    for stage_index, stage in enumerate(stages):
        parameter_map = parameter_object.GetDefaultParameterMap(stage)

        parameter_map["Metric"] = ["AdvancedMattesMutualInformation"]

        parameter_map["NumberOfResolutions"] = [str(number_of_resolutions)]
        parameter_map["MaximumNumberOfIterations"] = [str(maximum_iterations)]
        parameter_map["NumberOfSpatialSamples"] = [str(number_of_spatial_samples)]

        # Only the first stage is initialized. Later stages already receive
        # the preceding stage as their initial transform.
        use_automatic_initialization = (
            stage_index == 0 and method in _ELASTIX_INITIALIZATION_NAMES
        )
        parameter_map["AutomaticTransformInitialization"] = [
            "true" if use_automatic_initialization else "false"
        ]
        if use_automatic_initialization:
            parameter_map["AutomaticTransformInitializationMethod"] = [
                _ELASTIX_INITIALIZATION_NAMES[method]
            ]

        parameter_map["UseDirectionCosines"] = ["true"]
        parameter_map["DefaultPixelValue"] = ["-1024"]
        parameter_map["WriteResultImage"] = ["true"]

        # Do not silently weaken this criterion for unequal scan extents. A
        # wrong transform can otherwise finish successfully with little valid
        # overlap. Anatomy-driven initialization should create real overlap.
        if required_ratio_of_valid_samples is not None:
            parameter_map["RequiredRatioOfValidSamples"] = [
                str(float(required_ratio_of_valid_samples))
            ]

        parameter_object.AddParameterMap(parameter_map)

    return parameter_object


def _remove_previous_outputs(output_dir: Path) -> None:
    """Remove only files generated by this module during an earlier run."""
    registered_path = output_dir / "registered_baseline_ct.nii.gz"
    if registered_path.exists():
        registered_path.unlink()

    for path in output_dir.glob("TransformParameters.*.txt"):
        path.unlink()

    for path in output_dir.glob("result.*"):
        if path.is_file():
            path.unlink()

    log_path = output_dir / "elastix.log"
    if log_path.exists():
        log_path.unlink()


def _elastix_log_tail(output_dir: Path, *, line_count: int = 12) -> str:
    """Return the useful tail of the Elastix log for an error message."""
    log_path = output_dir / "elastix.log"
    if not log_path.is_file():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    useful = [line.strip() for line in lines if line.strip()]
    if not useful:
        return ""
    return "\nElastix log tail:\n" + "\n".join(useful[-line_count:])


def _make_itk_body_mask(itk, image, *, threshold_hu: float):
    """Build an unsigned-byte CT body mask for Elastix sampling."""
    mask_type = itk.Image[itk.UC, 3]
    threshold_filter = itk.BinaryThresholdImageFilter[
        type(image),
        mask_type,
    ].New()
    threshold_filter.SetInput(image)
    threshold_filter.SetLowerThreshold(float(threshold_hu))
    threshold_filter.SetUpperThreshold(1.0e6)
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)
    threshold_filter.UpdateLargestPossibleRegion()
    return threshold_filter.GetOutput()


def _parameter_values_from_text(text: str, name: str) -> tuple[str, ...]:
    match = re.search(
        rf"^\({re.escape(name)}\s+(.+?)\)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise RegistrationError(
            f"Elastix transform is missing parameter {name!r}."
        )
    return tuple(match.group(1).replace('"', "").split())


def _set_parameter_line(text: str, name: str, values: str) -> str:
    """Replace one Elastix parameter line, appending it if absent."""
    replacement = f"({name} {values})"
    pattern = re.compile(
        rf"^\({re.escape(name)}(?:\s+.*?)?\)\s*$",
        flags=re.MULTILINE,
    )
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)
    separator = "" if text.endswith("\n") else "\n"
    return text + separator + replacement + "\n"


def _flatten_rigid_combination_transform(
    itk,
    registration,
    transform_path: Path,
) -> tuple[float, ...]:
    """Save external initialization plus optimized rigid motion as one Euler."""
    try:
        combination = registration.GetCombinationTransform()
        origin = np.zeros(3, dtype=float)
        mapped_origin = np.asarray(
            combination.TransformPoint(tuple(origin)),
            dtype=float,
        )
        matrix = np.column_stack(
            [
                np.asarray(
                    combination.TransformPoint(tuple(np.eye(3)[axis])),
                    dtype=float,
                )
                - mapped_origin
                for axis in range(3)
            ]
        )

        # Remove only floating-point drift. A genuine non-rigid/non-Euler
        # result must not be silently written as a rigid transform.
        left, _, right = np.linalg.svd(matrix)
        rotation = left @ right
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        if float(np.max(np.abs(matrix - rotation))) > 1.0e-4:
            raise RegistrationError(
                "The combined initialization and registration transform is "
                "not rigid and cannot be flattened safely."
            )

        text = transform_path.read_text(encoding="utf-8")
        centre_values = _parameter_values_from_text(
            text,
            "CenterOfRotationPoint",
        )
        if len(centre_values) != 3:
            raise RegistrationError(
                "Elastix rigid transform has an invalid centre of rotation."
            )
        centre = tuple(float(value) for value in centre_values)

        euler = itk.Euler3DTransform[itk.D].New()
        euler.SetComputeZYX(False)
        euler.SetCenter(centre)
        matrix_value = (
            itk.matrix_from_array(rotation)
            if hasattr(itk, "matrix_from_array")
            else itk.GetMatrixFromArray(rotation)
        )
        euler.SetMatrix(matrix_value)
        euler.SetOffset(tuple(float(value) for value in mapped_origin))
        parameters = tuple(float(value) for value in euler.GetParameters())

        for point in (
            np.zeros(3, dtype=float),
            np.asarray((100.0, 0.0, 0.0)),
            np.asarray((0.0, 100.0, 0.0)),
            np.asarray((0.0, 0.0, 100.0)),
        ):
            expected = np.asarray(
                combination.TransformPoint(tuple(point)),
                dtype=float,
            )
            actual = np.asarray(euler.TransformPoint(tuple(point)), dtype=float)
            if not np.allclose(expected, actual, rtol=0.0, atol=1.0e-3):
                raise RegistrationError(
                    "Flattened rigid transform does not reproduce the full "
                    "Elastix transform chain."
                )

        text = _set_parameter_line(
            text,
            "TransformParameters",
            " ".join(f"{value:.17g}" for value in parameters),
        )
        text = _set_parameter_line(
            text,
            "CenterOfRotationPoint",
            " ".join(f"{value:.17g}" for value in centre),
        )
        text = _set_parameter_line(
            text,
            "InitialTransformParameterFileName",
            '"NoInitialTransform"',
        )
        text = _set_parameter_line(text, "ComputeZYX", '"false"')
        temporary_path = transform_path.with_name(transform_path.name + ".tmp")
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(transform_path)
        return parameters
    except RegistrationError:
        raise
    except Exception as exc:
        raise RegistrationError(
            "Could not flatten the anatomy initializer and optimized rigid "
            f"transform into one reusable Euler transform: {exc}"
        ) from exc


def _ct_body_dice(
    first: np.ndarray,
    second: np.ndarray,
    *,
    threshold_hu: float,
) -> float:
    """Return a coarse Dice score for two CT body masks on the same grid."""
    first_array = np.asarray(first)
    second_array = np.asarray(second)
    if first_array.shape != second_array.shape:
        return 0.0
    stride = tuple(
        max(1, int(np.ceil(size / 192.0))) for size in first_array.shape
    )
    slices = tuple(slice(None, None, step) for step in stride)
    first_mask = np.isfinite(first_array[slices]) & (
        first_array[slices] > threshold_hu
    )
    second_mask = np.isfinite(second_array[slices]) & (
        second_array[slices] > threshold_hu
    )
    denominator = int(np.count_nonzero(first_mask)) + int(
        np.count_nonzero(second_mask)
    )
    if denominator == 0:
        return 0.0
    intersection = int(np.count_nonzero(first_mask & second_mask))
    return 2.0 * intersection / float(denominator)


def _transformed_output_needs_update(
    output_path: Path,
    source_path: Path,
    transform_path: Path,
) -> bool:
    """Return whether a Transformix output is absent or older than an input."""
    if not output_path.is_file():
        return True

    try:
        output_modified_ns = output_path.stat().st_mtime_ns
        return (
            source_path.stat().st_mtime_ns > output_modified_ns
            or transform_path.stat().st_mtime_ns > output_modified_ns
        )
    except OSError:
        # Let the caller regenerate the output and report a useful transform
        # error instead of treating an unreadable cache entry as valid.
        return True


def _transform_volume_to_followup(
    source_volume: LoadedVolume,
    *,
    transform_path: str | Path,
    output_path: str | Path,
    source_pixel_kind: str,
    interpolation_order: int,
    result_pixel_type: str,
    role: str,
    force: bool = False,
) -> Path:
    """Apply a saved BL-to-FU Elastix transform to one source volume."""
    transform = Path(transform_path).expanduser()
    output = Path(output_path).expanduser()
    source = Path(source_volume.metadata.path).expanduser()

    if not source.is_file():
        raise RegistrationError(f"{role} source file not found: {source}")
    if not transform.is_file():
        raise RegistrationError(f"Rigid transform not found: {transform}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if not force and not _transformed_output_needs_update(
        output,
        source,
        transform,
    ):
        return output

    try:
        import itk
    except ImportError as exc:
        raise RegistrationError(
            f"ITKElastix is required to transform {role}."
        ) from exc

    if source_pixel_kind == "unsigned_short":
        source_pixel_type = itk.US
    elif source_pixel_kind == "float":
        source_pixel_type = itk.F
    else:
        raise ValueError(
            f"Unsupported Transformix source pixel kind: {source_pixel_kind!r}"
        )

    try:
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(str(transform))
        parameter_object.SetParameter(
            0,
            "ResampleInterpolator",
            (
                "FinalNearestNeighborInterpolator"
                if interpolation_order == 0
                else "FinalBSplineInterpolator"
            ),
        )
        parameter_object.SetParameter(
            0,
            "FinalBSplineInterpolationOrder",
            str(interpolation_order),
        )
        parameter_object.SetParameter(0, "DefaultPixelValue", "0")
        parameter_object.SetParameter(
            0,
            "ResultImagePixelType",
            result_pixel_type,
        )

        moving_image = itk.imread(str(source), source_pixel_type)
        transformix = itk.TransformixFilter.New(moving_image)
        transformix.SetTransformParameterObject(parameter_object)
        transformix.SetLogToConsole(False)
        if hasattr(transformix, "SetLogToFile"):
            transformix.SetLogToFile(False)
        transformix.UpdateLargestPossibleRegion()
        itk.imwrite(transformix.GetOutput(), str(output))
    except Exception as exc:
        raise RegistrationError(
            f"Could not transform {role} into FU space: {exc}"
        ) from exc

    if not output.is_file():
        raise RegistrationError(
            f"Transformix completed but did not write the expected {role} "
            f"output: {output}"
        )
    return output


def warp_baseline_mask_to_fu(
    baseline_mask: LoadedVolume,
    *,
    transform_path: str | Path,
    registered_mask_path: str | Path,
    force: bool = False,
) -> Path:
    """Warp a BL label mask into FU space using nearest-neighbour sampling."""
    return _transform_volume_to_followup(
        baseline_mask,
        transform_path=transform_path,
        output_path=registered_mask_path,
        source_pixel_kind="unsigned_short",
        interpolation_order=0,
        result_pixel_type="unsigned short",
        role="baseline lesion mask",
        force=force,
    )


def warp_baseline_pet_to_fu(
    baseline_pet: LoadedVolume,
    *,
    transform_path: str | Path,
    registered_pet_path: str | Path,
    force: bool = False,
) -> Path:
    """Warp continuous BL PET intensities into FU space using linear sampling."""
    return _transform_volume_to_followup(
        baseline_pet,
        transform_path=transform_path,
        output_path=registered_pet_path,
        source_pixel_kind="float",
        interpolation_order=1,
        result_pixel_type="float",
        role="baseline PET",
        force=force,
    )


def register_ct_pair(
    baseline_ct: LoadedVolume,
    followup_ct: LoadedVolume,
    *,
    patient_id: str = "<unknown>",
    output_dir: str | Path,
    stages: Iterable[str] = ("rigid", "affine"),
    number_of_resolutions: int = 3,
    maximum_iterations: int = 256,
    number_of_spatial_samples: int = 4096,
    initialization_method: str = "adaptive",
    extent_ratio_threshold: float = 0.75,
    extent_difference_threshold_mm: float = 80.0,
    anatomical_min_overlap_fraction: float = 0.65,
    anatomical_profile_stride: int = 4,
    body_mask_threshold_hu: float = -600.0,
    minimum_body_dice: float = 0.20,
    required_ratio_of_valid_samples: float | None = None,
    log_to_console: bool = False,
    overwrite: bool = True,
) -> RegistrationResult:
    """
    Register baseline CT into follow-up CT space using ITKElastix.

    initialization_method:
        ``adaptive`` uses geometrical-centre initialization only when BL and
        FU have similar physical extents. For substantially different scan
        extents it matches superior-inferior CT anatomy profiles and supplies
        the resulting translation as an explicit Elastix initial transform.
        Set an explicit method to override this decision.

    required_ratio_of_valid_samples:
        Optional Elastix RequiredRatioOfValidSamples value.  Keep this as None
        for the normal registration pass.  It exists so callers can explicitly
        request a different value. Anatomy-driven initialization never lowers
        it automatically.
    """
    _validate_ct_volume(baseline_ct, role="Baseline")
    _validate_ct_volume(followup_ct, role="Follow-up")
    normalised_stages = _normalise_stages(stages)
    resolved_initialization_method = _resolve_initialization_method(
        baseline_ct,
        followup_ct,
        initialization_method=initialization_method,
        extent_ratio_threshold=extent_ratio_threshold,
        extent_difference_threshold_mm=extent_difference_threshold_mm,
    )
    if not np.isfinite(body_mask_threshold_hu):
        raise ValueError("body_mask_threshold_hu must be finite.")
    if not 0.0 <= minimum_body_dice <= 1.0:
        raise ValueError("minimum_body_dice must be in [0, 1].")

    anatomical_prealignment: AnatomicalPrealignment | None = None
    if resolved_initialization_method == "anatomical":
        if normalised_stages != ("rigid",):
            raise ValueError(
                "Anatomy-driven initialization currently requires a "
                "rigid-only registration stage."
            )
        anatomical_prealignment = _estimate_anatomical_prealignment(
            baseline_ct,
            followup_ct,
            min_overlap_fraction=anatomical_min_overlap_fraction,
            inplane_stride=anatomical_profile_stride,
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        _remove_previous_outputs(out_dir)
    else:
        existing = list(out_dir.glob("TransformParameters.*.txt"))
        registered_path = out_dir / "registered_baseline_ct.nii.gz"
        if existing or registered_path.exists():
            raise RegistrationError(
                f"Registration outputs already exist in '{out_dir}'. "
                "Use overwrite=True or choose another output directory."
            )

    parameter_object = _build_parameter_object(
        normalised_stages,
        number_of_resolutions=number_of_resolutions,
        maximum_iterations=maximum_iterations,
        number_of_spatial_samples=number_of_spatial_samples,
        initialization_method=resolved_initialization_method,
        required_ratio_of_valid_samples=required_ratio_of_valid_samples,
    )

    try:
        import itk

        # Fixed = follow-up; moving = baseline.
        fixed_image = itk.imread(str(followup_ct.metadata.path), itk.F)
        moving_image = itk.imread(str(baseline_ct.metadata.path), itk.F)

        registration = itk.ElastixRegistrationMethod.New(
            fixed_image,
            moving_image,
        )
        registration.SetParameterObject(parameter_object)
        registration.SetOutputDirectory(str(out_dir))
        registration.SetLogToConsole(bool(log_to_console))

        if hasattr(registration, "SetLogToFile"):
            registration.SetLogToFile(True)

        # Keep these objects referenced until Update completes: some ITK
        # wrappers otherwise release temporary Python-owned pipeline objects.
        initial_transform = None
        fixed_mask = None
        moving_mask = None
        if anatomical_prealignment is not None:
            initial_transform = itk.TranslationTransform[itk.D, 3].New()
            initial_transform.SetOffset(
                anatomical_prealignment.translation_lps_mm
            )
            registration.SetExternalInitialTransform(initial_transform)

            fixed_mask = _make_itk_body_mask(
                itk,
                fixed_image,
                threshold_hu=body_mask_threshold_hu,
            )
            moving_mask = _make_itk_body_mask(
                itk,
                moving_image,
                threshold_hu=body_mask_threshold_hu,
            )
            registration.SetFixedMask(fixed_mask)
            registration.SetMovingMask(moving_mask)

        registration.UpdateLargestPossibleRegion()

        if anatomical_prealignment is not None:
            generated_transforms = sorted(
                out_dir.glob("TransformParameters.*.txt")
            )
            if len(generated_transforms) != 1:
                raise RegistrationError(
                    "Anatomy-driven rigid registration did not produce "
                    "exactly one transform parameter file."
                )
            _flatten_rigid_combination_transform(
                itk,
                registration,
                generated_transforms[0],
            )

        registered_itk = registration.GetOutput()

        registered_path = out_dir / "registered_baseline_ct.nii.gz"
        itk.imwrite(registered_itk, str(registered_path))

    except Exception as exc:
        anatomy_details = ""
        if anatomical_prealignment is not None:
            anatomy_details = (
                "; anatomy initial translation LPS mm="
                f"{anatomical_prealignment.translation_lps_mm}, "
                f"profile score={anatomical_prealignment.score:.3f}, "
                f"overlap={anatomical_prealignment.overlap_fraction:.3f}"
            )
        raise RegistrationError(
            f"CT registration failed for patient '{patient_id}' using "
            f"initialization '{resolved_initialization_method}'"
            f"{anatomy_details}: {exc}"
            f"{_elastix_log_tail(out_dir)}"
        ) from exc

    if not registered_path.exists():
        raise RegistrationError(
            "Registration completed but no registered CT was written: "
            f"{registered_path}"
        )

    transform_paths = tuple(
        sorted(
            out_dir.glob("TransformParameters.*.txt"),
            key=lambda path: path.name,
        )
    )

    if len(transform_paths) != len(normalised_stages):
        raise RegistrationError(
            "Registration completed, but the expected transform parameter files "
            f"were not found. Expected {len(normalised_stages)}, found "
            f"{len(transform_paths)} in '{out_dir}'."
        )

    registered_volume = load_nifti_volume(
        registered_path,
        role=f"{patient_id} registered BL CT",
    )

    if registered_volume.data.shape != followup_ct.data.shape:
        raise RegistrationError(
            "Registered BL CT does not match FU CT shape: "
            f"{registered_volume.data.shape} versus {followup_ct.data.shape}."
        )

    if not np.allclose(
        registered_volume.metadata.affine,
        followup_ct.metadata.affine,
        rtol=0.0,
        atol=1e-3,
    ):
        raise RegistrationError(
            "Registered BL CT does not match the FU CT affine geometry."
        )

    body_dice = _ct_body_dice(
        registered_volume.data,
        followup_ct.data,
        threshold_hu=body_mask_threshold_hu,
    )
    if body_dice < minimum_body_dice:
        raise RegistrationError(
            "Registration was rejected because the registered BL and FU body "
            f"masks have insufficient overlap (Dice={body_dice:.3f}, minimum="
            f"{minimum_body_dice:.3f}). The transform files were retained for "
            "diagnosis but must not be used for lesion mapping."
        )

    return RegistrationResult(
        patient_id=str(patient_id),
        stages=normalised_stages,
        registered_baseline_ct=registered_volume,
        registered_ct_path=registered_path.resolve(),
        transform_parameter_paths=tuple(
            path.resolve() for path in transform_paths
        ),
        output_dir=out_dir.resolve(),
        initialization_method=resolved_initialization_method,
        initial_translation_lps_mm=(
            anatomical_prealignment.translation_lps_mm
            if anatomical_prealignment is not None
            else None
        ),
        anatomical_alignment_score=(
            anatomical_prealignment.score
            if anatomical_prealignment is not None
            else None
        ),
        body_dice=body_dice,
    )


def register_patient_ct(
    pair: PatientPairVolumes,
    *,
    output_dir: str | Path,
    stages: Iterable[str] = ("rigid",),
    number_of_resolutions: int = 3,
    maximum_iterations: int = 256,
    number_of_spatial_samples: int = 4096,
    initialization_method: str = "adaptive",
    extent_ratio_threshold: float = 0.75,
    extent_difference_threshold_mm: float = 80.0,
    anatomical_min_overlap_fraction: float = 0.65,
    anatomical_profile_stride: int = 4,
    body_mask_threshold_hu: float = -600.0,
    minimum_body_dice: float = 0.20,
    required_ratio_of_valid_samples: float | None = None,
    log_to_console: bool = False,
    overwrite: bool = True,
) -> RegistrationResult:
    """Convenience wrapper for a loaded Cohort A patient pair."""
    if pair.baseline.ct is None:
        raise RegistrationError(
            f"Patient '{pair.patient_id}' has no loaded baseline CT."
        )
    if pair.followup.ct is None:
        raise RegistrationError(
            f"Patient '{pair.patient_id}' has no loaded follow-up CT."
        )

    return register_ct_pair(
        pair.baseline.ct,
        pair.followup.ct,
        patient_id=pair.patient_id,
        output_dir=output_dir,
        stages=stages,
        number_of_resolutions=number_of_resolutions,
        maximum_iterations=maximum_iterations,
        number_of_spatial_samples=number_of_spatial_samples,
        initialization_method=initialization_method,
        extent_ratio_threshold=extent_ratio_threshold,
        extent_difference_threshold_mm=extent_difference_threshold_mm,
        anatomical_min_overlap_fraction=anatomical_min_overlap_fraction,
        anatomical_profile_stride=anatomical_profile_stride,
        body_mask_threshold_hu=body_mask_threshold_hu,
        minimum_body_dice=minimum_body_dice,
        required_ratio_of_valid_samples=required_ratio_of_valid_samples,
        log_to_console=log_to_console,
        overwrite=overwrite,
    )
