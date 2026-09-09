from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


SUPPORTED_STAGES = {"rigid", "affine"}


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
    required_ratio_of_valid_samples: float | None = None,
):
    """
    Build conservative ITKElastix parameter maps.

    required_ratio_of_valid_samples is normally left as None, which preserves
    the Elastix default.  A lower explicit value may be used by a controlled
    fallback for BL/FU scans with substantially different fields of view.
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

    for stage in stages:
        parameter_map = parameter_object.GetDefaultParameterMap(stage)

        parameter_map["Metric"] = ["AdvancedMattesMutualInformation"]

        parameter_map["NumberOfResolutions"] = [str(number_of_resolutions)]
        parameter_map["MaximumNumberOfIterations"] = [str(maximum_iterations)]
        parameter_map["NumberOfSpatialSamples"] = [str(number_of_spatial_samples)]

        parameter_map["AutomaticTransformInitialization"] = ["true"]
        parameter_map["AutomaticTransformInitializationMethod"] = [
            "GeometricalCenter"
        ]

        parameter_map["UseDirectionCosines"] = ["true"]
        parameter_map["DefaultPixelValue"] = ["-1024"]
        parameter_map["WriteResultImage"] = ["true"]

        # IMPORTANT:
        # Do not lower this for normal patients.  The batch tool only supplies
        # an explicit ratio when its normal rigid pass has already failed.
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
    required_ratio_of_valid_samples: float | None = None,
    log_to_console: bool = False,
    overwrite: bool = True,
) -> RegistrationResult:
    """
    Register baseline CT into follow-up CT space using ITKElastix.

    required_ratio_of_valid_samples:
        Optional Elastix RequiredRatioOfValidSamples value.  Keep this as None
        for the normal registration pass.  It exists so callers can explicitly
        request a controlled low-overlap fallback without changing the default
        behaviour of every patient.
    """
    _validate_ct_volume(baseline_ct, role="Baseline")
    _validate_ct_volume(followup_ct, role="Follow-up")
    normalised_stages = _normalise_stages(stages)

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
            registration.SetLogToFile(False)

        registration.UpdateLargestPossibleRegion()

        registered_itk = registration.GetOutput()

        registered_path = out_dir / "registered_baseline_ct.nii.gz"
        itk.imwrite(registered_itk, str(registered_path))

    except Exception as exc:
        raise RegistrationError(
            f"CT registration failed for patient '{patient_id}': {exc}"
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

    return RegistrationResult(
        patient_id=str(patient_id),
        stages=normalised_stages,
        registered_baseline_ct=registered_volume,
        registered_ct_path=registered_path.resolve(),
        transform_parameter_paths=tuple(
            path.resolve() for path in transform_paths
        ),
        output_dir=out_dir.resolve(),
    )


def register_patient_ct(
    pair: PatientPairVolumes,
    *,
    output_dir: str | Path,
    stages: Iterable[str] = ("rigid",),
    number_of_resolutions: int = 3,
    maximum_iterations: int = 256,
    number_of_spatial_samples: int = 4096,
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
        required_ratio_of_valid_samples=required_ratio_of_valid_samples,
        log_to_console=log_to_console,
        overwrite=overwrite,
    )
