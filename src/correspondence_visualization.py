from pathlib import Path
import numpy as np
from .cohort_a_loading import load_nifti_volume
from .lesion_components import extract_individual_lesions
from .lesion_visualization import validate_image_mask_alignment, select_lesion_slice
from .multiplanar_view import extract_display_slice

OUTCOME_STYLES = {
    "MERGING": ("Automatically matched pair — merging", (0, 220, 255)),
    "MATCHED": ("Automatically matched pair", (0, 220, 255)),
    "NEW": ("Appearing lesion — no BL correspondence", (255, 190, 0)),
    "DISAPPEARING": ("Disappearing lesion — no FU correspondence", (255, 80, 190)),
}

def selected_lesion_mask(location, reference, fallback=None):
    #use saved native BL component ids/repeat component extraction
    #centroids mayb lie outside lesions, resolve relative mask paths from project root
    if location.mask_path:
        volume = load_nifti_volume(Path(location.mask_path).expanduser(), preserve_dtype=True)
    elif fallback is not None:
        volume = fallback
    else:
        raise ValueError("No segmentation mask is available for this lesion.")
    validate_image_mask_alignment(reference, volume)
    if location.component_index is None:
        raise ValueError("Feature CSV needs component_index to identify the lesion mask.")
    if location.mask_role == "BL_native_component_IDs_rigidly_resampled_to_FU":
        labels = volume.data
    else:
        labels = extract_individual_lesions(
            volume, patient_id="display", timepoint=location.timepoint,
            connectivity=location.connectivity,
        ).labelled_mask
    selected = labels == location.component_index
    coords = np.argwhere(selected)
    if not len(coords):
        raise ValueError(f"Mask does not contain lesion {location.lesion_id}.")
    if not np.allclose(coords.mean(axis=0), location.centroid_voxel, atol=0.01, rtol=0):
        raise ValueError("Mask and feature centroid disagree; regenerate features for these scans.")
    return selected

def lesion_display_slice(mask, axis):
    #use largest cross-section so hollow lesions remain visible
    index = select_lesion_slice(mask, axis=axis)
    return extract_display_slice(mask, axis=axis, index=index), index
