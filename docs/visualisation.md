# Visualise Matched Lesion Correspondences
This change covers matching BL and FU lesions. 
The matching algorithm, matching-details table and summary metrics remain unchanged.

## Use
Install the packages listed in `requirements.txt`. Then run this command from the project root:
```sh
streamlit run tools/align_longitudinal_patient_v4_mapped.py
```

The sidebar lets you select the patient, registration outputs, matching results and aligned feature csv. 
Choose a record under **Select BL lesion / correspondence (including appearing FU lesions)**. 
2 panels will show selected BL and FU ids, with their segmentation masks overlaid on CT.
The correspondence section always uses CT, even when the main viewer uses PET. 
The plane control offers axial, coronal and sagittal views.
The overlay colour and label identify each outcome:

- Cyan marks automatically matched BL and FU lesions, including merging cases.
- Amber marks an appearing FU lesion. The BL panel states that no corresponding lesion exists.
- Pink marks a disappearing BL lesion. The FU panel states that no corresponding lesion exists.

Each panel shows the selected lesion's largest cross-section. 
This keeps its mask visible when the centroid lies outside the segmented region. 
The panels display slice numbers and centroid coordinates, and may use different slices. 
Correspondence masks stay visible when the main viewer's mask toggle is off.

## Mask identity

The feature records retain `component_index`, `connectivity`, `mask_path` and `mask_role`. 
These fields link each selected lesion to its mask.

For batch-generated BL features, the viewer uses the saved native component labels after rigid transformation. 
Separate fragments with the same native ID remain one selection. 
For other masks, the viewer repeats the feature extractor's connected-component labelling. 
Older feature files without mask paths use the viewer's aligned masks.

The viewer reports missing component indices, missing files, geometry differences and centroid mismatches. 
These checks help prevent it from highlighting the wrong lesion.

The mask paths in the feature CSV must remain accessible. 
The program resolves relative paths from the project root. 
After moving outputs between machines, update these paths or regenerate the features.

## Verification

The automated suite passed **63 tests** using Python 3.12 and synthetic fixtures. Run the existing tests and the added P31-20 tests with:
```sh
python -m pytest -q
```

The added tests cover component selection, separate fragments sharing a BL label, and all three slice axes. 
They also cover empty centroid slices, invalid geometry or features, outcome colours, and panels with or without corresponding lesions.

The original archive contained no patient NIfTI data. Later testing used an external dataset, and four of five patients completed processing. 
Patient `0e51011a4c` was excluded from matching after upstream mask resampling lost two small BL lesions. This issue remains unresolved.
