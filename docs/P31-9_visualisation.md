# P31-9 — Visualise PET/CT and Lesion Overlays

The supplied Cohort A dataset contains CT images only. 
So this task uses CT for real data inspection. 
And this tool also supports PET if valid PET paths exist.

```bash
python tools/visualise_cohort_a_overlays.py \
  --pairs outputs/cohort_a_subset/cohort_a_subset_pairs.csv \
  --data-root /path/to/Longitudinal_CT_v2 \
  --out-dir outputs/P31-9 \
  --modality ct \
  --max-patients 3
```

The tool selects the axial slice containing the most lesion voxels. 
Users can also specify slices with `--baseline-slice` and `--followup-slice`.
Each figure displays bl and fu images side by side. 
And the top row shows the original CT images, the bottom row shows the corresponding lesion masks overlaid on the CT images.
Before rendering, the tool checks the image and mask shapes. 
And it also checks their affine matrices to prevent incorrect overlays.
The generated PNG files were manually inspected for three Cohort A patients. 
And the results are recorded in `manual_inspection.csv` as `PASS` or `FAIL`.
