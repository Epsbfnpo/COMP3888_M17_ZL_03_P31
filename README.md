# COMP3888_M17_ZL_03_P31

P31 longitudinal PET/CT lesion quantification project.

## Cohort A subset manifest

This repository includes tooling to prepare a small Cohort A patient subset for
pipeline development without requiring the full dataset. The manifest records
baseline and follow-up scans separately, including CT, PET, lesion-mask, and
available expert-reference paths, while explicitly flagging missing files.

## Environment setup

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Generate the manifest

Place or mount the Cohort A subset at a known local path, for example
`data/cohort_a/`. The script supports Cohort A folders with `inputsTr/` plus
either `targetsTr/` or `outputsTr/` masks. The data folder is local/private and
is ignored by Git.

```powershell
python tools/prepare_cohort_a_subset.py `
  --root data/cohort_a `
  --out-dir outputs/cohort_a_subset `
  --max-patients 5
```

By default, CSV paths are written relative to `--root`, so the manifest stays
portable across cloned environments that keep the same dataset layout.

To create a copied portable subset beside the manifest:

```powershell
python tools/prepare_cohort_a_subset.py `
  --root data/cohort_a `
  --out-dir outputs/cohort_a_subset `
  --max-patients 5 `
  --copy-files `
  --path-mode relative-to-manifest
```

## Load the manifest

```python
import pandas as pd

scan_manifest = pd.read_csv("outputs/cohort_a_subset/cohort_a_subset_manifest.csv")
pair_manifest = pd.read_csv("outputs/cohort_a_subset/cohort_a_subset_pairs.csv")
```
