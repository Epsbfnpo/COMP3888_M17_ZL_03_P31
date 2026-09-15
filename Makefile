PYTHON ?= python3
DATA ?= $(if $(COHORT_A_ROOT),$(COHORT_A_ROOT),data/cohort_a)
OUT_DIR ?= outputs/cohort_a_subset
MAX_PATIENTS ?= 5
PAIRS ?= $(OUT_DIR)/cohort_a_subset_pairs.csv
LESIONS_DIR ?= outputs/cohort_a_lesions

.PHONY: setup manifest lesions pipeline test
.DEFAULT_GOAL := pipeline


setup:
	"$(PYTHON)" -m pip install -r requirements.txt

manifest:
	"$(PYTHON)" tools/prepare_cohort_a_subset.py --root "$(DATA)" --out-dir "$(OUT_DIR)" --max-patients "$(MAX_PATIENTS)"



#run the existing extraction CLI for each patient in pair manifest
lesions:
	"$(PYTHON)" -c 'import csv, pathlib, subprocess, sys; \
		rows = list(csv.DictReader(pathlib.Path(sys.argv[1]).open(newline="", encoding="utf-8"))); \
		rows or sys.exit("No patients found in pair manifest"); \
		[subprocess.run([sys.executable, "tools/extract_cohort_a_lesions.py", "--pairs", sys.argv[1], "--data-root", sys.argv[2], "--patient-id", row["patient_id"], "--out", str(pathlib.Path(sys.argv[3]) / (row["patient_id"] + ".csv"))], check=True) for row in rows]' "$(PAIRS)" "$(DATA)" "$(LESIONS_DIR)"




#recursive make preserves variable overrides and order even with make -j
pipeline: manifest
	$(MAKE) lesions

test:
	"$(PYTHON)" -m pytest -q
