PYTHON ?= python
DATA ?= ./data
MAX_PATIENTS ?= 999999
PATIENT_IDS ?=
OUT_DIR ?= outputs/cohort_a_subset
PATIENT_OUTPUT_ROOT ?= outputs/patients
DASHBOARD ?= tools/align_longitudinal_patient_v5_mapped.py
PORT ?= 8501

PAIRS := $(OUT_DIR)/cohort_a_subset_pairs.csv

export DATA_ROOT := $(DATA)
export P31_PAIR_MANIFEST := $(PAIRS)
export P31_PATIENT_OUTPUT_ROOT := $(PATIENT_OUTPUT_ROOT)

MANIFEST_ARGS := --max-patients "$(MAX_PATIENTS)"
ifneq ($(strip $(PATIENT_IDS)),)
MANIFEST_ARGS := --patient-ids "$(PATIENT_IDS)"
endif

.PHONY: setup manifest dashboard run test
.DEFAULT_GOAL := run

setup:
	"$(PYTHON)" -m pip install -r requirements.txt

manifest:
	"$(PYTHON)" tools/prepare_cohort_a_subset.py --root "$(DATA)" --out-dir "$(OUT_DIR)" $(MANIFEST_ARGS) --path-mode relative-to-root

dashboard:
	"$(PYTHON)" -m streamlit run "$(DASHBOARD)" --server.port "$(PORT)"

run: manifest
	"$(PYTHON)" -m streamlit run "$(DASHBOARD)" --server.port "$(PORT)"

test:
	"$(PYTHON)" -m pytest -q
