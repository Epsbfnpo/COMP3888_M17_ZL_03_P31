PYTHON ?= python
DATA ?= ./data
MAX_PATIENTS ?= 999999
PATIENT_IDS ?=
OUT_DIR ?= outputs/cohort_a_subset
PATIENT_OUTPUT_ROOT ?= outputs/patients
DASHBOARD ?= tools/align_longitudinal_patient_v5_mapped.py
PORT ?= 8501

TEST_DIR ?= tests
COVERAGE_DIR ?= htmlcov

TEST_FILES := $(sort $(wildcard \
	$(TEST_DIR)/test_p31_26*.py \
	$(TEST_DIR)/test_p31_27*.py \
	$(TEST_DIR)/test_p31_28*.py \
	$(TEST_DIR)/test_p31_29*.py \
	$(TEST_DIR)/test_p31_30*.py \
	$(TEST_DIR)/test_p31_31*.py \
	$(TEST_DIR)/test_p31_32*.py \
	$(TEST_DIR)/test_p31_33*.py \
	$(TEST_DIR)/test_p31_34*.py \
))

PAIRS := $(OUT_DIR)/cohort_a_subset_pairs.csv

export DATA_ROOT := $(DATA)
export P31_PAIR_MANIFEST := $(PAIRS)
export P31_PATIENT_OUTPUT_ROOT := $(PATIENT_OUTPUT_ROOT)

MANIFEST_ARGS := --max-patients "$(MAX_PATIENTS)"
ifneq ($(strip $(PATIENT_IDS)),)
MANIFEST_ARGS := --patient-ids "$(PATIENT_IDS)"
endif

.PHONY: setup manifest dashboard run test test-coverage coverage show-tests \
	test-p31-27 test-p31-29 test-p31-31 test-p31-32 test-p31-33 test-p31-34 test-assigned
.DEFAULT_GOAL := run

setup:
	"$(PYTHON)" -m pip install -r requirements.txt

manifest:
	"$(PYTHON)" tools/prepare_cohort_a_subset.py --root "$(DATA)" --out-dir "$(OUT_DIR)" $(MANIFEST_ARGS) --path-mode relative-to-root

dashboard:
	"$(PYTHON)" -m streamlit run "$(DASHBOARD)" --server.port "$(PORT)"

run: manifest
	"$(PYTHON)" -m streamlit run "$(DASHBOARD)" --server.port "$(PORT)"

show-tests:
	@printf '%s\n' $(TEST_FILES)

test:
	@if [ -z "$(strip $(TEST_FILES))" ]; then echo "ERROR: no P31-26 to P31-34 test files found under $(TEST_DIR)"; exit 1; fi
	"$(PYTHON)" -m pytest -q $(TEST_FILES)

test-coverage:
	@if [ -z "$(strip $(TEST_FILES))" ]; then echo "ERROR: no P31-26 to P31-34 test files found under $(TEST_DIR)"; exit 1; fi
	"$(PYTHON)" -m pytest -q $(TEST_FILES) --cov=src --cov-branch --cov-report=term-missing --cov-report=html:$(COVERAGE_DIR) --cov-report=xml:coverage.xml
	@echo "Coverage report: $(COVERAGE_DIR)/index.html"

coverage: test-coverage

test-p31-27:
	"$(PYTHON)" -m pytest -q $(TEST_DIR)/test_p31_27_lesion_features_functional.py

test-p31-29:
	"$(PYTHON)" -m pytest -q $(TEST_DIR)/test_p31_29_pair_cost_functional.py

test-p31-31:
	"$(PYTHON)" -m pytest -q $(TEST_DIR)/test_p31_31_visualization_summary_functional.py

test-assigned:
	"$(PYTHON)" -m pytest -q \
		$(TEST_DIR)/test_p31_27_lesion_features_functional.py \
		$(TEST_DIR)/test_p31_29_pair_cost_functional.py \
		$(TEST_DIR)/test_p31_31_visualization_summary_functional.py

test-p31-32:
	"$(PYTHON)" -m pytest -q -rA $(TEST_DIR)/test_p31_32_dashboard_responsiveness.py

test-p31-33:
	"$(PYTHON)" -m pytest -q -rA $(TEST_DIR)/test_p31_33_pipeline_stability.py

test-p31-34:
	"$(PYTHON)" -m pytest -q $(TEST_DIR)/test_p31_34_robustness.py
