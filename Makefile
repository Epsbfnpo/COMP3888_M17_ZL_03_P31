PYTHON ?= python
DATA ?= ./data
MAX_PATIENTS ?= 99999
PATIENT_IDS ?=
OUT_DIR ?= outputs/cohort_a_subset
PATIENT_OUTPUT_ROOT ?= outputs/patients
DASHBOARD ?= tools/align_longitudinal_patient_v5_mapped.py
PORT ?= 8501

TEST_DIR ?= tests
COVERAGE_DIR ?= htmlcov

# P31 tests are functional tests; other test_*.py files are unit tests.
FUNCTIONAL_TEST_FILES := $(sort $(wildcard $(TEST_DIR)/test_p31_*.py))
UNIT_TEST_FILES := $(filter-out $(FUNCTIONAL_TEST_FILES),$(sort $(wildcard $(TEST_DIR)/test_*.py)))
ALL_TEST_FILES := $(sort $(FUNCTIONAL_TEST_FILES) $(UNIT_TEST_FILES))

PAIRS := $(OUT_DIR)/cohort_a_subset_pairs.csv

export DATA_ROOT := $(DATA)
export P31_PAIR_MANIFEST := $(PAIRS)
export P31_PATIENT_OUTPUT_ROOT := $(PATIENT_OUTPUT_ROOT)

MANIFEST_ARGS := --max-patients "$(MAX_PATIENTS)"
ifneq ($(strip $(PATIENT_IDS)),)
MANIFEST_ARGS := --patient-ids "$(PATIENT_IDS)"
endif

.PHONY: setup manifest dashboard run test test-functional test-unit test-coverage test-functional-coverage test-unit-coverage
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
	@if [ -z "$(strip $(ALL_TEST_FILES))" ]; then echo "ERROR: no test files found under $(TEST_DIR)"; exit 1; fi
	"$(PYTHON)" -m pytest -q $(ALL_TEST_FILES)

test-functional:
	@if [ -z "$(strip $(FUNCTIONAL_TEST_FILES))" ]; then echo "ERROR: no P31 functional tests found under $(TEST_DIR)"; exit 1; fi
	"$(PYTHON)" -m pytest -q $(FUNCTIONAL_TEST_FILES)

test-unit:
	@if [ -z "$(strip $(UNIT_TEST_FILES))" ]; then echo "ERROR: no unit tests found under $(TEST_DIR)"; exit 1; fi
	"$(PYTHON)" -m pytest -q $(UNIT_TEST_FILES)

# Statement coverage and branch coverage are both included by --cov-branch.
# Keep separate reports and raw coverage data for each test group.
test-functional-coverage:
	@if [ -z "$(strip $(FUNCTIONAL_TEST_FILES))" ]; then echo "ERROR: no P31 functional tests found under $(TEST_DIR)"; exit 1; fi
	COVERAGE_FILE=.coverage.functional "$(PYTHON)" -m pytest -q $(FUNCTIONAL_TEST_FILES) --cov=src --cov-branch --cov-report=term-missing --cov-report=html:$(COVERAGE_DIR)/functional --cov-report=xml:coverage-functional.xml --cov-report=json:coverage-functional.json
	@echo "Functional coverage: $(COVERAGE_DIR)/functional/index.html | coverage-functional.xml | coverage-functional.json"

test-unit-coverage:
	@if [ -z "$(strip $(UNIT_TEST_FILES))" ]; then echo "ERROR: no unit tests found under $(TEST_DIR)"; exit 1; fi
	COVERAGE_FILE=.coverage.unit "$(PYTHON)" -m pytest -q $(UNIT_TEST_FILES) --cov=src --cov-branch --cov-report=term-missing --cov-report=html:$(COVERAGE_DIR)/unit --cov-report=xml:coverage-unit.xml --cov-report=json:coverage-unit.json
	@echo "Unit coverage: $(COVERAGE_DIR)/unit/index.html | coverage-unit.xml | coverage-unit.json"

test-coverage:
	@if [ -z "$(strip $(ALL_TEST_FILES))" ]; then echo "ERROR: no test files found under $(TEST_DIR)"; exit 1; fi
	COVERAGE_FILE=.coverage.all "$(PYTHON)" -m pytest -q $(ALL_TEST_FILES) --cov=src --cov-branch --cov-report=term-missing --cov-report=html:$(COVERAGE_DIR) --cov-report=xml:coverage.xml --cov-report=json:coverage.json
	@echo "All-tests coverage: $(COVERAGE_DIR)/index.html | coverage.xml | coverage.json"
