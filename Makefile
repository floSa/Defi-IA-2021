# Orchestration entry points. Uses a local virtualenv in .venv.
# On Windows, run these from WSL or Git Bash.

PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
PY := $(BIN)/python
PIP := $(BIN)/pip

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(VENV): requirements.txt  ## Create venv and install core deps
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: install
install: $(VENV)  ## Install core (CPU) environment

.PHONY: install-dl
install-dl: install  ## Add deep-learning extras (run on a GPU machine)
	$(PIP) install -r requirements-dl.txt

.PHONY: data
data:  ## Extract the raw competition zip into data/raw/
	@mkdir -p data/raw
	@unzip -o -q defi-ia-insa-toulouse.zip -d data/raw && echo "Extracted to data/raw/"

.PHONY: test
test:  ## Run the test suite
	$(PY) -m pytest

.PHONY: lint
lint:  ## Lint with ruff
	$(BIN)/ruff check src tests

.PHONY: eda
eda:  ## Print the dataset summary report
	$(PY) scripts/explore_data.py

.PHONY: clean
clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info
