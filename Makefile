#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TAL-S2-OutilsCorpus
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python$(PYTHON_VERSION)
PROJECT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt --ignore-requires-python



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	aws s3 sync s3://tal-m1-corpus/data/ \
		data/


## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	aws s3 sync data/ \
		s3://tal-m1-corpus/data


UNAME_S := $(shell uname -s)
## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv --system-site-packages --prompt $(VENV_NAME) .venv
	@echo ">>> New virtualenv created. Activate with:"
	@echo "source .venv/bin/activate"
ifeq ($(UNAME_S),Darwin)
	@sed -e 's/($(VENV_NAME)) /$(VENV_NAME)/g' -i '' .venv/bin/activate
else
	@sed -e 's/($(VENV_NAME)) /$(VENV_NAME)/g' -i .venv/bin/activate
endif



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data:
	$(PYTHON_INTERPRETER) outils_corpus/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
