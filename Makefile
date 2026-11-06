VENV := .venv
PY := $(VENV)/bin/python

$(VENV)/.ok: requirements.txt
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	touch $(VENV)/.ok

setup: $(VENV)/.ok

validate-mock: setup
	$(PY) contracts/validate_handoff.py handoff/mock

test: setup
	$(PY) -m pytest modeling/tests -q

check: validate-mock test

.PHONY: setup validate-mock test check
