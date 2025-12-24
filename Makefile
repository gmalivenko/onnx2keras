PYTHONPATH := .
POETRY_MODULE := PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python poetry run python -m
PYTEST := $(POETRY_MODULE) pytest

.PHONY: run_tests
run_tests:
	$(PYTEST) test -v


.PHONY: poetry_update
poetry_update:
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python poetry update

.PHONY: test_models
test_models:
	$(PYTEST) test/models -v

.PHONY: watch
watch:
	$(POETRY_MODULE) pytest_watch --runner "python -m pytest -v -k $(K)"

.PHONY: lint
lint:
	$(POETRY_MODULE) mypy --install-types --non-interactive .

.PHONY: lint_strict_code
lint_strict_code:
	$(POETRY_MODULE) mypy --install-types --non-interactive --strict code_loader

.PHONY: lint_tests
lint_tests:
	$(POETRY_MODULE) mypy --install-types --non-interactive tests

.PHONY: test_with_coverage
test_with_coverage:
	$(PYTEST) --cov=code_loader --cov-branch --no-cov-on-fail --cov-report term-missing --cov-report html -v tests/


