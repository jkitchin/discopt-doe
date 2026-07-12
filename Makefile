# Release / QC helper targets. See the release checklist in CLAUDE.md.
# Everything runs through uv so the pinned environment is used.

.PHONY: help test lint docs build check preflight clean

help:
	@echo "Targets:"
	@echo "  test       run the full test suite, including slow tests"
	@echo "  lint       ruff check + ruff format --check"
	@echo "  docs       build the Jupyter Book with warnings-as-errors"
	@echo "  build      wipe dist/ and build the sdist + wheel"
	@echo "  check      twine check the built artifacts"
	@echo "  preflight  test + lint + docs (run before a release)"
	@echo "  clean      remove build artifacts"

test:
	uv run pytest tests/ -m "slow or not slow"

lint:
	uv run ruff check .
	uv run ruff format --check .

docs:
	uv run jupyter-book build docs/ -W

build:
	rm -rf dist
	uv build

check:
	uvx twine check dist/*

preflight: test lint docs

clean:
	rm -rf dist docs/_build
