.PHONY: help test lint fmt typecheck examples build clean

help:
	@echo "test       run the test suite (no extra deps)"
	@echo "lint       ruff"
	@echo "typecheck  mypy on src/"
	@echo "examples   run all four offline examples"
	@echo "build      build sdist + wheel into dist/"
	@echo "clean      remove caches and build artefacts"

test:
	python -m unittest discover -s tests -v

lint:
	ruff check src tests examples

fmt:
	ruff format src tests examples

typecheck:
	mypy src

examples:
	@for f in examples/*.py; do echo "== $$f"; python "$$f"; done

build:
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache runs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
