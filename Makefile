# NkwaPay credit scoring: one command reruns the whole analysis.
#
#   make            data check, notebooks 01-13, tests, claim check, report, slides (~15 min)
#   make check      confirm the raw data files are in data/raw/ with the expected shapes
#   make test       run the tests
#   make verify     recompute the quoted numbers without rerunning the notebooks
#   make report     rebuild the report (Word, and PDF if xelatex is installed)
#   make slides     rebuild the debrief deck
#   make score      score the example applications with the shipped model (needs no data)
#   make clean      remove caches

PY := python

.DEFAULT_GOAL := all
.PHONY: all check test verify report slides score clean setup

all:
	bash scripts/run_all.sh

setup:
	$(PY) -m pip install -r requirements.txt

check:
	$(PY) -m pytest -q tests/test_data_present.py

test:
	$(PY) -m pytest -q tests

verify:
	$(PY) scripts/verify_claims.py

report:
	bash scripts/build_report.sh

slides:
	$(PY) slides/build_slides.py

score:
	@for f in examples/*.json; do echo "== $$f"; $(PY) -W ignore scripts/score.py $$f; done

clean:
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name .ipynb_checkpoints -type d -prune -exec rm -rf {} +
