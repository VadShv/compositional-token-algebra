# ---------------------------------------------------------------------------
# CTA reproduction Makefile
#
# Fast targets (seconds, no GPT-2 forward passes): plots, gate
# Slow targets (minutes on CPU, run frozen GPT-2): corpus, cache, main
#
#   make setup     install Python dependencies
#   make corpus    download code corpus + generate ticket corpus
#   make gate      train the learnable gate from shipped caches -> results_gate.json
#   make plots     regenerate both paper figures from shipped/updated results
#   make main      (SLOW) full CTA-vs-baseline sweep -> results_main.json
#   make frontier  (SLOW) heuristic gated frontier -> results_frontier.json
#   make cache     (SLOW) rebuild per-span gate-training caches (needs corpus)
#   make killer    CTA vs prefix-caching comparison (fast)
#   make diagnose  position vs content-opacity ablation (moderate)
#   make walltime  (SLOW) real CPU wall-clock: size ladder + length sweep
#   make plots-walltime  regenerate the wall-clock figure from shipped JSON
#   make paper     compile the LaTeX preprint (requires a LaTeX toolchain)
#   make all-fast  gate + plots (full fast reproduction)
#   make clean     remove Python caches
# ---------------------------------------------------------------------------

PY ?= python
export OMP_NUM_THREADS ?= 2
EXP := experiments

.PHONY: setup corpus gate plots main frontier cache killer diagnose walltime plots-walltime paper all-fast clean

setup:
	$(PY) -m pip install -r requirements.txt

corpus:
	bash data/download_corpus.sh

gate:
	cd $(EXP) && $(PY) train_gate.py

plots:
	cd $(EXP) && $(PY) plot.py && $(PY) plot_gate.py

main:
	cd $(EXP) && $(PY) run_main.py

frontier:
	cd $(EXP) && $(PY) gate.py

cache:
	cd $(EXP) && $(PY) build_cache.py

killer:
	cd $(EXP) && $(PY) killer.py

diagnose:
	cd $(EXP) && $(PY) diagnose.py

walltime:
	cd $(EXP) && $(PY) benchmark_walltime.py

plots-walltime:
	cd $(EXP) && $(PY) plot_walltime.py

paper:
	cd paper && tectonic cta.tex || pdflatex cta.tex

all-fast: gate plots

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
