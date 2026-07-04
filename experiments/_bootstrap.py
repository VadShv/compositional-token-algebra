"""Path bootstrap so experiment scripts can import the `cta` package, the data
generators, and locate the repo results/ directory regardless of CWD."""
import os, sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
DATA = os.path.join(REPO_ROOT, "data")
RESULTS = os.path.join(REPO_ROOT, "results")
RESULTS_RAW = os.path.join(RESULTS, "raw")
RESULTS_FIG = os.path.join(RESULTS, "figures")
CORPUS = os.path.join(DATA, "corpus")

for p in (SRC, DATA, os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.makedirs(RESULTS_RAW, exist_ok=True)
os.makedirs(RESULTS_FIG, exist_ok=True)
os.makedirs(CORPUS, exist_ok=True)
