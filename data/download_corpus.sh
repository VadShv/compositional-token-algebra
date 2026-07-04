#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_corpus.sh
#
# Reproduces the exact code corpus used in the CTA experiments by fetching six
# real-world Python source files from their upstream GitHub repositories, then
# generates the synthetic Jira-style ticket corpus.
#
# The code files are fetched from the `main` branch of each project. Upstream
# files may drift over time; if you need the byte-exact snapshot used in the
# paper, use the pinned copies shipped under data/corpus/ in the release
# tarball instead of re-downloading.
#
# Usage:
#   bash data/download_corpus.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS_DIR="${SCRIPT_DIR}/corpus"
mkdir -p "${CORPUS_DIR}"

# filename  ->  raw GitHub URL   (order matters: code_0 .. code_5)
declare -a URLS=(
  "code_0.py|https://raw.githubusercontent.com/psf/requests/main/src/requests/models.py"
  "code_1.py|https://raw.githubusercontent.com/psf/requests/main/src/requests/sessions.py"
  "code_2.py|https://raw.githubusercontent.com/pallets/flask/main/src/flask/app.py"
  "code_3.py|https://raw.githubusercontent.com/pallets/click/main/src/click/core.py"
  "code_4.py|https://raw.githubusercontent.com/python/cpython/main/Lib/json/decoder.py"
  "code_5.py|https://raw.githubusercontent.com/python/cpython/main/Lib/argparse.py"
)

echo ">> Downloading code corpus into ${CORPUS_DIR}"
for entry in "${URLS[@]}"; do
  fname="${entry%%|*}"
  url="${entry##*|}"
  echo "   - ${fname}  <-  ${url}"
  curl -fsSL "${url}" -o "${CORPUS_DIR}/${fname}"
done

echo ">> Generating synthetic Jira ticket corpus (tickets.txt)"
python "${SCRIPT_DIR}/make_tickets.py"

echo ">> Done. Corpus files:"
ls -la "${CORPUS_DIR}"
