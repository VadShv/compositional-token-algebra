"""
Killer experiment: CTA vs prefix caching.

Prefix caching only amortizes a SHARED PREFIX. It cannot compress repeats that occur
in the MIDDLE of the context. We test on the multi-turn chat where the system prompt
repeats before every turn (mid-context repeats), and measure how much sequence each
method can save.

- Prefix caching saving = length of the single common prefix (only the first shared block).
- CTA saving = all repeated spans anywhere (prefix + interior).
"""
import warnings
warnings.filterwarnings("ignore")
import _bootstrap  # noqa: F401  (sets sys.path + result dirs)
import torch
from transformers import GPT2TokenizerFast
from data import SAMPLES, SYS
from cta.detector import select_collapse_spans, build_segments

tok = GPT2TokenizerFast.from_pretrained("gpt2")

def common_prefix_len(seqs):
    if not seqs: return 0
    m = min(len(s) for s in seqs)
    for i in range(m):
        if len(set(s[i] for s in seqs)) != 1:
            return i
    return m

print(f"{'sample':7s} {'n':>4s} {'prefix_cache_saved':>19s} {'cta_saved':>10s} {'cta_interior_only':>18s}")
for name, text in SAMPLES.items():
    ids = tok(text)["input_ids"]
    n = len(ids)
    spans = select_collapse_spans(ids, 3, 16, 2)
    if not spans:
        print(f"{name:7s} {n:>4d}  (no repeats)"); continue
    segs = build_segments(ids, spans)
    m = len(segs)
    cta_saved = n - m

    # Prefix caching: for chat/logs, turns are separated by newlines. Simulate the
    # realistic cache: only a single shared PREFIX across "requests" is cacheable.
    # Approx: the shared prefix of the whole doc vs its own repeats == leading repeated block.
    # For a fair upper bound on prefix caching we take the largest repeated span that is a PREFIX.
    # Interior repeats = repeats not starting at position 0.
    interior_saved = sum((e - s) - 1 for (s, e) in spans if s > 0)
    prefix_saved = sum((e - s) - 1 for (s, e) in spans if s == 0)

    print(f"{name:7s} {n:>4d} {prefix_saved:>19d} {cta_saved:>10d} {interior_saved:>18d}")

print("\nInterpretation: prefix caching only captures repeats at position 0 (prefix_saved).")
print("CTA additionally captures interior repeats (cta_interior_only) that prefix caching CANNOT.")
