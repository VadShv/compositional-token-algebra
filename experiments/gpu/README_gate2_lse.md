# Gate 2 · Level 1 — LSE-pooling in KV space

Closes the "black hole" found in the original Gate 2: collapsing a repeated span
by **averaging its embeddings** (`e_bar = Σ πᵢ xᵢ`) degraded continuation
perplexity on Qwen2.5-3B from **1.67 → 3.20**, because a single mean vector,
re-projected once, cannot carry the span's *attention mass* over a non-linear
softmax.

## The fix (math)

Attention over a span for query `q`:

```
o_true(q) = Σ_i softmax_i(q·k_i) · v_i
```

Level 1 collapses **after** the K/V projection, on the cached keys/values, with
log-sum-exp so the collapsed token carries the span's total attention mass:

```
q·k_bar  =!  log Σ_i e^{q·k_i}
         ≈  log k + q·μ_k + ½ q^T Σ_k q          (Taylor around centroid μ_k)

k_bar = μ_k + (log k / ||q*||²) q*  +  ½ Σ_k q*        (mass term + curvature)
v_bar = Σ_i ω_i v_i,   ω_i = softmax(||k_i|| / √d)     (mass-weighted values)
```

- **`log k` mass term** — the piece CTA-mean lacks; the main expected win.
- **quadratic term** — curvature from the key covariance `Σ_k` (ablated separately).
- **`Σ_k` spectrum** — reported per span. Fast-decaying ⇒ one vector suffices
  (Level 1 enough). Flat ⇒ escalate to **Level 2 (rank-r)**, where attention
  error is bounded by `σ_{r+1}`.

## Run

Open in Colab (T4), run top to bottom (~3-4 min), send back `results_gate2_lse.json`:

https://colab.research.google.com/github/VadShv/compositional-token-algebra/blob/main/experiments/gpu/cta_gate2_lse.ipynb

Four decode paths on the same RAG prompt, identical KV length `m` for all CTA
variants (so this is purely a quality contest):

| path | collapse | expected |
|---|---|---|
| baseline | none, KV=n | reference PPL |
| cta_mean | mean of embeddings, KV=m | PPL ≈ 3.20 (the regression) |
| cta_lse_mass_quad | LSE keys + mass + curvature, KV=m | target: PPL back toward baseline |
| cta_lse_mass_only | LSE keys + mass, no curvature, KV=m | ablation |

## Files

- `../../src/cta/kv_lse.py` — `collapse_cache_lse`, `build_dynamic_cache`, `sigma_spectrum`.
- `gate2_lse.py` — standalone script version.
- `smoke_lse.py` — local CPU smoke test on 0.5B (shape/cache/position sanity).

## Verdict rubric

- **GO (Level 1)** — `cta_lse_*` PPL meaningfully closer to baseline than `cta_mean`.
- **PARTIAL** — helps but gap remains + `σ_k` shows 2-3 large eigenvalues → do Level 2.
- **NO-GO** — LSE ≈ mean → loss isn't in K/V aggregation → revisit q*/RoPE or Level 3.
