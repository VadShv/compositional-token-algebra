# Gate 2 — decode / KV-cache validation

**Kill-test.** Everything else in this repo (prefill wall-clock, collapse-ratio)
measures a **single forward pass**. Products live on **decode** —
autoregressive generation over a KV-cache. Gate 2 asks the only remaining
product-killing question:

> After we collapse a prompt (`n → m`) and build a KV-cache over the **m**
> pooled `e_bar` positions, can we generate **coherent** continuations on top of
> that compressed cache, and is decode actually faster / lighter?

## What the notebook does

`cta_gate2_decode.ipynb` (run on a free Colab T4, ~2-3 min):

1. Builds a RAG-like prompt with real internal repetition.
2. **A — baseline**: standard greedy decode over the full prompt (KV-cache
   length `n`).
3. **B — CTA-decode**: collapse the prompt, prefill the pooled embeddings with
   `use_cache=True` (KV-cache length `m`), then greedily decode on top of it.
4. Reports:
   - `greedy_match_rate` — do B's tokens match A's?
   - `ppl_under_full_model` — perplexity of B's continuation judged by the FULL
     (uncollapsed) model given the FULL prompt (the honest judge);
   - `per_tok_ms` — decode wall-clock per token, baseline vs CTA;
   - `kv_positions` — cached positions (`n` vs `m`) = the memory story.

## Reading the result

- **GO** — CTA-decode text coherent, `ppl_under_full_model` within ~1.5× of
  baseline `ppl_self`, and `cta.per_tok_ms ≤ baseline`. Decode survives.
- **PARTIAL** — readable but degraded, or not faster → position CTA as a
  **prefill accelerator** only (Variant A).
- **NO-GO** — incoherent / PPL blows up → generation over the collapsed cache is
  broken; product limited to offline / prefill metrics.

## Prior from a local 0.5B smoke test (CPU)

A quick correctness run on `Qwen2.5-0.5B` (aggressive collapse, ratio ≈ 0.12)
confirms the mechanism runs end-to-end and the **KV-cache is physically
smaller** (205 → 38 positions). But it also shows the honest risk: the baseline
correctly continued the RAG content while CTA-decode **drifted off-topic** — the
pooled `e_bar` tokens did not carry enough detail for the model to attend back to
specifics during generation. The 3B Colab run measures whether this holds,
softens, or breaks at product scale. This is exactly what Gate 2 is here to
catch before any vLLM/plugin work.
