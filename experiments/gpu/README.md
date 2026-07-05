# GPU validation (Test 1.1)

Does the CTA speedup survive on a real GPU with **FlashAttention**? This is the
main infrastructure risk: on GPU, attention is already IO-cheap, so the fear is
that the win collapses.

## The short answer (roofline)

CTA shortens the sequence `n → m`, which cuts *everything length-linear* (QKVO
projections, MLP, LM head). FlashAttention only makes the O(n²) attention **math**
IO-cheap — it does **not** reduce FLOPs, and it never touched the linear part.

Run `python roofline_cta.py` for the analytic breakdown. Result: on Qwen2.5-3B/7B
at 0.5k–8k, **70–98% of prefill compute is length-linear**, so shrinking the token
count wins in a plane *orthogonal* to FlashAttention.

| Qwen2.5-3B | 512 | 2048 | 8192 |
|---|---|---|---|
| length-linear share | 97.6% | 91.1% | 71.9% |
| attention (quadratic) | 2.4% | 8.9% | 28.1% |
| speedup ceiling (FLOPs) | 1.22× | 1.64× | 1.96× |

These are optimistic ceilings (ignore detection/compose overhead + GPU
underutilization). The measurement below reports the *realized* number.

## Run the measurement (Google Colab)

1. Open **`cta_gpu_colab.ipynb`** in Colab.
2. `Runtime → Change runtime type → GPU` (free T4 is fine).
3. `Runtime → Run all`.
   * Free T4 (16 GB) → auto-selects **Qwen2.5-3B**.
   * ≥24 GB GPU → auto-selects **Qwen2.5-7B**.
   * Attention runs on the fused FlashAttention kernel (PyTorch **SDPA** backend by
     default; set `USE_FLASH_ATTN=True` to build `flash-attn` instead).
4. It saves **`results_gpu.json`** — download it and send it back for analysis.

## Protocol (Variant A — prefill-only)

Single forward over the prefix, `baseline (n)` vs `CTA (m)`, GPU + FA backend,
sweeping length 512→8192. Proper GPU timing (warmup + `cuda.synchronize()` +
median). We report **two** speedups so the number is honest:

* `speedup_fwd` — forward-only (the compute ceiling).
* `speedup_total` — including detection + compose overhead (real end-to-end).

Prefill-only deliberately avoids the KV-cache integration problem (collapsed tokens
in a paged KV-cache during decode is a separate, hard engineering task). Variant B
(end-to-end generation) is the follow-up once Variant A shows go.

## Go / no-go

* **GO** — `speedup_fwd` rises with length and `speedup_total` crosses 1.0 and grows
  (≈1.0× at 512 → >1.3× at 4k–8k). The win survives FA.
* **NO-GO / nuance** — `speedup_fwd` stays ≤1.0 even at 4k–8k → GPU underutilized;
  re-test at larger batch / longer context before concluding.

Short-length overhead (CPU-side span detection + the Python collapse loop) is a
known, fixable engineering cost — vectorize the collapse, move detection off the
hot path — not a refutation of the mechanism.
