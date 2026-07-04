"""
Wall-clock benchmark for CTA vs baseline across a ladder of model sizes.

Motivation: the paper reports an ATTENTION-FLOPS PROXY (n_layer * 2 * L^2 * d).
A proxy predicts a saving; it does not prove one. On real hardware the story is
more nuanced: (a) attention is not the only cost — MLP + lm_head + embedding are
linear in L and do not shrink quadratically; (b) CTA adds its own overhead
(repeated-span detection + compose). This script measures the REAL end-to-end
wall-clock time and peak memory, and decomposes the CTA cost into
detect+compose overhead vs. the transformer forward, on the SAME inputs, for
gpt2 (124M) -> gpt2-medium (355M) -> gpt2-large (774M).

Everything runs on CPU (frozen models, no grad). Timing uses warm-up passes and
the median of several repeats. Peak process RSS is captured per model.

Outputs: results/raw/results_walltime.json
"""
import warnings, os, time, json, gc, statistics, resource
warnings.filterwarnings("ignore")
import _bootstrap  # noqa: F401  (sets sys.path + result dirs)
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from cta.detector import select_collapse_spans, build_segments
from cta.model import baseline_forward, cta_forward, attention_flops, input_embeddings
from cta.algebra import compose

torch.manual_seed(0)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

OUT = os.path.join(_bootstrap.RESULTS_RAW, "results_walltime.json")

MODELS = ["gpt2", "gpt2-medium", "gpt2-large"]
N_PARAMS = {"gpt2": 124, "gpt2-medium": 355, "gpt2-large": 774}  # millions (approx)

# --- Build a long, repetition-heavy input from the real code corpus -----------
# Longer sequences make the quadratic attention saving observable against the
# linear (non-shrinking) MLP/head costs.
def load_corpus_text():
    files = sorted(
        os.path.join(_bootstrap.CORPUS, f)
        for f in os.listdir(_bootstrap.CORPUS)
        if f.startswith("code_") and f.endswith(".py")
    )
    # Concatenate a couple of files to get a realistically long, repeat-rich blob.
    text = "\n\n".join(open(f).read() for f in files[:2])
    return text


def peak_rss_mb():
    # ru_maxrss is in KB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def median_time(fn, repeats, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts)


@torch.no_grad()
def ppl_from_logits(logits, target_ids):
    """Mean next-token NLL -> perplexity, over positions that predict target_ids.
    logits: [L, V] aligned so logits[i] predicts token i+1."""
    L = min(logits.shape[0] - 1, target_ids.shape[0] - 1)
    lp = torch.log_softmax(logits[:L], dim=-1)
    nll = -lp[torch.arange(L), target_ids[1:L + 1]].mean().item()
    return float(torch.exp(torch.tensor(nll))), nll


def bench_one_model(model_name, ids, segments, repeats):
    print(f"\n=== {model_name} ({N_PARAMS[model_name]}M) ===", flush=True)
    gc.collect()
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.n_embd
    n_layer = model.config.n_layer

    n = ids.shape[0]

    # --- time baseline forward (full length n) ---
    def run_baseline():
        baseline_forward(model, ids)
    base_med, base_min = median_time(run_baseline, repeats)

    # --- time CTA overhead (detect already done; compose per span) ---
    def run_compose_only():
        emb, _ = input_embeddings(model, ids)
        for (s, e, is_col) in segments:
            if is_col:
                compose(emb[s:e], mode="norm")
    comp_med, _ = median_time(run_compose_only, repeats)

    # --- time full CTA forward (compose + collapsed transformer) ---
    def run_cta():
        cta_forward(model, ids, segments, score_mode="norm")
    cta_med, cta_min = median_time(run_cta, repeats)

    # --- collapsed length m ---
    logits_b, nb = baseline_forward(model, ids)
    logits_c, m, _ = cta_forward(model, ids, segments, score_mode="norm")

    # quality at matched next-token positions (whole-sequence proxy PPL)
    base_ppl, base_nll = ppl_from_logits(logits_b, ids)

    flops_ratio = (m ** 2) / (n ** 2)
    proxy_base = attention_flops(n, n_layer, d)
    proxy_cta = attention_flops(m, n_layer, d)

    peak = peak_rss_mb()

    row = {
        "model": model_name, "params_M": N_PARAMS[model_name],
        "n_layer": n_layer, "d": d,
        "n_tokens": n, "m_tokens": m,
        "len_ratio": m / n,
        "baseline_ms": base_med * 1000, "baseline_ms_min": base_min * 1000,
        "cta_ms": cta_med * 1000, "cta_ms_min": cta_min * 1000,
        "compose_overhead_ms": comp_med * 1000,
        "speedup": base_med / cta_med,
        "attn_flops_ratio": flops_ratio,
        "proxy_speedup_attn_only": proxy_base / proxy_cta,
        "base_ppl": base_ppl,
        "peak_rss_mb": peak,
    }
    print(f"  n={n} -> m={m} (len ratio {m/n:.3f}, attn-flops ratio {flops_ratio:.3f})")
    print(f"  baseline: {row['baseline_ms']:.1f} ms   CTA: {row['cta_ms']:.1f} ms   "
          f"WALL-CLOCK speedup: {row['speedup']:.2f}x")
    print(f"  compose overhead: {row['compose_overhead_ms']:.1f} ms "
          f"({100*comp_med/cta_med:.1f}% of CTA time)")
    print(f"  attention-only proxy speedup: {row['proxy_speedup_attn_only']:.2f}x "
          f"(what the paper's FLOPs proxy predicts)")
    print(f"  peak RSS: {peak:.0f} MB")

    del model
    gc.collect()
    return row


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")  # same BPE for all GPT-2 sizes
    text = load_corpus_text()
    ids_full = torch.tensor(tok(text)["input_ids"])

    # Cap length so gpt2-large stays tractable on 2 vCPU, but long enough that the
    # quadratic term matters. 512 tokens is a realistic prompt length.
    MAXLEN = int(os.environ.get("CTA_MAXLEN", "512"))
    ids = ids_full[:MAXLEN]

    spans = select_collapse_spans(ids.tolist(), k_min=3, k_max=16, f_min=2)
    segments = build_segments(ids.tolist(), spans)
    n_col = sum(1 for _, _, c in segments if c)
    print(f"input: {ids.shape[0]} tokens -> {len(segments)} segments "
          f"({n_col} collapsed spans)")

    repeats = int(os.environ.get("CTA_REPEATS", "5"))
    rows = []
    for mname in MODELS:
        try:
            rows.append(bench_one_model(mname, ids, segments, repeats))
        except Exception as ex:
            print(f"  !! {mname} failed: {ex}", flush=True)

    json.dump(rows, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")

    # summary table
    print("\n" + "=" * 78)
    print(f"{'model':<14}{'params':>7}{'n->m':>12}{'base ms':>10}{'cta ms':>9}"
          f"{'speedup':>9}{'proxy':>8}{'RAM MB':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r['model']:<14}{r['params_M']:>6}M{r['n_tokens']:>6}->{r['m_tokens']:<4}"
              f"{r['baseline_ms']:>10.1f}{r['cta_ms']:>9.1f}{r['speedup']:>8.2f}x"
              f"{r['proxy_speedup_attn_only']:>7.2f}x{r['peak_rss_mb']:>9.0f}")


if __name__ == "__main__":
    main()
