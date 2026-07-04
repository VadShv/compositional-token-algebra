"""
Wall-clock validation of CTA on a MODERN backbone: Qwen2.5-0.5B.

Why: the paper's experiments use GPT-2 (learned positions, dense attention,
1024 window). Qwen2.5 is a 2024 architecture (RoPE, grouped-query attention,
SwiGLU, 32k window). This script checks that the parameter-free compose/decompose
and the wall-clock win TRANSFER to that architecture, and that reversibility
holds at machine epsilon.

CTA operates purely on input embeddings, so the forward path is:
  emb = model.get_input_embeddings()(ids)
  h   = model.model(inputs_embeds=emb).last_hidden_state
  lg  = model.lm_head(h)
which is architecture-agnostic (any HF causal LM supporting inputs_embeds).

Everything on CPU, frozen, no grad. Warm-up + median of repeats.
Qwen's 32k window lets us measure sequences longer than the GPT-2 limit (1024),
where the quadratic attention term dominates more strongly.

Output: results/raw/results_qwen_walltime.json, results/raw/results_qwen_reversibility.json

Run (from repo root or experiments/):
  OMP_NUM_THREADS=2 python3 experiments/benchmark_qwen.py
Deps: torch, transformers>=4.40 (Qwen2), plus the local `cta` package.
Qwen2.5-0.5B (~1 GB) is pulled from HuggingFace on first run.
"""
import warnings, os, sys, time, json, statistics, resource
warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401  (sets sys.path: src/, data/; result dirs)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cta.detector import select_collapse_spans, build_segments
from cta.algebra import compose, decompose

torch.manual_seed(0)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print(f"loading {MODEL_NAME} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval()
cfg = model.config
print(f"  layers={cfg.num_hidden_layers} d={cfg.hidden_size} "
      f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
      f"ctx={cfg.max_position_embeddings}", flush=True)


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def median_time(fn, repeats=5, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def load_corpus_text():
    files = sorted(
        os.path.join(_bootstrap.CORPUS, f)
        for f in os.listdir(_bootstrap.CORPUS)
        if f.startswith("code_") and f.endswith(".py")
    )
    return "\n\n".join(open(f).read() for f in files[:3])


@torch.no_grad()
def emb_of(ids):
    return model.get_input_embeddings()(ids.unsqueeze(0)).squeeze(0)  # [n,d]


@torch.no_grad()
def baseline_forward(ids):
    emb = emb_of(ids)
    h = model.model(inputs_embeds=emb.unsqueeze(0)).last_hidden_state
    _ = model.lm_head(h)
    return ids.shape[0]


@torch.no_grad()
def cta_forward(ids, segments, mode="norm"):
    emb = emb_of(ids)
    vecs = []
    for (s, e, is_col) in segments:
        if is_col:
            e_bar, R, pi = compose(emb[s:e], mode=mode)
            vecs.append(e_bar)
        else:
            vecs.append(emb[s])
    collapsed = torch.stack(vecs, 0)  # [m,d]
    h = model.model(inputs_embeds=collapsed.unsqueeze(0)).last_hidden_state
    _ = model.lm_head(h)
    return collapsed.shape[0]


# ---------------------------------------------------------------------------
# 1) Reversibility on real Qwen embeddings (all four scoring functions)
# ---------------------------------------------------------------------------
print("\n[1] reversibility on Qwen embeddings ...", flush=True)
text = load_corpus_text()
all_ids = tok(text, return_tensors="pt").input_ids.squeeze(0)
emb_full = emb_of(all_ids[:2048])
rev = {}
g = torch.Generator().manual_seed(0)
for mode in ["uniform", "norm", "posdecay", "selfconsist"]:
    max_err = 0.0
    for _ in range(200):
        k = int(torch.randint(2, 16, (1,), generator=g).item())
        start = int(torch.randint(0, emb_full.shape[0] - k, (1,), generator=g).item())
        span = emb_full[start:start + k]
        e_bar, R, pi = compose(span, mode=mode)
        recon = decompose(e_bar, R)
        max_err = max(max_err, (recon - span).abs().max().item())
    rev[mode] = max_err
    print(f"    {mode:12s} max L-inf err = {max_err:.2e}", flush=True)
json.dump(rev, open(os.path.join(_bootstrap.RESULTS_RAW, "results_qwen_reversibility.json"), "w"), indent=2)


# ---------------------------------------------------------------------------
# 2) Wall-clock speedup vs sequence length
# ---------------------------------------------------------------------------
print("\n[2] wall-clock vs length ...", flush=True)
rows = []
for L in [256, 512, 1024, 2048]:
    ids = all_ids[:L]
    spans = select_collapse_spans(ids.tolist(), k_min=3, k_max=16, f_min=2)
    segs = build_segments(ids.tolist(), spans)
    m = len(segs)
    n_col = sum(1 for _, _, c in segs if c)
    if m >= L:
        print(f"    L={L}: no collapse candidates, skip", flush=True)
        continue
    t_base = median_time(lambda: baseline_forward(ids))
    t_cta = median_time(lambda: cta_forward(ids, segs))
    sp = t_base / t_cta
    row = {"len": L, "m": m, "collapsed_spans": n_col,
           "baseline_ms": round(t_base * 1000, 1),
           "cta_ms": round(t_cta * 1000, 1),
           "speedup": round(sp, 3),
           "peak_rss_mb": round(peak_rss_mb(), 0)}
    rows.append(row)
    print(f"    L={L:5d} -> m={m:4d} ({n_col} spans)  "
          f"base={row['baseline_ms']:.0f}ms  cta={row['cta_ms']:.0f}ms  "
          f"speedup={sp:.2f}x", flush=True)

out = {"model": MODEL_NAME, "params_M": 494,
       "n_layer": cfg.num_hidden_layers, "d": cfg.hidden_size,
       "arch": "RoPE + GQA + SwiGLU (2024)", "rows": rows}
json.dump(out, open(os.path.join(_bootstrap.RESULTS_RAW, "results_qwen_walltime.json"), "w"), indent=2)
print("\nsaved results_qwen_walltime.json + results_qwen_reversibility.json", flush=True)
