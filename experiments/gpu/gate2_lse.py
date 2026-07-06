"""
Gate 2 — Level 1 (LSE-pooling in KV space).

Adds a THIRD decode path to the Gate 2 kill-test:

  A. BASELINE   : full-prompt decode over KV-cache length n.
  B. CTA-MEAN   : collapse EMBEDDINGS (e_bar) -> prefill -> KV length m -> decode.
                  (the original Gate 2 CTA; PPL degraded 1.67 -> 3.20)
  C. CTA-LSE    : full prefill -> compress the KV-cache itself with log-sum-exp
                  pooling of keys/values inside spans -> KV length m -> decode.
                  Attacks the ROOT cause: attention non-linearity over a
                  collapsed span. Adds the missing `log k` attention-mass term.

Hypothesis: C keeps the m-length KV (memory + decode-speed win) while recovering
quality lost by B, because keys/values are pooled AFTER the projection with the
correct attention-mass term instead of averaging embeddings BEFORE it.

Metrics per path: prefill ms, decode ms/tok, KV positions, greedy-match vs
baseline, PPL of continuation under the FULL model (honest judge).
Also reports Sigma_k spectrum per span -> data-driven signal for escalating to
Level 2 (rank-r) when key dispersion is high.

Run on Colab T4. Model: Qwen2.5-3B, bf16.
"""
import sys, time, json, math
import torch

sys.path.insert(0, "/content/compositional-token-algebra/src")
try:
    from cta.detector import select_collapse_spans, build_segments
    from cta.algebra import compose
    from cta.kv_lse import collapse_cache_lse, build_dynamic_cache, sigma_spectrum
except Exception:
    sys.path.insert(0, "src")
    from cta.detector import select_collapse_spans, build_segments
    from cta.algebra import compose
    from cta.kv_lse import collapse_cache_lse, build_dynamic_cache, sigma_spectrum

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
N_GEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE)
model.eval()
cfg = model.config
print(f"model={MODEL_NAME} layers={cfg.num_hidden_layers} d={cfg.hidden_size} "
      f"device={DEVICE} dtype={DTYPE}", flush=True)


def make_rag_prompt():
    doc = ("Retrieved document. User {u} logged in from 10.0.{a}.{b} at "
           "2026-07-{d}. Session established. Role=admin, region=eu-west. "
           "Auth method oauth2. Token TTL 3600s. Rate limit 1000 req/min. "
           "Endpoint /api/v2/login returned 200 OK. ")
    import random
    random.seed(0)
    ctx = ""
    for i in range(10):
        ctx += doc.format(u=1000 + i, a=random.randint(0, 255),
                          b=random.randint(0, 255), d=random.randint(10, 28))
    q = ("\n\nBased on the retrieved documents above, summarize the common "
         "authentication configuration in one sentence:")
    return ctx + q


def emb_of(ids):
    return model.model.embed_tokens(ids)


def seg_of(ids):
    spans = select_collapse_spans(ids.tolist(), k_min=3, k_max=16, f_min=2)
    return build_segments(ids.tolist(), spans)


@torch.no_grad()
def decode_over_cache(past, first_logits_hidden, start_pos, n_gen):
    """Greedy-decode n_gen tokens on top of an existing cache `past` whose length
    is `start_pos`. first_logits_hidden = last hidden state of the prefill step.
    position_ids continue from start_pos (RoPE anchor = compressed length)."""
    logits = model.lm_head(first_logits_hidden)
    next_id = int(logits.argmax(-1))
    gen_ids = [next_id]
    cur = start_pos
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_gen - 1):
        te = model.model.embed_tokens(
            torch.tensor([[next_id]], device=DEVICE)).to(DTYPE)
        pos = torch.tensor([[cur]], device=DEVICE)
        out = model.model(inputs_embeds=te, past_key_values=past,
                          use_cache=True, position_ids=pos)
        past = out.past_key_values
        next_id = int(model.lm_head(out.last_hidden_state[:, -1, :]).argmax(-1))
        gen_ids.append(next_id)
        cur += 1
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    per_tok_ms = (time.time() - t0) / max(1, n_gen - 1) * 1000
    return gen_ids, per_tok_ms


@torch.no_grad()
def path_baseline(ids, n_gen):
    emb = emb_of(ids).unsqueeze(0)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = model.model(inputs_embeds=emb, use_cache=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    prefill = (time.time() - t0) * 1000
    past = out.past_key_values
    L = ids.shape[0]
    gen, per = decode_over_cache(past, out.last_hidden_state[:, -1, :], L, n_gen)
    return gen, prefill, per, past.get_seq_length()


@torch.no_grad()
def path_cta_mean(ids, n_gen, mode="norm"):
    emb = emb_of(ids)
    segs = seg_of(ids)
    vecs = []
    for (s, e, is_col) in segs:
        if is_col:
            e_bar, R, pi = compose(emb[s:e], mode=mode)
            vecs.append(e_bar)
        else:
            vecs.append(emb[s])
    collapsed = torch.stack(vecs, 0).unsqueeze(0)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = model.model(inputs_embeds=collapsed, use_cache=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    prefill = (time.time() - t0) * 1000
    past = out.past_key_values
    m = collapsed.shape[1]
    gen, per = decode_over_cache(past, out.last_hidden_state[:, -1, :], m, n_gen)
    return gen, prefill, per, past.get_seq_length(), m


@torch.no_grad()
def path_cta_lse(ids, n_gen, use_quad=True, mass=True):
    """Full prefill -> LSE-compress the KV-cache -> decode over compressed cache.
    NOTE: prefill here processes the full prompt (cost ~ baseline prefill); the
    win is in DECODE (m-length cache) + MEMORY. Quality is the target metric."""
    emb = emb_of(ids).unsqueeze(0)
    segs = seg_of(ids)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = model.model(inputs_embeds=emb, use_cache=True)
    full_past = out.past_key_values
    # compress cache
    new_layers = collapse_cache_lse(full_past, segs, q_ref_per_layer=None,
                                    use_quad=use_quad, mass=mass)
    comp = build_dynamic_cache(new_layers)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    prefill = (time.time() - t0) * 1000
    m = comp.get_seq_length()

    # re-derive the last hidden state as if decoding from the compressed cache:
    # feed the LAST real token embedding at position m-? -- simplest honest anchor
    # is to reuse the full model's last hidden state (the compression only changes
    # the PAST that future tokens attend to, not the current step's own hidden).
    first_hidden = out.last_hidden_state[:, -1, :]
    gen, per = decode_over_cache(comp, first_hidden, m, n_gen)

    # Sigma spectrum diagnostics on layer 0 (first collapsed span)
    spec = None
    try:
        K0 = full_past.layers[0].keys[0] if hasattr(full_past, "layers") \
            else full_past.key_cache[0][0]
        for (s, e, is_col) in segs:
            if is_col and (e - s) >= 2:
                spec = sigma_spectrum(K0[:, s:e, :])
                break
    except Exception:
        spec = None
    return gen, prefill, per, m, spec


@torch.no_grad()
def ppl_of_continuation(prompt_ids, gen_ids):
    full = torch.cat([prompt_ids, torch.tensor(gen_ids, device=DEVICE)])
    emb = emb_of(full).unsqueeze(0)
    out = model.model(inputs_embeds=emb)
    logits = model.lm_head(out.last_hidden_state)[0]
    n = len(prompt_ids)
    tgt = torch.tensor(gen_ids, device=DEVICE)
    lp = torch.log_softmax(logits[n - 1:n - 1 + len(gen_ids)], dim=-1)
    ll = lp[torch.arange(len(gen_ids)), tgt]
    return float(torch.exp(-ll.mean()))


def match_rate(a, b):
    return round(sum(1 for x, y in zip(a, b) if x == y) / len(a), 3)


def main():
    prompt = make_rag_prompt()
    ids = tok(prompt, add_special_tokens=False,
              return_tensors="pt").input_ids[0].to(DEVICE)
    n = len(ids)
    print(f"\nprompt tokens n={n}", flush=True)

    a_ids, a_pre, a_per, a_kv = path_baseline(ids, N_GEN)
    b_ids, b_pre, b_per, b_kv, b_m = path_cta_mean(ids, N_GEN)
    # Level 1 variants: mass-only vs mass+quadratic curvature
    c_ids, c_pre, c_per, c_m, c_spec = path_cta_lse(ids, N_GEN, use_quad=True, mass=True)
    d_ids, d_pre, d_per, d_m, _ = path_cta_lse(ids, N_GEN, use_quad=False, mass=True)

    res = {
        "model": MODEL_NAME, "n_prompt": n, "n_gen": N_GEN,
        "collapse_ratio": round(b_m / n, 4),
        "baseline": {"prefill_ms": round(a_pre, 1), "per_tok_ms": round(a_per, 2),
                     "kv_positions": a_kv, "ppl_self": round(ppl_of_continuation(ids, a_ids), 3)},
        "cta_mean": {"prefill_ms": round(b_pre, 1), "per_tok_ms": round(b_per, 2),
                     "kv_positions": b_kv, "ppl": round(ppl_of_continuation(ids, b_ids), 3),
                     "greedy_match": match_rate(a_ids, b_ids)},
        "cta_lse_mass_quad": {"prefill_ms": round(c_pre, 1), "per_tok_ms": round(c_per, 2),
                     "kv_positions": c_m, "ppl": round(ppl_of_continuation(ids, c_ids), 3),
                     "greedy_match": match_rate(a_ids, c_ids)},
        "cta_lse_mass_only": {"prefill_ms": round(d_pre, 1), "per_tok_ms": round(d_per, 2),
                     "kv_positions": d_m, "ppl": round(ppl_of_continuation(ids, d_ids), 3),
                     "greedy_match": match_rate(a_ids, d_ids)},
        "sigma_k_top5_layer0_span0": c_spec,
        "baseline_text": tok.decode(a_ids),
        "cta_mean_text": tok.decode(b_ids),
        "cta_lse_text": tok.decode(c_ids),
    }
    print("\n===== GATE 2 — LEVEL 1 (LSE) RESULTS =====")
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_text")}, indent=2))
    print("\n--- BASELINE ---\n", res["baseline_text"])
    print("\n--- CTA-MEAN ---\n", res["cta_mean_text"])
    print("\n--- CTA-LSE (mass+quad) ---\n", res["cta_lse_text"])
    with open("results_gate2_lse.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\nSaved results_gate2_lse.json")


if __name__ == "__main__":
    main()
