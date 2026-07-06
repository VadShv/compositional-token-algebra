"""
Gate 2 (kill-test): does CTA survive DECODE (autoregressive generation over a
KV-cache built from the COLLAPSED prompt)?

Everything proven so far is PREFILL-only. Products live on decode. The question:
after we collapse a prompt (n -> m) and build a KV-cache over m positions
(keys/values of the pooled e_bar tokens), can we generate coherent continuations
on top of that compressed cache, and is decode actually faster / lighter?

We compare, on the SAME prompt, generating N new tokens three ways:

  A. BASELINE      : standard generation over the full prompt (KV-cache length n).
  B. CTA-DECODE    : collapse prompt -> prefill collapsed embeds with use_cache
                     -> KV-cache length m -> generate N tokens on top of it.
  C. (reference)    : greedy tokens of A vs B compared for agreement + we measure
                     perplexity of B's continuation under the BASELINE model
                     (i.e. does the full model consider B's output plausible?).

Metrics:
  - coherence: do B's greedy tokens match A's? (exact-match rate over N steps)
  - quality  : PPL of B's generated continuation scored by the FULL (baseline)
               model given the FULL prompt (the honest judge).
  - speed    : decode wall-clock per token, baseline vs CTA.
  - memory   : KV-cache tensor size (n vs m positions).

Run on Colab T4. Model: Qwen2.5-3B (same as GPU prefill test) with bf16.
Falls back to Qwen2.5-0.5B if memory is tight.
"""
import sys, time, json, math
import torch

sys.path.insert(0, "/content/compositional-token-algebra/src")
try:
    from cta.detector import select_collapse_spans, build_segments
    from cta.algebra import compose
except Exception:
    # local run fallback
    sys.path.insert(0, "src")
    from cta.detector import select_collapse_spans, build_segments
    from cta.algebra import compose

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
N_GEN = 64          # tokens to generate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE)
model.eval()
cfg = model.config
print(f"model={MODEL_NAME} layers={cfg.num_hidden_layers} d={cfg.hidden_size} "
      f"device={DEVICE} dtype={DTYPE}", flush=True)


# ---------- RAG-like prompt with real internal repetition ----------
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


@torch.no_grad()
def collapse_prompt_embeds(ids, mode="norm"):
    """Return collapsed embeddings [m, d] and (n, m)."""
    emb = emb_of(ids)  # [n, d]
    spans = select_collapse_spans(ids.tolist(), k_min=3, k_max=16, f_min=2)
    segs = build_segments(ids.tolist(), spans)
    vecs = []
    for (s, e, is_col) in segs:
        if is_col:
            e_bar, R, pi = compose(emb[s:e], mode=mode)
            vecs.append(e_bar)
        else:
            vecs.append(emb[s])
    collapsed = torch.stack(vecs, dim=0)  # [m, d]
    return collapsed, len(ids), collapsed.shape[0]


@torch.no_grad()
def generate_from_embeds(inputs_embeds, n_gen, tag=""):
    """Prefill inputs_embeds [L, d] with use_cache, then greedily decode n_gen tokens.
    position_ids continue from L (RoPE sees compressed positions for CTA — the
    honest setup, same anchor convention as prefill)."""
    L = inputs_embeds.shape[0]
    # prefill
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = model.model(inputs_embeds=inputs_embeds.unsqueeze(0), use_cache=True)
    past = out.past_key_values
    h_last = out.last_hidden_state[:, -1, :]
    logits = model.lm_head(h_last)
    next_id = int(logits.argmax(-1))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    prefill_t = time.time() - t0

    gen_ids = [next_id]
    cur_pos = L  # next position index
    # decode loop
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_gen - 1):
        tok_emb = model.model.embed_tokens(
            torch.tensor([[next_id]], device=DEVICE)).to(inputs_embeds.dtype)
        pos = torch.tensor([[cur_pos]], device=DEVICE)
        out = model.model(inputs_embeds=tok_emb, past_key_values=past,
                          use_cache=True, position_ids=pos)
        past = out.past_key_values
        logits = model.lm_head(out.last_hidden_state[:, -1, :])
        next_id = int(logits.argmax(-1))
        gen_ids.append(next_id)
        cur_pos += 1
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    decode_t = time.time() - t0
    per_tok_ms = decode_t / max(1, (n_gen - 1)) * 1000
    return gen_ids, prefill_t * 1000, per_tok_ms, past


@torch.no_grad()
def ppl_of_continuation(prompt_ids, gen_ids):
    """Honest judge: perplexity of gen_ids under the FULL model given the FULL prompt."""
    full = torch.cat([prompt_ids, torch.tensor(gen_ids, device=DEVICE)])
    emb = emb_of(full).unsqueeze(0)
    out = model.model(inputs_embeds=emb)
    logits = model.lm_head(out.last_hidden_state)[0]  # [L, V]
    n = len(prompt_ids)
    # predict gen token t from position n+t-1
    tgt = torch.tensor(gen_ids, device=DEVICE)
    lp = torch.log_softmax(logits[n - 1:n - 1 + len(gen_ids)], dim=-1)
    ll = lp[torch.arange(len(gen_ids)), tgt]
    return float(torch.exp(-ll.mean()))


def kv_positions(past):
    """Number of cached positions (per layer key length)."""
    try:
        # DynamicCache
        return past.get_seq_length()
    except Exception:
        try:
            return past[0][0].shape[-2]
        except Exception:
            return None


def main():
    prompt = make_rag_prompt()
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0].to(DEVICE)
    n = len(ids)
    print(f"\nprompt tokens n={n}", flush=True)

    # --- A. baseline full-prompt decode ---
    emb_full = emb_of(ids)
    a_ids, a_prefill, a_pertok, a_past = generate_from_embeds(emb_full, N_GEN, "baseline")
    a_kv = kv_positions(a_past)

    # --- B. CTA collapsed decode ---
    collapsed, _, m = collapse_prompt_embeds(ids)
    b_ids, b_prefill, b_pertok, b_past = generate_from_embeds(collapsed, N_GEN, "cta")
    b_kv = kv_positions(b_past)

    # --- coherence & quality ---
    match = sum(1 for x, y in zip(a_ids, b_ids) if x == y) / len(a_ids)
    a_text = tok.decode(a_ids)
    b_text = tok.decode(b_ids)
    a_ppl = ppl_of_continuation(ids, a_ids)
    b_ppl = ppl_of_continuation(ids, b_ids)

    res = {
        "model": MODEL_NAME, "n_prompt": n, "m_collapsed": m,
        "collapse_ratio": round(m / n, 4), "n_gen": N_GEN,
        "baseline": {"prefill_ms": round(a_prefill, 1), "per_tok_ms": round(a_pertok, 2),
                     "kv_positions": a_kv, "ppl_self": round(a_ppl, 3)},
        "cta": {"prefill_ms": round(b_prefill, 1), "per_tok_ms": round(b_pertok, 2),
                "kv_positions": b_kv, "ppl_under_full_model": round(b_ppl, 3)},
        "greedy_match_rate": round(match, 3),
        "baseline_text": a_text, "cta_text": b_text,
    }
    print("\n===== GATE 2 RESULTS =====")
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("baseline_text", "cta_text")}, indent=2))
    print("\n--- BASELINE continuation ---\n", a_text)
    print("\n--- CTA-DECODE continuation ---\n", b_text)
    with open("results_gate2.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\nSaved results_gate2.json")


if __name__ == "__main__":
    main()
