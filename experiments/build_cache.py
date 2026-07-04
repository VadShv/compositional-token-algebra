"""
Precompute per-span training cache for the learnable gate.

For each text chunk we detect collapsed spans and, for every span, compute:
  features: q (next-token emb), e_bar, res_energy, span_len
  outcomes: NLL_pooled and NLL_expanded for the downstream target token that this
            span is the immediate context for (the token right after the span).

This decouples the expensive frozen-GPT-2 forward passes (done once here) from the
cheap gate training (done many epochs later on cached tensors).

Approximation for tractability: we evaluate each span's LOCAL effect by comparing,
on the SAME collapsed sequence, the prediction of the post-span target when THIS
span is pooled vs expanded (all OTHER spans pooled). This is the marginal utility of
expanding this one span — exactly what a per-span gate needs.
"""
import warnings, os, glob, torch, torch.nn.functional as F
warnings.filterwarnings("ignore")
import _bootstrap  # noqa: F401  (sets sys.path + result dirs)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from cta.detector import select_collapse_spans, build_segments
from cta.algebra import compose
from cta.model import input_embeddings

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2"); model.eval()
for p in model.parameters(): p.requires_grad_(False)
D = model.config.n_embd
CHUNK = 384  # tokens per chunk (keeps seq short for CPU)


@torch.no_grad()
def emit_with_one_expanded(emb, segments, expand_si):
    """Build collapsed embeds where only segment `expand_si` is expanded; others pooled.
    Returns embeds [m,d] and dict seg->last_row."""
    vecs, seg_last = [], {}
    for si,(s,e,c) in enumerate(segments):
        if c and si != expand_si:
            e_bar,R,pi = compose(emb[s:e]); vecs.append(e_bar)
        elif c and si == expand_si:
            for t in range(s,e): vecs.append(emb[t])
        else:
            vecs.append(emb[s])
        seg_last[si] = len(vecs)-1
    return torch.stack(vecs), seg_last


@torch.no_grad()
def process_text(text, max_chunks=12, max_spans_per_chunk=12):
    # cap input to avoid tokenizing megabytes; ~ max_chunks*CHUNK tokens needed
    text = text[: (max_chunks + 2) * CHUNK * 5]
    ids_all = tok(text, truncation=False)["input_ids"]
    print(f"  tokenized {len(ids_all)} ids", flush=True)
    rows = []
    n_chunks = 0
    for c0 in range(0, len(ids_all), CHUNK):
        if n_chunks >= max_chunks: break
        ids = torch.tensor(ids_all[c0:c0+CHUNK])
        if ids.shape[0] < 20: continue
        n_chunks += 1
        spans = select_collapse_spans(ids.tolist(), 3, 16, 2)
        if not spans: continue
        segments = build_segments(ids.tolist(), spans)
        emb,_ = input_embeddings(model, ids)
        collapsed_ids = [si for si,(s,e,c) in enumerate(segments) if c][:max_spans_per_chunk]

        # baseline all-pooled pass
        vecs_pooled, seg_last_p = emit_with_one_expanded(emb, segments, expand_si=-1)
        out_p = model.transformer(inputs_embeds=vecs_pooled.unsqueeze(0))
        logits_p = model.lm_head(out_p.last_hidden_state.squeeze(0))

        for si in collapsed_ids:
            s,e,_ = segments[si]
            # target = token right after this span (start of next segment)
            if si+1 >= len(segments): continue
            tgt_pos = segments[si+1][0]
            tgt = ids[tgt_pos]
            # pooled NLL: predict from last row of span si in all-pooled layout
            nll_pooled = -F.log_softmax(logits_p[seg_last_p[si]], -1)[tgt].item()
            # expanded NLL: expand only si
            vecs_e, seg_last_e = emit_with_one_expanded(emb, segments, expand_si=si)
            out_e = model.transformer(inputs_embeds=vecs_e.unsqueeze(0))
            logits_e = model.lm_head(out_e.last_hidden_state.squeeze(0))
            nll_exp = -F.log_softmax(logits_e[seg_last_e[si]], -1)[tgt].item()

            e_bar,R,pi = compose(emb[s:e])
            q = emb[tgt_pos] if tgt_pos < emb.shape[0] else emb[-1]
            rows.append({
                "q": q, "e_bar": e_bar,
                "res_energy": R.norm().item(), "span_len": float(e-s),
                "nll_pooled": nll_pooled, "nll_exp": nll_exp,
            })
        print(f"  chunk {n_chunks}: {len(collapsed_ids)} spans, total rows={len(rows)}", flush=True)
    return rows


if __name__ == "__main__":
    import pickle
    code_files = sorted(glob.glob(os.path.join(_bootstrap.CORPUS, "code_*.py")))
    code_text = "\n".join(open(f).read() for f in code_files)
    ticket_text = open(os.path.join(_bootstrap.CORPUS, "tickets.txt")).read()

    for name, text in [("code", code_text), ("tickets", ticket_text)]:
        print(f"processing {name}...", flush=True)
        rows = process_text(text)
        q = torch.stack([r["q"] for r in rows])
        e = torch.stack([r["e_bar"] for r in rows])
        re_ = torch.tensor([r["res_energy"] for r in rows])
        sl = torch.tensor([r["span_len"] for r in rows])
        nllp = torch.tensor([r["nll_pooled"] for r in rows])
        nlle = torch.tensor([r["nll_exp"] for r in rows])
        torch.save({"q":q,"e_bar":e,"res_energy":re_,"span_len":sl,
                    "nll_pooled":nllp,"nll_exp":nlle}, os.path.join(_bootstrap.RESULTS_RAW, f"cache_{name}.pt"))
        print(f"{name}: {len(rows)} spans | mean nll_pooled={nllp.mean():.3f} "
              f"nll_exp={nlle.mean():.3f} | pooling hurts by {(nllp-nlle).mean():.3f} nats avg")
