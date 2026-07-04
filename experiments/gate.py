"""
Gated CTA: selectively expand collapsed spans.

Mechanism: for each collapsed span we compute an 'expand utility' and expand the
top-B spans (budget B) back to full tokens; the rest stay pooled. This traces the
quality vs attention-FLOPs frontier.

Utility signal (parameter-free, causal-safe): the L2 norm of the span's residual
energy  ||R||_F  — spans whose internal tokens deviate most from their centroid lose
the most information when pooled, so they are the best expansion candidates.
This is a cheap proxy for the learnable gate gamma(q, e_bar) in the spec.
"""
import warnings, os, json
warnings.filterwarnings("ignore")
import _bootstrap  # noqa: F401  (sets sys.path + result dirs)
import torch, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from data import SAMPLES
from cta.detector import select_collapse_spans, build_segments
from cta.algebra import compose
from cta.model import input_embeddings, baseline_forward, attention_flops

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2"); model.eval()
for p in model.parameters(): p.requires_grad_(False)
N_LAYER, D = model.config.n_layer, model.config.n_embd


@torch.no_grad()
def build_collapsed_input(model, input_ids, segments, expand_flags, score_mode="uniform"):
    """expand_flags: dict seg_index->bool. If a collapsed seg is expanded, emit its
    original tokens; else emit pooled vector. Returns embeds [m',d], seg_meta, and
    for each emitted row the original position it predicts-from mapping.
    Also returns, for each ORIGINAL segment, the collapsed row index of its LAST emitted row."""
    emb, _ = input_embeddings(model, input_ids)
    vecs, row_seg, seg_last_row = [], [], {}
    for si, (s, e, is_col) in enumerate(segments):
        if is_col and not expand_flags.get(si, False):
            e_bar, R, pi = compose(emb[s:e], mode=score_mode)
            vecs.append(e_bar); row_seg.append(si)
        else:
            for t in range(s, e):
                vecs.append(emb[t]); row_seg.append(si)
        seg_last_row[si] = len(vecs) - 1
    collapsed = torch.stack(vecs).unsqueeze(0)
    return collapsed, seg_last_row


@torch.no_grad()
def residual_energy(model, input_ids, segments, score_mode="uniform"):
    emb, _ = input_embeddings(model, input_ids)
    energy = {}
    for si, (s, e, is_col) in enumerate(segments):
        if is_col:
            e_bar, R, pi = compose(emb[s:e], mode=score_mode)
            energy[si] = R.norm().item()
    return energy


@torch.no_grad()
def eval_gated(input_ids, budget_frac, score_mode="uniform"):
    """budget_frac in [0,1]: fraction of collapsed spans allowed to expand (highest residual energy first)."""
    spans = select_collapse_spans(input_ids.tolist(), 3, 16, 2)
    segments = build_segments(input_ids.tolist(), spans)
    m_full = len(segments)
    collapsed_ids = [si for si,(s,e,c) in enumerate(segments) if c]
    energy = residual_energy(model, input_ids, segments, score_mode)
    order = sorted(collapsed_ids, key=lambda si: -energy[si])
    n_expand = int(round(budget_frac * len(collapsed_ids)))
    expand_flags = {si: True for si in order[:n_expand]}

    collapsed, seg_last_row = build_collapsed_input(model, input_ids, segments, expand_flags, score_mode)
    m_eff = collapsed.shape[1]
    out = model.transformer(inputs_embeds=collapsed)
    logits = model.lm_head(out.last_hidden_state.squeeze(0))
    base_logits, n = baseline_forward(model, input_ids)

    # eval targets: start-of-segment for segments after any collapsed(unexpanded) span
    seen=False; prefix=[]
    for si,(s,e,c) in enumerate(segments):
        prefix.append(seen); seen = seen or (c and not expand_flags.get(si,False))
    bn=cn=t=0
    for j in range(1, m_full):
        if not prefix[j]: continue
        s_j = segments[j][0]; tgt = input_ids[s_j]
        prev_row = seg_last_row[j-1]
        bn += -F.log_softmax(base_logits[s_j-1],-1)[tgt].item()
        cn += -F.log_softmax(logits[prev_row],-1)[tgt].item()
        t += 1
    if t==0: return None
    return {
        "base_ppl": torch.exp(torch.tensor(bn/t)).item(),
        "cta_ppl": torch.exp(torch.tensor(cn/t)).item(),
        "flops_ratio": attention_flops(m_eff,N_LAYER,D)/attention_flops(n,N_LAYER,D),
        "len_eff": m_eff, "n": n, "n_expand": n_expand, "n_collapsed_spans": len(collapsed_ids),
        "n_targets": t,
    }


if __name__ == "__main__":
    import json
    frontier = {}
    for name, text in SAMPLES.items():
        ids = torch.tensor(tok(text)["input_ids"])
        if not select_collapse_spans(ids.tolist(),3,16,2):
            continue
        pts = []
        for bf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r = eval_gated(ids, bf)
            if r: pts.append({"budget": bf, **r})
        frontier[name] = pts
        print(f"\n=== {name} ===")
        print(f"{'budget':>7s} {'flops%':>7s} {'base_ppl':>9s} {'cta_ppl':>9s} {'d_ppl%':>8s} {'expanded':>9s}")
        for p in pts:
            d = 100*(p['cta_ppl']-p['base_ppl'])/p['base_ppl']
            print(f"{p['budget']:>7.2f} {100*p['flops_ratio']:>6.1f}% {p['base_ppl']:>9.3f} "
                  f"{p['cta_ppl']:>9.3f} {d:>+7.1f}% {p['n_expand']:>3d}/{p['n_collapsed_spans']:<3d}")
    with open(os.path.join(_bootstrap.RESULTS_RAW, "results_frontier.json"),"w") as f:
        json.dump(frontier, f, indent=2, default=float)
    print("\nsaved results_frontier.json")
