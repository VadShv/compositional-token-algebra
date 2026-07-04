"""Diagnose the source of degradation: position compression vs content opacity.

Variant A (compressed pos): collapsed seq uses positions 0..m-1 (what we ran).
Variant B (anchor pos):     collapsed token keeps original anchor position id;
                            we feed explicit position_ids to the transformer.

If B >> A, damage is from position scrambling. If both bad, damage is content opacity.
"""
import warnings
warnings.filterwarnings("ignore")
import _bootstrap  # noqa: F401  (sets sys.path + result dirs)
import torch, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from data import SAMPLES
from cta.detector import select_collapse_spans, build_segments
from cta.algebra import compose
from cta.model import input_embeddings, baseline_forward

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2"); model.eval()
for p in model.parameters(): p.requires_grad_(False)


@torch.no_grad()
def cta_with_positions(model, input_ids, segments, use_anchor_pos, score_mode="uniform"):
    emb, _ = input_embeddings(model, input_ids)
    vecs, pos_ids, seg_meta = [], [], []
    for (s, e, is_col) in segments:
        if is_col:
            e_bar, R, pi = compose(emb[s:e], mode=score_mode)
            vecs.append(e_bar)
        else:
            vecs.append(emb[s])
        pos_ids.append(s if use_anchor_pos else len(vecs) - 1)
        seg_meta.append((s, e, is_col))
    collapsed = torch.stack(vecs).unsqueeze(0)
    position_ids = torch.tensor(pos_ids).unsqueeze(0)
    out = model.transformer(inputs_embeds=collapsed, position_ids=position_ids)
    hidden = out.last_hidden_state.squeeze(0)
    logits = model.lm_head(hidden)
    return logits, seg_meta


@torch.no_grad()
def eval_variant(input_ids, use_anchor_pos):
    spans = select_collapse_spans(input_ids.tolist(), 3, 16, 2)
    segments = build_segments(input_ids.tolist(), spans)
    base_logits, _ = baseline_forward(model, input_ids)
    cta_logits, seg_meta = cta_with_positions(model, input_ids, segments, use_anchor_pos)
    m = len(segments)
    seen = False; prefix = []
    for (s,e,c) in segments:
        prefix.append(seen);  seen = seen or c
    bn=cn=t=0
    for j in range(1, m):
        if not prefix[j]: continue
        s_j = seg_meta[j][0]; tgt = input_ids[s_j]
        bn += -F.log_softmax(base_logits[s_j-1],-1)[tgt].item()
        cn += -F.log_softmax(cta_logits[j-1],-1)[tgt].item()
        t += 1
    return torch.exp(torch.tensor(bn/t)).item(), torch.exp(torch.tensor(cn/t)).item(), t

print(f"{'sample':7s} {'base_ppl':>9s} {'cta(compressed)':>16s} {'cta(anchor_pos)':>16s}")
for name, text in SAMPLES.items():
    ids = torch.tensor(tok(text)["input_ids"])
    if not select_collapse_spans(ids.tolist(),3,16,2):
        print(f"{name:7s}  (no repeats)"); continue
    b, c_comp, _ = eval_variant(ids, use_anchor_pos=False)
    _, c_anch, _ = eval_variant(ids, use_anchor_pos=True)
    print(f"{name:7s} {b:>9.3f} {c_comp:>16.3f} {c_anch:>16.3f}")
