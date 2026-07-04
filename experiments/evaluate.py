"""
Evaluation harness for CTA vs baseline on repetition-heavy text.

Fair comparison protocol:
  - Both models predict the SAME set of original target tokens.
  - Eval targets = the first token of each segment that comes AFTER at least one
    collapsed span earlier in the sequence. At these positions, baseline attends to
    the FULL repeated spans while CTA attends to POOLED single tokens -> tests whether
    context compression hurts next-token prediction.
  - Baseline logits at original position (t-1) predict token t.
  - CTA logits at collapsed segment index (j-1) predict the token at original start of segment j.

Metrics: NLL / perplexity on common targets, attention-FLOPs ratio, lossless residual error.
"""
import _bootstrap  # noqa: F401  (sets sys.path + result dirs)
import torch
import torch.nn.functional as F
from cta.detector import select_collapse_spans, build_segments
from cta.model import baseline_forward, cta_forward, attention_flops, input_embeddings
from cta.algebra import compose, decompose


@torch.no_grad()
def evaluate_sequence(model, input_ids, k_min=3, k_max=16, f_min=2, score_mode="norm"):
    n = input_ids.shape[0]
    spans = select_collapse_spans(input_ids.tolist(), k_min, k_max, f_min)
    if not spans:
        return None
    segments = build_segments(input_ids.tolist(), spans)
    m = len(segments)

    # Baseline over full sequence
    base_logits, _ = baseline_forward(model, input_ids)      # [n, V]
    # CTA over collapsed sequence
    cta_logits, m2, seg_meta = cta_forward(model, input_ids, segments, score_mode)  # [m, V]

    # Determine eval targets: start-of-segment tokens for segments j>=1 where some
    # earlier segment was collapsed.
    collapsed_prefix = []
    seen_collapsed = False
    for (s, e, is_col) in segments:
        collapsed_prefix.append(seen_collapsed)
        if is_col:
            seen_collapsed = True

    base_nll, cta_nll, n_targets = 0.0, 0.0, 0
    for j in range(1, m):
        if not collapsed_prefix[j]:
            continue  # only eval where context compression actually happened
        s_j = seg_meta[j][0]                 # original index of target token
        target = input_ids[s_j]
        # baseline predicts target from position s_j - 1
        b_lp = F.log_softmax(base_logits[s_j - 1], dim=-1)[target]
        # CTA predicts target from collapsed segment index j-1
        c_lp = F.log_softmax(cta_logits[j - 1], dim=-1)[target]
        base_nll += -b_lp.item()
        cta_nll += -c_lp.item()
        n_targets += 1

    if n_targets == 0:
        return None

    n_layer = model.config.n_layer
    d = model.config.n_embd
    flops_base = attention_flops(n, n_layer, d)
    flops_cta = attention_flops(m, n_layer, d)

    # lossless check on one collapsed span
    emb, _ = input_embeddings(model, input_ids)
    err = 0.0
    for (s, e, is_col) in segments:
        if is_col:
            e_bar, R, pi = compose(emb[s:e], mode=score_mode)
            recon = decompose(e_bar, R)
            err = max(err, (recon - emb[s:e]).abs().max().item())

    return {
        "n": n, "m": m, "n_targets": n_targets,
        "base_nll": base_nll / n_targets, "cta_nll": cta_nll / n_targets,
        "base_ppl": torch.exp(torch.tensor(base_nll / n_targets)).item(),
        "cta_ppl": torch.exp(torch.tensor(cta_nll / n_targets)).item(),
        "flops_base": flops_base, "flops_cta": flops_cta,
        "flops_ratio": flops_cta / flops_base,
        "len_ratio": m / n,
        "lossless_err": err,
    }
