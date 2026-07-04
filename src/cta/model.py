"""
CTA integration over a frozen GPT-2.

Two forward paths that we compare on IDENTICAL target positions:

  BASELINE: standard GPT-2 over full input embeddings (length n).
  CTA:      collapse repeated spans into single pooled vectors (compose),
            run GPT-2 over the shortened sequence (length m < n),
            then read out hidden states at "query positions" that fall AFTER
            collapsed regions to predict the next token.

We compare next-token prediction quality at a common set of eval positions:
positions immediately following a collapsed span (the context they attend to
differs: baseline sees full span, CTA sees a single pooled token).

Attention FLOPs are proxied by sum over layers of L^2 * d (self-attention QK^T + AV),
so the ratio (m/n)^2 captures the quadratic saving.

The pooled span vector carries positional info implicitly because we compose the
POST-embedding vectors (wte+wpe). We assign the collapsed token the position id of
the span's FIRST token (anchor), which is the conservative choice.
"""
import torch
import torch.nn.functional as F
from .algebra import compose


@torch.no_grad()
def input_embeddings(model, input_ids):
    """Return token embeddings wte(input_ids) ONLY. Positional embeddings are added
    by model.transformer(inputs_embeds=...) internally, over the (possibly collapsed)
    sequence positions. This is the honest setup: a collapsed sequence occupies
    compressed positions.  input_ids: [n] -> [n, d]"""
    wte = model.transformer.wte(input_ids)
    return wte, torch.arange(input_ids.shape[0], device=input_ids.device)


@torch.no_grad()
def run_blocks(model, inp_embeds, position_ids=None):
    """Run GPT-2 blocks over precomputed input embeddings [m, d] (causal).
    We pass inputs_embeds through the transformer but must avoid double-adding wpe.
    Trick: temporarily zero the position embedding contribution by subtracting wpe
    for the default positions inside the caller. Simpler: use inputs_embeds and let
    HF add wpe for positions 0..m-1. Since our compose already mixed original wpe into
    the residual, we instead pass RAW token-mixed embeddings WITHOUT wpe from caller.
    Returns final hidden states [m, d] (after ln_f)."""
    out = model.transformer(inputs_embeds=inp_embeds.unsqueeze(0))
    return out.last_hidden_state.squeeze(0)  # [m, d]


@torch.no_grad()
def baseline_forward(model, input_ids):
    emb, _ = input_embeddings(model, input_ids)
    hidden = run_blocks(model, emb)
    logits = model.lm_head(hidden)  # [n, V]
    return logits, input_ids.shape[0]


@torch.no_grad()
def cta_forward(model, input_ids, segments, score_mode="norm"):
    """Collapse segments and run. Returns:
       logits_collapsed [m, V], m, mapping seg_index->(orig_start,orig_end,is_collapsed)
    """
    emb, _ = input_embeddings(model, input_ids)  # [n, d]
    collapsed_vectors = []
    seg_meta = []
    for (s, e, is_col) in segments:
        if is_col:
            e_bar, R, pi = compose(emb[s:e], mode=score_mode)
            collapsed_vectors.append(e_bar)
        else:
            collapsed_vectors.append(emb[s])  # single token
        seg_meta.append((s, e, is_col))
    collapsed = torch.stack(collapsed_vectors, dim=0)  # [m, d]
    hidden = run_blocks(model, collapsed)
    logits = model.lm_head(hidden)  # [m, V]
    return logits, collapsed.shape[0], seg_meta


def attention_flops(seq_len, n_layer, d):
    """Proxy for self-attention cost: per layer 2 * L^2 * d  (QK^T and AV)."""
    return n_layer * 2 * (seq_len ** 2) * d
