"""
CTA Level 1 — LSE-pooling in KV space.

Root cause of Gate 2 degradation (CTA-mean, PPL 1.67 -> 3.20):
we collapse spans BEFORE the K/V projection, at the embedding level:
    e_bar = sum_i pi_i x_i           (weighted mean of embeddings)
The model then re-projects the single e_bar into ONE key/value. But attention
is non-linear in q, and softmax weights e^{q.k_i} depend on q which is unknown
at collapse time. A single mean vector cannot carry the span's attention mass.

Level 1 collapses AFTER the K/V projection, directly on the cached keys/values
of each layer, using log-sum-exp so the collapsed token carries the span's TOTAL
attention mass (the missing `log k` term), not just its centroid.

------------------------------------------------------------------------------
DERIVATION (per head, per span of k tokens with keys k_i, values v_i)

True contribution of the span to attention for query q:
    o_true(q) = sum_i w_i(q) v_i,   w_i(q) = e^{q.k_i} / sum_j e^{q.k_j}

We want ONE (k_bar, v_bar) that reproduces this as a single cache entry.

Value  v_bar : best q-independent estimate of sum_i w_i(q) v_i.
    Without knowing q, weight by key mass (a q-free importance proxy):
        v_bar = sum_i omega_i v_i,  omega_i = softmax_i( ||k_i|| / sqrt(d) )

Key   k_bar : the collapsed token's logit q.k_bar must equal the span's
    aggregated logit so it wins the RIGHT share of attention among the rest of
    the context. The correct aggregate of logits under softmax is log-sum-exp:
        q.k_bar  =!  log sum_i e^{q.k_i}
    Exact only if k_i are colinear. First-order (Taylor) around the centroid
    mu_k = mean_i k_i, with key covariance Sigma_k:
        log sum_i e^{q.k_i} ~ log k + q.mu_k + 1/2 q^T Sigma_k q
    =>  k_bar = mu_k + (log k / ||q*||^2) q*  + 1/2 Sigma_k q*
    The `log k` MASS term is what CTA-mean lacks and is the main expected win.
    q* is a reference query (we test q*=mu_k, self-consistent, and q*=mean of a
    future-window of queries).

Applicability bound: single-vector error grows with lambda_max(Sigma_k). If the
span's keys are very dispersed, Level 1 is insufficient -> escalate to Level 2
(rank-r). We report Sigma_k spectrum so the escalation is data-driven.
------------------------------------------------------------------------------
"""
import math
import torch


def _lse_collapse_kv(keys, values, q_ref=None, use_quad=True, mass=True):
    """Collapse one span's per-head keys/values into a single (k_bar, v_bar).

    keys, values : [H, k, d_head]  (already projected, per attention head)
    q_ref        : [H, d_head] reference query for the linearization, or None
                   -> self-consistent q_ref = mu_k.
    Returns k_bar, v_bar : [H, 1, d_head]
    """
    H, k, d = keys.shape
    scale = 1.0 / math.sqrt(d)

    # --- value: mass-weighted (q-free) pooling ---
    key_mass = keys.norm(dim=-1) * scale             # [H, k]
    omega = torch.softmax(key_mass, dim=-1)          # [H, k]
    v_bar = (omega.unsqueeze(-1) * values).sum(dim=1, keepdim=True)  # [H,1,d]

    # --- key: centroid + log(k) mass term + optional quadratic curvature ---
    mu_k = keys.mean(dim=1)                           # [H, d]
    if q_ref is None:
        q_ref = mu_k                                  # self-consistent
    q_norm2 = (q_ref * q_ref).sum(-1, keepdim=True).clamp_min(1e-6)  # [H,1]

    # mass term: shift k_bar along q_ref so that q.k_bar picks up + log k
    mass_term = (math.log(k) / q_norm2) * q_ref if mass else 0.0     # [H,d]

    quad_term = 0.0
    if use_quad:
        # Sigma_k q_ref  (curvature of LSE), per head
        centered = keys - mu_k.unsqueeze(1)          # [H,k,d]
        # (1/k) sum_i (k_i-mu)(k_i-mu)^T q_ref  ==  (1/k) sum_i (centered.q) centered
        proj = (centered * q_ref.unsqueeze(1)).sum(-1, keepdim=True)  # [H,k,1]
        Sq = (proj * centered).mean(dim=1)           # [H,d]
        quad_term = 0.5 * Sq

    k_bar = (mu_k + mass_term + quad_term).unsqueeze(1)  # [H,1,d]
    return k_bar, v_bar


def sigma_spectrum(keys):
    """Return top eigenvalues of the span key covariance (escalation signal)."""
    H, k, d = keys.shape
    mu = keys.mean(dim=1, keepdim=True)
    c = (keys - mu).reshape(H * k, d)
    cov = (c.T @ c) / max(1, H * k - 1)
    try:
        ev = torch.linalg.eigvalsh(cov.float())
        return ev.flip(0)[:5].tolist()
    except Exception:
        return None


def collapse_cache_lse(past_kv, segments, q_ref_per_layer=None,
                       use_quad=True, mass=True):
    """Build a compressed KV-cache (length m) from a full cache (length n) by
    LSE-pooling keys/values inside each collapsed segment, per layer & head.

    past_kv   : list over layers of (key, value), each [1, H, n, d_head]
                (legacy tuple format) OR a transformers Cache exposing
                .key_cache / .value_cache lists.
    segments  : list of (start, end, is_collapsed) over the ORIGINAL n tokens.
    q_ref_per_layer : optional list [L] of [H, d_head] reference queries.
    Returns   : list over layers of (key, value) each [1, H, m, d_head].
    """
    # normalize input to per-layer (k, v) tensors across transformers versions
    if hasattr(past_kv, "key_cache"):                 # <=4.x DynamicCache
        layers = list(zip(past_kv.key_cache, past_kv.value_cache))
    elif hasattr(past_kv, "layers"):                  # >=5.x DynamicCache
        layers = [(l.keys, l.values) for l in past_kv.layers]
    else:                                             # legacy tuple format
        layers = list(past_kv)

    new_layers = []
    for li, (K, V) in enumerate(layers):
        # K,V: [1, H, n, d]
        Kb, Vb = K[0], V[0]                     # [H, n, d]
        q_ref = None
        if q_ref_per_layer is not None:
            q_ref = q_ref_per_layer[li]
        new_k, new_v = [], []
        for (s, e, is_col) in segments:
            if is_col and (e - s) >= 2:
                kk = Kb[:, s:e, :]              # [H, k, d]
                vv = Vb[:, s:e, :]
                q = q_ref if q_ref is not None else None
                k_bar, v_bar = _lse_collapse_kv(kk, vv, q_ref=q,
                                                use_quad=use_quad, mass=mass)
                new_k.append(k_bar)            # [H,1,d]
                new_v.append(v_bar)
            else:
                new_k.append(Kb[:, s:s + 1, :])
                new_v.append(Vb[:, s:s + 1, :])
        nk = torch.cat(new_k, dim=1).unsqueeze(0)  # [1,H,m,d]
        nv = torch.cat(new_v, dim=1).unsqueeze(0)
        new_layers.append((nk, nv))
    return new_layers


def build_dynamic_cache(new_layers):
    """Assemble a transformers DynamicCache from per-layer (k,v) tensors via the
    public update() API (version-robust: no reliance on internal field names).
    Each entry: [1, H, m, d_head].
    """
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for li, (k, v) in enumerate(new_layers):
        cache.update(k.contiguous(), v.contiguous(), li)
    return cache
