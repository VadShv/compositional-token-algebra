"""
Roofline / analytic FLOP estimate for CTA on decoder-only transformers.

Goal: BEFORE spending GPU hours, answer the key infrastructure question ---
"does FlashAttention eat CTA's win?" --- with arithmetic.

CTA shortens the sequence n -> m. That reduces EVERYTHING that is length-linear
(QKVO projections, MLP, LM head) AND the quadratic attention term. FlashAttention
does NOT reduce FLOPs of attention --- it only removes the O(n^2) memory traffic
(no materialized score matrix), making attention IO-cheap. So the compute FLOPs of
attention are unchanged by FA; what FA changes is whether attention is memory-bound.

The honest question is therefore: at a given length, what FRACTION of prefill
compute is length-linear (projections + MLP + head) vs the quadratic attention
score/aggregate? If the linear part dominates at the lengths we care about, then
shrinking n -> m yields a proportional win REGARDLESS of FA, because FA never
touched the linear part. If attention dominates AND FA has made it nearly free in
wall-clock, the realized win at short lengths will be modest.

Per-layer forward FLOPs (multiply-add counted as 2 FLOPs), sequence length n:

  QKV proj      : 2 * n * d * (d + 2*d_kv)        # GQA: KV width d_kv = d * kv/heads
  attn scores   : 2 * n^2 * d                     # Q.K^T   (quadratic)
  attn aggregate: 2 * n^2 * d                     # A.V     (quadratic)
  out proj      : 2 * n * d * d
  MLP (SwiGLU)  : 2 * n * d * (3 * d_ff)          # gate, up, down
LM head (once)  : 2 * n * d * vocab

We report, per length, the quadratic share and the projected CTA speedup ceiling
under a measured collapse ratio r = m/n (from our GPT-2/Qwen-0.5B runs, r ~ 0.63
at 2k). The ceiling on total compute is:
    time(CTA)/time(base) ~ [linear*r + quad*r^2] / [linear + quad]
"""
import json, os

# ---- Model configs (from HF config.json) --------------------------------
CONFIGS = {
    "Qwen2.5-0.5B": dict(L=24, d=896,  d_ff=4864,  heads=14, kv=2,  vocab=151936),
    "Qwen2.5-3B":   dict(L=36, d=2048, d_ff=11008, heads=16, kv=2,  vocab=151936),
    "Qwen2.5-7B":   dict(L=28, d=3584, d_ff=18944, heads=28, kv=4,  vocab=152064),
    "Llama-3-8B":   dict(L=32, d=4096, d_ff=14336, heads=32, kv=8,  vocab=128256),
}

LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]

# Measured collapse ratio r = m/n from our runs (Qwen-0.5B code corpus):
#   256->228 (0.89), 512->421 (0.82), 1024->727 (0.71), 2048->1286 (0.63)
# We extrapolate the deep-redundancy regime; also show a conservative r=0.75.
COLLAPSE_R = {512: 0.82, 1024: 0.71, 2048: 0.63, 4096: 0.60, 8192: 0.58, 16384: 0.55}


def layer_flops(n, c):
    d, d_ff, heads, kv = c["d"], c["d_ff"], c["heads"], c["kv"]
    d_kv = d * kv // heads
    qkv = 2 * n * d * (d + 2 * d_kv)
    out = 2 * n * d * d
    mlp = 2 * n * d * (3 * d_ff)
    linear = qkv + out + mlp
    scores = 2 * n * n * d
    aggr = 2 * n * n * d
    quad = scores + aggr
    return linear, quad


def total_flops(n, c):
    lin_l, quad_l = layer_flops(n, c)
    L = c["L"]
    linear = lin_l * L + 2 * n * c["d"] * c["vocab"]  # + LM head (length-linear)
    quad = quad_l * L
    return linear, quad


def analyze(name):
    c = CONFIGS[name]
    print(f"\n{'='*78}\n{name}  (L={c['L']}, d={c['d']}, d_ff={c['d_ff']}, "
          f"GQA {c['heads']}/{c['kv']}, vocab={c['vocab']})\n{'='*78}")
    print(f"{'len':>6} {'linear%':>9} {'attn(quad)%':>12} {'r=m/n':>7} "
          f"{'CTA time ceil':>14} {'speedup':>9}")
    rows = []
    for n in LENGTHS:
        linear, quad = total_flops(n, c)
        tot = linear + quad
        lin_pct = 100 * linear / tot
        quad_pct = 100 * quad / tot
        r = COLLAPSE_R.get(n, 0.6)
        # compute ceiling: linear scales ~r, quad scales ~r^2
        cta_frac = (linear * r + quad * r * r) / tot
        speedup = 1.0 / cta_frac
        rows.append(dict(len=n, linear_pct=round(lin_pct, 1),
                         attn_quad_pct=round(quad_pct, 1), r=r,
                         cta_time_ceiling=round(cta_frac, 3),
                         speedup_ceiling=round(speedup, 2)))
        print(f"{n:>6} {lin_pct:>8.1f}% {quad_pct:>11.1f}% {r:>7.2f} "
              f"{cta_frac:>13.3f} {speedup:>8.2f}x")
    return rows


def main():
    out = {}
    for name in CONFIGS:
        out[name] = analyze(name)
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "roofline_estimate.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nsaved {path}")
    print("""
INTERPRETATION
--------------
* 'linear%'  = share of prefill COMPUTE that is length-linear (QKVO + MLP + head).
               CTA shrinks this by ~r regardless of FlashAttention.
* 'attn(quad)%' = share in the O(n^2) attention math. FA makes this IO-cheap but
               does NOT cut its FLOPs; CTA shrinks it by ~r^2.
* 'speedup_ceiling' = optimistic upper bound on prefill speedup at that length,
               ignoring detection/compose overhead and kernel-launch effects.

If linear% stays high (>60%) at 2k-8k, CTA's win survives FA because most of the
work was never attention to begin with --- the win rides on shrinking the token
count through MLP/projections/head. That is the go-signal to confirm on real GPU.
""")


if __name__ == "__main__":
    main()
