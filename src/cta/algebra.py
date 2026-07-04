"""
Compositional Token Algebra (CTA) — core operators.

compose:  (x_a..x_b) -> (e_bar, R)   weighted pooling + residuals
decompose:(e_bar, R) -> (x_a..x_b)   exact inverse (lossless by construction)

No trainable parameters in compose/decompose. The scoring function g(.) is fixed.
"""
import torch


def score_fn(x, mode="norm", lam=0.5):
    """Fixed (non-trainable) scoring g(x_i) over a span. x: [k, d] -> logits [k]."""
    k = x.shape[0]
    if mode == "uniform":
        return torch.zeros(k, dtype=x.dtype, device=x.device)
    if mode == "norm":
        return x.norm(dim=-1)
    if mode == "posdecay":
        idx = torch.arange(k, dtype=x.dtype, device=x.device)
        return -lam * (k - 1 - idx)  # emphasize right boundary
    if mode == "selfconsist":
        mean = x.mean(dim=0, keepdim=True)
        return (x * mean).sum(-1)
    raise ValueError(mode)


def compose(span, mode="norm"):
    """span: [k, d]. Returns e_bar [d], R [k, d], pi [k]."""
    logits = score_fn(span, mode)
    pi = torch.softmax(logits, dim=0)          # [k]
    e_bar = (pi.unsqueeze(-1) * span).sum(0)   # [d]
    R = span - e_bar.unsqueeze(0)              # [k, d]  residual store
    return e_bar, R, pi


def decompose(e_bar, R):
    """Exact inverse: x_i = e_bar + R_i."""
    return e_bar.unsqueeze(0) + R             # [k, d]


if __name__ == "__main__":
    torch.manual_seed(0)
    max_err = 0.0
    for mode in ["uniform", "norm", "posdecay", "selfconsist"]:
        for _ in range(200):
            k = torch.randint(2, 20, (1,)).item()
            d = 768
            span = torch.randn(k, d)
            e_bar, R, pi = compose(span, mode)
            recon = decompose(e_bar, R)
            err = (recon - span).abs().max().item()
            max_err = max(max_err, err)
        print(f"mode={mode:12s}  max reconstruction error so far: {max_err:.2e}")
    print(f"\nLOSSLESS CHECK: global max error = {max_err:.2e}")
    assert max_err < 1e-4, "not lossless!"
    print("PASS: decompose(compose(x)) == x within machine epsilon")
