"""
Learnable expand-gate for CTA.

Per collapsed span s the gate decides expand (1) vs keep-pooled (0).
Input features (parameter-free to compute, then fed to a small MLP):
  - e_bar        : pooled span embedding                       [d]
  - q            : "consumer query" = token embedding of the   [d]
                   token immediately AFTER the span (the position that must
                   predict the next token using this span as context)
  - res_energy   : ||R||_F  (scalar, how much info pooling drops)
  - span_len     : k        (scalar)

Output: expand logit -> Gumbel-Sigmoid (differentiable hard 0/1 at train,
hard threshold at eval).

Loss = CrossEntropy(next-token prediction over eval targets)
     + lambda_sparsity * mean(expand_prob)          # push toward fewer expansions (fewer FLOPs)

The GPT-2 backbone stays FROZEN. Only the gate (~few k params) is trained.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class GateMLP(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        # input: q (d) + e_bar (d) + [res_energy, span_len] (2 scalars, normalized)
        self.net = nn.Sequential(
            nn.Linear(2 * d + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q, e_bar, res_energy, span_len):
        # q, e_bar: [N, d]; res_energy, span_len: [N]
        feats = torch.cat([
            q, e_bar,
            res_energy.unsqueeze(-1),
            span_len.unsqueeze(-1),
        ], dim=-1)
        return self.net(feats).squeeze(-1)  # [N] logits


def gumbel_sigmoid(logits, tau=1.0, hard=False, training=True):
    if training:
        # Binary concrete / Gumbel-Sigmoid
        u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        noise = torch.log(u) - torch.log(1 - u)
        y = torch.sigmoid((logits + noise) / tau)
    else:
        y = torch.sigmoid(logits)
    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard + (y - y.detach())  # straight-through
    return y
