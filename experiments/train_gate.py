"""
Train the learnable gate on cached per-span outcomes and trace the quality/FLOPs frontier.

Differentiable objective (GPT-2 frozen; only gate trained):
  For each span i with expand-prob p_i (from Gumbel-Sigmoid of gate logit):
     expected_nll_i = p_i * nll_exp_i + (1 - p_i) * nll_pooled_i
  Loss = mean_i expected_nll_i  +  lambda * mean_i p_i
                (task CE proxy)      (sparsity: fewer expansions -> fewer FLOPs)

Sweeping lambda traces the frontier. We compare against the residual-energy heuristic
at matched expansion budgets.
"""
import os, json, torch, torch.nn as nn
import _bootstrap  # noqa: F401  (sets sys.path + result dirs)
from cta.learned_gate import GateMLP, gumbel_sigmoid

torch.manual_seed(0)


def load(name):
    d = torch.load(os.path.join(_bootstrap.RESULTS_RAW, f"cache_{name}.pt"))
    # normalize scalar features
    d["res_n"] = (d["res_energy"] - d["res_energy"].mean()) / (d["res_energy"].std() + 1e-6)
    d["len_n"] = (d["span_len"] - d["span_len"].mean()) / (d["span_len"].std() + 1e-6)
    return d


def train_gate(d, lam, epochs=400, lr=1e-3):
    D = d["q"].shape[1]
    gate = GateMLP(D)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        logits = gate(d["q"], d["e_bar"], d["res_n"], d["len_n"])
        p = gumbel_sigmoid(logits, tau=0.7, hard=False, training=True)
        exp_nll = p * d["nll_exp"] + (1 - p) * d["nll_pooled"]
        loss = exp_nll.mean() + lam * p.mean()
        loss.backward(); opt.step()
    return gate


@torch.no_grad()
def eval_gate(gate, d):
    logits = gate(d["q"], d["e_bar"], d["res_n"], d["len_n"])
    p = torch.sigmoid(logits)
    expand = (p > 0.5).float()
    frac_expand = expand.mean().item()
    # realized NLL under hard decision
    nll = expand * d["nll_exp"] + (1 - expand) * d["nll_pooled"]
    return frac_expand, nll.mean().item()


@torch.no_grad()
def eval_heuristic(d, frac_target):
    """Residual-energy heuristic: expand top-frac by res_energy."""
    n = d["res_energy"].shape[0]
    k = int(round(frac_target * n))
    order = torch.argsort(d["res_energy"], descending=True)
    expand = torch.zeros(n); expand[order[:k]] = 1
    nll = expand * d["nll_exp"] + (1 - expand) * d["nll_pooled"]
    return expand.mean().item(), nll.mean().item()


if __name__ == "__main__":
    import json
    results = {}
    for name in ["code", "tickets"]:
        d = load(name)
        allpool = d["nll_pooled"].mean().item()
        allexp = d["nll_exp"].mean().item()
        pts_learned, pts_heur = [], []
        for lam in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0]:
            gate = train_gate(d, lam)
            fe, nll = eval_gate(gate, d)
            pts_learned.append({"lam": lam, "frac_expand": fe, "nll": nll})
            # matched-budget heuristic
            feh, nllh = eval_heuristic(d, fe)
            pts_heur.append({"frac_expand": feh, "nll": nllh})
        results[name] = {"all_pool_nll": allpool, "all_exp_nll": allexp,
                         "learned": pts_learned, "heuristic": pts_heur, "n_spans": len(d["q"])}
        print(f"\n=== {name} (n={len(d['q'])}) ===")
        print(f"all-pooled NLL={allpool:.3f}  all-expanded NLL={allexp:.3f}")
        print(f"{'lambda':>7s} {'expand%':>8s} {'learned_nll':>12s} {'heur_nll@same':>14s} {'gain':>7s}")
        for l, h in zip(pts_learned, pts_heur):
            gain = h["nll"] - l["nll"]
            print(f"{l['lam']:>7.2f} {100*l['frac_expand']:>7.1f}% {l['nll']:>12.3f} "
                  f"{h['nll']:>14.3f} {gain:>+7.3f}")
    json.dump(results, open(os.path.join(_bootstrap.RESULTS_RAW, "results_gate.json"),"w"), indent=2, default=float)
    print("\nsaved results_gate.json")
