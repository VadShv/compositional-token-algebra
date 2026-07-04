import os, json, matplotlib
matplotlib.use("Agg")
import _bootstrap  # noqa: F401  (sets sys.path + result dirs)
import matplotlib.pyplot as plt

R = json.load(open(os.path.join(_bootstrap.RESULTS_RAW, "results_gate.json")))
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

for ax, name in zip(axes, ["code", "tickets"]):
    d = R[name]
    lx = [100*p["frac_expand"] for p in d["learned"]]
    ly = [p["nll"] for p in d["learned"]]
    hx = [100*p["frac_expand"] for p in d["heuristic"]]
    hy = [p["nll"] for p in d["heuristic"]]
    # sort by expand frac for clean lines
    ls = sorted(zip(lx, ly)); hs = sorted(zip(hx, hy))
    ax.plot([a for a,_ in ls], [b for _,b in ls], "-o", color="#2a9d8f",
            lw=2.2, ms=6, label="Learned gate (MLP)")
    ax.plot([a for a,_ in hs], [b for _,b in hs], "--s", color="#e76f51",
            lw=2, ms=5, label="Residual-energy heuristic")
    ax.axhline(d["all_pool_nll"], color="gray", ls=":", lw=1.3, label="all-pooled (max FLOPs saving)")
    ax.axhline(d["all_exp_nll"], color="black", ls="-.", lw=1.1, label="all-expanded (no saving)")
    ax.set_xlabel("Spans expanded (%)  — lower = fewer attention FLOPs")
    ax.set_ylabel("Next-token NLL (nats)  — lower = better")
    ax.set_title(f"{name}: quality / compute frontier")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(_bootstrap.RESULTS_FIG, "gate_frontier.png"), dpi=140, bbox_inches="tight")
print("saved gate_frontier.png")
