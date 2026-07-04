import os, json, matplotlib
matplotlib.use("Agg")
import _bootstrap  # noqa: F401  (sets sys.path + result dirs)
import matplotlib.pyplot as plt

frontier = json.load(open(os.path.join(_bootstrap.RESULTS_RAW, "results_frontier.json")))
killer = {"code":(0,256),"logs":(3,189),"chat":(15,89)}

fig, axes = plt.subplots(1, 2, figsize=(13,5))

# --- Plot 1: quality/FLOPs frontier ---
ax = axes[0]
colors = {"code":"#2a9d8f","logs":"#e76f51","chat":"#264653"}
for name, pts in frontier.items():
    xs = [100*p["flops_ratio"] for p in pts]
    ys = [100*(p["cta_ppl"]-p["base_ppl"])/p["base_ppl"] for p in pts]
    ax.plot(xs, ys, "-o", color=colors[name], label=name, linewidth=2, markersize=6)
ax.axhline(0, color="gray", ls="--", lw=1, label="baseline quality")
ax.axhline(2, color="green", ls=":", lw=1, label="+2% target")
ax.set_xlabel("Attention FLOPs (% of baseline)")
ax.set_ylabel("Perplexity change vs baseline (%)")
ax.set_title("Quality / Compute frontier (gated expansion)")
ax.set_ylim(-25, 120)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# --- Plot 2: CTA vs prefix caching ---
ax = axes[1]
names = list(killer.keys())
pref = [killer[n][0] for n in names]
cta = [killer[n][1] for n in names]
x = range(len(names))
ax.bar([i-0.2 for i in x], pref, 0.4, label="Prefix caching", color="#adb5bd")
ax.bar([i+0.2 for i in x], cta, 0.4, label="CTA (all repeats)", color="#2a9d8f")
ax.set_xticks(list(x)); ax.set_xticklabels(names)
ax.set_ylabel("Tokens saved (sequence length)")
ax.set_title("CTA vs Prefix Caching: interior repeats")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(_bootstrap.RESULTS_FIG, "cta_results.png"), dpi=140, bbox_inches="tight")
print("saved cta_results.png")
