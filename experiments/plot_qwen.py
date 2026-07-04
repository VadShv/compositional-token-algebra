"""Plot measured CTA wall-clock speedup vs sequence length on Qwen2.5-0.5B.

Companion to plot_walltime.py (GPT-2). Reads results_qwen_walltime.json and
renders results/figures/qwen_walltime.png in the same visual style as the
right-hand panel of walltime_results.png (teal line, per-point annotations,
no-speedup baseline).
"""
import json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401  (sets sys.path + result dirs)

data = json.load(open(os.path.join(_bootstrap.RESULTS_RAW, "results_qwen_walltime.json")))
rows = data["rows"]

teal, orange, gray = "#2a9d8f", "#e76f51", "#8a8a8a"

fig, ax = plt.subplots(figsize=(7.2, 5.2))

Ls = [r["len"] for r in rows]
sp = [r["speedup"] for r in rows]

ax.plot(Ls, sp, "-o", color=teal, lw=2.4, ms=7, label="Measured wall-clock speedup")
for lx, ly in zip(Ls, sp):
    ax.annotate(f"{ly:.2f}x", (lx, ly), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, fontweight="bold", color="#1d3557")
ax.axhline(1.0, color="gray", ls=":", lw=1, label="no speedup")

ax.set_xlabel("Sequence length (tokens)")
ax.set_ylabel("Speedup vs baseline (x)")
ax.set_title("Qwen2.5-0.5B: measured speedup rises with LENGTH\n"
             "1.06x -> 1.58x (256 -> 2048 tok); reversibility 3.73e-9")
ax.set_xticks(Ls)
ax.set_ylim(0.95, max(sp) + 0.18)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(_bootstrap.RESULTS_FIG, "qwen_walltime.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
