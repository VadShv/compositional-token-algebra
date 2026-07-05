"""Plot measured CTA prefill speedup vs length on GPU (Qwen2.5-3B, Tesla T4, FA/SDPA).

Reads results_gpu_qwen3b_t4.json, renders results/figures/gpu_walltime.png in the
same style as qwen_walltime.png. Shows forward-only speedup (compute ceiling) and
end-to-end speedup incl. detection+compose overhead, to prove the win is not hidden
in preprocessing.
"""
import json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
data = json.load(open(os.path.join(REPO, "results", "raw", "results_gpu_qwen3b_t4.json")))
rows = data["rows"]

teal, orange = "#2a9d8f", "#e76f51"
Ls = [r["len"] for r in rows]
fwd = [r["speedup_fwd"] for r in rows]
tot = [r["speedup_total"] for r in rows]

fig, ax = plt.subplots(figsize=(7.4, 5.2))
ax.plot(Ls, fwd, "-o", color=teal, lw=2.4, ms=7, label="Forward-only (compute ceiling)")
ax.plot(Ls, tot, "--s", color=orange, lw=2.0, ms=6,
        label="End-to-end (incl. detection + compose)")
for lx, ly in zip(Ls, fwd):
    ax.annotate(f"{ly:.2f}x", (lx, ly), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, fontweight="bold", color="#1d3557")
ax.axhline(1.0, color="gray", ls=":", lw=1, label="no speedup")

ax.set_xlabel("Prefill length (tokens)")
ax.set_ylabel("Speedup vs baseline (x)")
ax.set_title(f"{data['model']} on {data['gpu']} (FlashAttention/{data['attn_impl']})\n"
             "Prefill speedup SURVIVES FA and rises with length: 1.30x -> 1.91x")
ax.set_xticks(Ls)
ax.set_ylim(0.95, max(fwd) + 0.22)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(REPO, "results", "figures", "gpu_walltime.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
