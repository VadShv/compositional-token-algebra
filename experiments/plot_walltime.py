import json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401  (sets sys.path + result dirs)
models = json.load(open(os.path.join(_bootstrap.RESULTS_RAW, "results_walltime.json")))
lens = json.load(open(os.path.join(_bootstrap.RESULTS_RAW, "results_walltime_len.json")))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
teal, orange, gray = "#2a9d8f", "#e76f51", "#8a8a8a"

# --- Plot 1: wall-clock time by model size (baseline vs CTA) ---
ax = axes[0]
labels = [f"{m['model'].replace('gpt2','GPT-2').replace('-',' ')}\n({m['params_M']}M)" for m in models]
x = range(len(models))
base = [m["baseline_ms"] for m in models]
cta = [m["cta_ms"] for m in models]
w = 0.38
ax.bar([i - w/2 for i in x], base, w, label="Baseline", color=gray)
ax.bar([i + w/2 for i in x], cta, w, label="CTA", color=teal)
for i, m in enumerate(models):
    ax.text(i, max(base[i], cta[i]) + 60, f"{m['speedup']:.2f}x",
            ha="center", fontsize=11, fontweight="bold", color="#1d3557")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Forward pass wall-clock (ms, CPU)")
ax.set_title("Measured wall-clock at fixed length (512 tok)\nspeedup ~1.5x, roughly flat vs model size")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

# --- Plot 2: measured speedup vs sequence length, gpt2 ---
ax = axes[1]
Ls = [r["len"] for r in lens]
real = [r["speedup"] for r in lens]
ax.plot(Ls, real, "-o", color=teal, lw=2.4, ms=7, label="Measured wall-clock speedup")
for lx, ly in zip(Ls, real):
    ax.annotate(f"{ly:.2f}x", (lx, ly), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, fontweight="bold", color="#1d3557")
ax.axhline(1.0, color="gray", ls=":", lw=1, label="no speedup")
ax.axvline(1024, color="#c00", ls="--", lw=1.2, alpha=0.7)
ax.text(1024, 1.02, " GPT-2 ctx limit", color="#c00", fontsize=8.5, ha="right", rotation=90, va="bottom")
ax.set_xlabel("Sequence length (tokens)")
ax.set_ylabel("Speedup vs baseline (x)")
ax.set_title("Measured speedup rises with LENGTH\n1.29x -> 1.71x (256 -> 1024 tok)")
ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(_bootstrap.RESULTS_FIG, "walltime_results.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
