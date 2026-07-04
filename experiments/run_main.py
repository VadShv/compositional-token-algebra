"""Main experiment: CTA vs baseline across repetition-heavy samples and scoring modes."""
import warnings, os, json
warnings.filterwarnings("ignore")
import _bootstrap  # noqa: F401  (sets sys.path + result dirs)
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from data import SAMPLES
from evaluate import evaluate_sequence

torch.manual_seed(0)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

MODES = ["uniform", "norm", "posdecay", "selfconsist"]
results = {}

for name, text in SAMPLES.items():
    ids = torch.tensor(tok(text)["input_ids"])
    per_mode = {}
    for mode in MODES:
        r = evaluate_sequence(model, ids, k_min=3, k_max=16, f_min=2, score_mode=mode)
        per_mode[mode] = r
    results[name] = per_mode

# Print table
print(f"{'sample':7s} {'mode':12s} {'n':>4s} {'m':>4s} {'len%':>6s} {'flops%':>7s} "
      f"{'base_ppl':>9s} {'cta_ppl':>9s} {'d_ppl%':>7s} {'lossless':>9s} {'tgts':>5s}")
print("-" * 92)
for name, per_mode in results.items():
    for mode, r in per_mode.items():
        if r is None:
            print(f"{name:7s} {mode:12s}  -- no repeated spans detected --")
            continue
        dppl = 100 * (r["cta_ppl"] - r["base_ppl"]) / r["base_ppl"]
        print(f"{name:7s} {mode:12s} {r['n']:>4d} {r['m']:>4d} "
              f"{100*r['len_ratio']:>5.1f}% {100*r['flops_ratio']:>6.1f}% "
              f"{r['base_ppl']:>9.3f} {r['cta_ppl']:>9.3f} {dppl:>+6.1f}% "
              f"{r['lossless_err']:>9.1e} {r['n_targets']:>5d}")

with open(os.path.join(_bootstrap.RESULTS_RAW, "results_main.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nsaved results_main.json")
