"""Local CPU smoke test of the LSE pipeline on the tiny 0.5B model.
Goal: catch shape/cache/position bugs, NOT to judge quality."""
import sys, json, torch
sys.path.insert(0, "src")
from cta.detector import select_collapse_spans, build_segments
from cta.algebra import compose
from cta.kv_lse import collapse_cache_lse, build_dynamic_cache, sigma_spectrum
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
print("layers", model.config.num_hidden_layers, "d", model.config.hidden_size)

# small prompt with real repetition
doc = "Auth method oauth2. Token TTL 3600s. Rate limit 1000 req/min. "
prompt = doc * 6 + "\nSummarize the config:"
ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
n = len(ids)
segs = build_segments(ids.tolist(),
        select_collapse_spans(ids.tolist(), k_min=3, k_max=16, f_min=2))
m = len(segs)
print(f"n={n} m={m} ratio={m/n:.3f}")

emb = model.model.embed_tokens(ids).unsqueeze(0)
with torch.no_grad():
    out = model.model(inputs_embeds=emb, use_cache=True)
full_past = out.past_key_values
print("full cache seq len:", full_past.get_seq_length())

new_layers = collapse_cache_lse(full_past, segs, use_quad=True, mass=True)
comp = build_dynamic_cache(new_layers)
print("compressed cache seq len:", comp.get_seq_length(), "(expected", m, ")")

# decode 8 tokens over compressed cache
first_hidden = out.last_hidden_state[:, -1, :]
next_id = int(model.lm_head(first_hidden).argmax(-1))
gen = [next_id]; cur = m
with torch.no_grad():
    for _ in range(7):
        te = model.model.embed_tokens(torch.tensor([[next_id]]))
        pos = torch.tensor([[cur]])
        o = model.model(inputs_embeds=te, past_key_values=comp,
                        use_cache=True, position_ids=pos)
        comp = o.past_key_values
        next_id = int(model.lm_head(o.last_hidden_state[:, -1, :]).argmax(-1))
        gen.append(next_id); cur += 1
print("decoded ok, gen ids:", gen)
print("gen text:", repr(tok.decode(gen)))

# sigma spectrum on first collapsed span
K0 = full_past.layers[0].keys[0]
for (s, e, is_col) in segs:
    if is_col and (e - s) >= 2:
        print("sigma_k top5 (layer0, span0):", sigma_spectrum(K0[:, s:e, :]))
        break
print("SMOKE OK")
