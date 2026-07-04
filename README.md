# Compositional Token Algebra (CTA)

> Reversible, session-scoped, parameter-free token compression for compute
> amortization in frozen transformer LMs — with a learnable per-span gate that
> traces the quality/compute frontier.

**Reference implementation and full reproduction package for the CTA preprint.**
📄 **Preprint (PDF):** [`paper/cta.pdf`](paper/cta.pdf)

Русская версия — ниже, в конце README ([перейти к русскому описанию](#русская-версия)).

---

## What is CTA?

Long transformer contexts repeatedly re-encode the *same* spans of tokens —
identical import blocks, boilerplate functions, repeated system prompts,
templated ticket fields. Every repeat pays the full \(O(L^2)\) attention cost
again, even though it carries no new information.

**Compositional Token Algebra (CTA)** treats the token stream as an algebra.
Within a single context (session) it:

1. **Detects** repeated spans with a rolling hash (parameter-free).
2. **Composes** each repeated span into a single *composite embedding*
   \(\bar{e}\) — a function of the original token embeddings, adding **no new
   parameters**.
3. Keeps the composition **losslessly reversible**: `decompose(compose(x)) == x`
   to float32 machine precision, so nothing is thrown away.
4. **Gates** which spans to expand back to full tokens vs. keep pooled, trading
   perplexity for attention FLOPs.

CTA is, to our knowledge, the only method that is simultaneously **reversible**,
**session-scoped**, **parameter-free**, and a **compute-amortization** technique
(rather than lossy compression or a learned-tokenizer replacement). See the
Related Work section of the preprint for the full comparison against ASG,
R3Mem, H-Net, Dynamic Token Pooling, ToMe, and BLT.

### Two gating strategies

- **Residual-energy heuristic** (parameter-free): expand the spans whose interior
  tokens deviate most from their centroid — i.e. the ones that lose the most
  information when pooled.
- **Learnable gate (`GateMLP`)**: a tiny MLP
  (`Linear(2d+2 → 128) → GELU → Linear(128 → 1)`, ~200K params) that predicts a
  per-span expand score from the query vector \(q\), the span mean embedding
  \(\bar{e}\), and two span statistics. Trained with a differentiable
  **Gumbel-Sigmoid** relaxation and a **CE + sparsity (FLOPs-budget)** loss. The
  backbone GPT-2 stays completely frozen.

**Headline result:** the learnable gate **strictly dominates** the heuristic at
*every* compute budget on both the code and ticket benchmarks.

---

## Key results

All numbers are produced with a **frozen GPT-2 small** (124M, 12 layers,
\(d=768\)) on repetition-heavy corpora. Attention FLOPs use the proxy
\(n_\text{layer}\cdot 2\cdot L^2\cdot d\).

### 1. Reversibility (lossless)

The compose/decompose operators are exactly invertible: **max reconstruction
error `4.77e-7`** (float32 machine epsilon) across all scoring modes. CTA never
discards information — expansion is always available.

### 2. Naive opaque collapse is *not* enough

Collapsing repeated spans into opaque composites and leaving them pooled
degrades quality sharply, and the damage grows with how much *distinguishing*
content follows each repeat:

| Sample | Tokens (→) | Attn FLOPs | PPL (base → CTA) | ΔPPL |
|--------|-----------:|-----------:|-----------------:|-----:|
| code   | 394 → 138  | 12.3%      | 33.63 → 38.95    | +15.8% |
| logs   | 232 → 40   | 3.0%       | 3.57 → 246.95    | +6810% |
| chat   | 131 → 27   | 4.2%       | 3.65 → 592.83    | +16165% |

**Diagnosis:** the damage is **content opacity**, not positional scrambling. An
anchor-position control (keeping original position ids) was *worse*, not better
(code 38.9 → 98.6), ruling out position as the cause. → `make diagnose`.

### 3. Gated expansion recovers — and beats — the baseline (heuristic)

Selectively expanding the highest-utility spans traces a quality/compute
frontier. On code, CTA becomes **better than the full baseline while using half
the attention FLOPs**:

| Sample | Budget | Attn FLOPs | ΔPPL vs baseline |
|--------|-------:|-----------:|-----------------:|
| code   | 25%    | 49.8%      | **−11.7%** (better) |
| code   | 50%    | 69.3%      | **−12.3%** (better) |
| logs   | 75%    | 89.9%      | +52.3% |
| chat   | 75%    | 86.7%      | +87.6% |

→ `make frontier`

### 4. Learnable gate strictly dominates the heuristic

At matched expansion budgets (next-token NLL, lower is better; all-pooled /
all-expanded reference: code 6.16 / 6.03, tickets 9.82 / 6.75):

**code** (n = 144 spans)

| λ (sparsity) | Expanded | Learned NLL | Heuristic NLL | Gain |
|-------------:|---------:|------------:|--------------:|-----:|
| 0.05 | 41.0% | 5.069 | 6.159 | +1.090 |
| 0.10 | 37.5% | **5.059** | 6.047 | +0.989 |
| 3.00 |  8.3% | 5.624 | 6.025 | +0.400 |

At λ=0.10 the learned gate reaches **5.059 NLL — below the all-expanded 6.03** —
while expanding only 37.5% of spans.

**tickets** (n = 80 spans)

| λ (sparsity) | Expanded | Learned NLL | Heuristic NLL | Gain |
|-------------:|---------:|------------:|--------------:|-----:|
| 1.5 | 65.0% | 6.798 | 7.823 | +1.025 |
| 3.0 | 42.5% | 7.290 | 8.517 | +1.227 |

→ `make gate`

### 5. Killer experiment: CTA vs prefix caching

Prefix caching only amortizes a *shared prefix*. CTA also compresses **interior**
repeats that prefix caching structurally cannot reach:

| Method | code | logs | chat |
|--------|-----:|-----:|-----:|
| Prefix caching (tokens saved) | 0 | 3 | 15 |
| **CTA (tokens saved)** | **256** | **192** | **104** |

→ `make killer`

### 6. Where CTA wins — the redundancy boundary

A pooling-harm diagnostic shows a **23× gap**: pooling costs **0.133 nats** on
code vs **3.063 nats** on tickets.

- **Deep redundancy** (code, boilerplate — repeats are interchangeable): CTA is a
  clear win, big FLOPs savings at *better* perplexity.
- **Shallow redundancy** (structured logs/tickets — a repeated template precedes
  distinguishing content): CTA needs aggressive expansion; savings are modest.

### 7. Wall-clock validation (measured on real hardware)

All numbers below are **measured end-to-end CPU wall-clock** (warm-up + median
of repeats, peak RSS) across a ladder of model sizes and a sweep of lengths.
These are realized speedups, not theoretical predictions. `→ make walltime`

**By model size** (fixed 512 tokens, n=512 → m=340 after collapse):

| Model | Params | Baseline | CTA | Speedup | Peak RAM |
|-------|-------:|---------:|----:|:-------:|---------:|
| GPT-2 small  | 124M | 651 ms  | 429 ms  | **1.52×** | 1241 MB |
| GPT-2 medium | 355M | 1622 ms | 1070 ms | **1.52×** | 2187 MB |
| GPT-2 large  | 774M | 3529 ms | 2480 ms | **1.42×** | 3893 MB |

Compose overhead is **< 0.3%** of CTA time (negligible). The speedup is roughly
**flat vs model size** at a fixed length — CTA does *not* get relatively faster
on bigger models. (A quadratic attention-FLOPs count predicts a larger ~2.3×
saving; the realized number is lower because MLP, `lm_head`, and embeddings are
linear in L and do not shrink — which is exactly why we report measured
wall-clock rather than a FLOPs estimate.)

**By sequence length** (GPT-2 small; where the quadratic term actually bites):

| Raw tokens | Collapsed m | Baseline | CTA | Speedup |
|-----------:|------------:|---------:|----:|:-------:|
| 256  | 193 | 319 ms  | 246 ms | 1.29× |
| 512  | 340 | 620 ms  | 438 ms | 1.42× |
| 768  | 485 | 1025 ms | 620 ms | **1.65×** |
| 1024 | 709 | 1339 ms | 784 ms | **1.71×** |

The speedup **grows with context length** as attention comes to dominate.
GPT-2's 1024-token positional limit caps the baseline sweep (raw sequences
beyond 1024 crash in `wpe`), so the largest wins live just below that ceiling.

**Bonus — context-window extension.** Because collapse is reversible and
session-scoped, CTA can fit **~1900 raw tokens into GPT-2's 1024-token window**
(raw 1900 → m 1022; the baseline overflows outright past 1024). That is an
effective **~1.9× context extension** for free on redundant inputs. This is a
genuine, useful side-effect — but an honest one: the extension factor equals the
collapse ratio, so it is **corpus-dependent**, degrades on shallow-redundancy
text, and is **not** a replacement for architectural long-context methods.

`→ make walltime` runs the full benchmark (SLOW: downloads/loads all three
backbones); `→ make plots-walltime` regenerates the figure from shipped JSON.

### Figures

![Quality/compute frontier and CTA vs prefix caching](results/figures/cta_results.png)

![Learned gate vs residual-energy heuristic frontier](results/figures/gate_frontier.png)

![Wall-clock speedup by model size and by sequence length](results/figures/walltime_results.png)

The first two figures appear in the preprint; the wall-clock figure backs the
new Wall-Clock Validation section of the preprint. Regenerate the frontier
figures with `make plots` and the wall-clock figure with `make plots-walltime`.

---

## Repository layout

```
cta-repo/
├── README.md                 # this file
├── LICENSE                   # MIT (corpus files keep their upstream licenses)
├── requirements.txt          # torch, transformers, numpy, matplotlib
├── Makefile                  # reproduction targets (see below)
├── paper/
│   ├── cta.pdf               # the preprint (10 pages, arXiv-ready)
│   ├── cta.tex               # LaTeX source
│   ├── cta_results.png       # figure 1
│   ├── gate_frontier.png     # figure 2
│   └── walltime_results.png  # figure 3 (wall-clock validation)
├── src/cta/                  # the CTA package (importable: `import cta`)
│   ├── __init__.py
│   ├── algebra.py            # compose / decompose / score_fn (reversible)
│   ├── detector.py           # rolling-hash repeated-span detection + segments
│   ├── model.py              # frozen-GPT-2 forward paths + attention_flops proxy
│   └── learned_gate.py       # GateMLP + gumbel_sigmoid
├── data/
│   ├── download_corpus.sh    # fetch the 6 GitHub code files + build tickets
│   ├── data.py               # inline SAMPLES (code/logs/chat/prose) + SYS prompt
│   ├── make_tickets.py       # synthetic Jira ticket generator (seeded)
│   └── corpus/               # populated by download_corpus.sh (code_*.py, tickets.txt)
├── experiments/
│   ├── _bootstrap.py         # path setup so every script runs from any CWD
│   ├── evaluate.py           # per-sequence CTA vs baseline PPL + FLOPs
│   ├── run_main.py           # main sweep (SLOW)   -> results_main.json
│   ├── gate.py               # heuristic frontier (SLOW) -> results_frontier.json
│   ├── build_cache.py        # per-span NLL cache (SLOW) -> cache_*.pt
│   ├── train_gate.py         # train learnable gate (FAST) -> results_gate.json
│   ├── killer.py             # CTA vs prefix caching (FAST)
│   ├── diagnose.py           # position vs content-opacity ablation
│   ├── plot.py               # -> results/figures/cta_results.png
│   ├── plot_gate.py          # -> results/figures/gate_frontier.png
│   ├── benchmark_walltime.py # real CPU wall-clock: size ladder + length sweep (SLOW)
│   └── plot_walltime.py      # -> results/figures/walltime_results.png
└── results/
    ├── raw/                  # shipped results JSON + prebuilt gate caches (.pt)
    └── figures/              # regenerated PNGs
```

---

## Installation

Requires **Python ≥ 3.10** (developed on 3.14, CPU-only — no GPU needed).

```bash
git clone <this-repo-url> cta-repo
cd cta-repo
python -m venv .venv && source .venv/bin/activate   # optional
make setup          # == pip install -r requirements.txt
```

The first run downloads GPT-2 small (~500 MB) from HuggingFace and caches it.

---

## Reproducing the results

The repository **ships the prebuilt gate-training caches**
(`results/raw/cache_code.pt`, `cache_tickets.pt`) and all result JSON files, so
the headline learned-gate result and both figures reproduce in **seconds**,
without any GPU and without re-running the expensive frozen-GPT-2 forward
passes.

### Fast path (recommended, ~1 minute, no backbone passes)

```bash
make all-fast       # = make gate + make plots
```

- `make gate` retrains the `GateMLP` from the shipped caches and writes
  `results/raw/results_gate.json`. You should see, for **code** at λ=0.10,
  `learned_nll ≈ 5.059` beating `heur_nll ≈ 6.047` (gain +0.989); for **tickets**
  at λ=1.5, `6.798` vs `7.823` (gain +1.025). Training is deterministic
  (`torch.manual_seed(0)`) — the output matches the shipped `results_gate.json`
  byte-for-byte.
- `make plots` regenerates both paper figures into `results/figures/`.

Other fast/moderate checks:

```bash
make killer         # CTA vs prefix caching (256/192/104 tokens saved)
make diagnose       # confirms damage = content opacity, not position
```

### Full path (SLOW on CPU — runs the frozen GPT-2 backbone)

To rebuild everything from raw text, including the expensive passes:

```bash
make corpus         # download the 6 GitHub code files + generate tickets.txt
make main           # CTA vs baseline sweep       -> results_main.json
make frontier       # heuristic gated frontier    -> results_frontier.json
make cache          # rebuild per-span caches     -> results/raw/cache_*.pt
make gate           # retrain gate on fresh caches -> results_gate.json
make plots          # figures
```

> **CPU note.** On a 2-vCPU machine the backbone passes are slow, so
> `build_cache.py` is capped (code: `max_chunks=12`, `max_spans_per_chunk=12`;
> tickets: `8, 10`; `CHUNK=384`). These caps reproduce the paper's cache sizes
> (code n=144, tickets n=80). Set `OMP_NUM_THREADS=2` (the Makefile does this
> for you).

### The code corpus

`make corpus` downloads six real Python source files from upstream GitHub
(requests, Flask, Click, CPython) and generates 60 seeded synthetic Jira
tickets. The code files keep their **upstream licenses** and are not
redistributed here — only the download script is. Because upstream files drift,
the byte-exact snapshot used in the paper is the copy already present under
`data/corpus/` in a release tarball.

### Compiling the preprint

The PDF is included at [`paper/cta.pdf`](paper/cta.pdf). To rebuild it you need a
LaTeX toolchain ([Tectonic](https://tectonic-typesetting.github.io/) recommended):

```bash
make paper          # cd paper && tectonic cta.tex
```

---

## Verifying the core claim yourself

The single most important claim — *the learnable gate beats the parameter-free
heuristic at every budget* — takes about a minute to check end-to-end:

```bash
make setup
make gate           # reads shipped caches, retrains gate, prints the frontier table
```

Compare the printed `learned_nll` vs `heur_nll@same` columns: the `gain` is
positive on every row, for both **code** and **tickets**.

---

## Citation

```bibtex
@misc{cta2026,
  title  = {Compositional Token Algebra: Reversible, Session-Scoped,
            Parameter-Free Token Compression with a Learnable Compute Gate},
  author = {Serzhantov, Vladimir},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Code and experiment scripts: **MIT** (see [`LICENSE`](LICENSE)). The downloadable
code corpus consists of third-party open-source files under their own upstream
licenses (Apache-2.0 / BSD / PSF) and is not redistributed under MIT.

---
---

<a name="русская-версия"></a>
# Compositional Token Algebra (CTA) — русское описание

> Обратимое, ограниченное сессией, беспараметрическое сжатие токенов для
> амортизации вычислений в замороженных трансформерах — с обучаемым
> per-span gate, который трассирует фронтир «качество / вычисления».

📄 **Препринт (PDF):** [`paper/cta.pdf`](paper/cta.pdf)

## Идея

В длинном контексте одни и те же участки токенов (импорты, шаблонный код,
повторяющиеся системные промпты, поля тикетов) кодируются заново снова и снова,
каждый раз платя полную стоимость внимания \(O(L^2)\), хотя новой информации не
несут.

**CTA** рассматривает поток токенов как алгебру. В рамках одной сессии он:

1. **Находит** повторяющиеся спаны rolling-hash-детектором (без параметров).
2. **Сворачивает (compose)** повтор в один композитный эмбеддинг \(\bar{e}\) —
   функцию исходных эмбеддингов, **без новых параметров**.
3. Делает свёртку **обратимой без потерь**: `decompose(compose(x)) == x` с
   точностью float32.
4. **Гейтит**, какие спаны разворачивать обратно в полные токены, а какие
   оставить свёрнутыми — обменивая perplexity на FLOPs внимания.

Насколько нам известно, CTA — единственный метод, который одновременно
**обратим**, **ограничен сессией**, **беспараметричен** и является техникой
**амортизации вычислений** (а не сжатием с потерями и не заменой токенизатора).

### Два способа гейтинга

- **Эвристика на остаточной энергии** (без параметров): разворачиваем спаны,
  чьи внутренние токены сильнее всего отклоняются от центроида.
- **Обучаемый gate (`GateMLP`)**: крошечный MLP
  (`Linear(2d+2 → 128) → GELU → Linear(128 → 1)`), предсказывающий score
  разворачивания по вектору запроса \(q\), среднему эмбеддингу спана \(\bar{e}\)
  и статистикам спана. Обучается через **Gumbel-Sigmoid** с лоссом
  **CE + sparsity (бюджет FLOPs)**. Backbone GPT-2 полностью заморожен.

**Главный результат:** обучаемый gate **строго доминирует** над эвристикой на
*любом* бюджете вычислений и на code, и на tickets (см. таблицы выше в
английской части и рисунки в `results/figures/`).

## Установка и воспроизведение

Нужен **Python ≥ 3.10**, только CPU, GPU не требуется.

```bash
make setup          # pip install -r requirements.txt
make all-fast       # обучение gate из готовых кэшей + перерисовка графиков (~1 мин)
```

Репозиторий уже содержит предпосчитанные кэши
(`results/raw/cache_*.pt`) и все JSON с результатами, поэтому ключевой результат
и оба рисунка воспроизводятся за секунды — без GPU и без дорогих проходов через
GPT-2. Обучение детерминировано (`torch.manual_seed(0)`), вывод совпадает с
приложенным `results_gate.json` побайтово.

Полный путь «с нуля» (медленно на CPU — реальные проходы через backbone):

```bash
make corpus         # скачать 6 файлов кода с GitHub + сгенерировать tickets.txt
make main           # полный прогон CTA vs baseline
make frontier       # эвристический фронтир
make cache          # пересобрать per-span кэши
make gate           # переобучить gate на свежих кэшах
make plots          # рисунки
```

Быстрая проверка главного тезиса:

```bash
make gate           # смотрите столбцы learned_nll vs heur_nll@same — gain > 0 везде
```

## Лицензия

Код и скрипты экспериментов — **MIT**. Скачиваемый корпус кода — сторонние
open-source файлы под их собственными лицензиями (Apache-2.0 / BSD / PSF), под
MIT не распространяется.
