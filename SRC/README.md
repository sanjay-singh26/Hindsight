# Hindsight

**Hindsight** is a benchmark and evaluation library for measuring *concept misbinding*: the failure mode in which a Document AI system extracts the correct string value from a document but assigns it to the wrong semantic field (e.g., reading "$4,523.67" correctly but labeling it `net_pay` instead of `gross_pay`).

The core metric is **Concept-Binding Accuracy (CBA)**, evaluated under a value-first, Hungarian-matching protocol and a three-metric framework ($Rec_{val}$, $CBA_{cond}$, $ACC_{joint}$, $\Delta$). This repository is the code and data release accompanying the paper *"Beyond Field F1: Concept-Binding Accuracy and the Hindsight Benchmark for Document AI."* See `../REPRODUCING.md` (repository root) for a table-by-table / figure-by-figure map from paper claims to the exact commands that reproduce them.

## Benchmark coverage

Nine datasets spanning five document domains, matching Table 4 of the paper:

| Domain | Dataset | Concepts | Families | Notebook |
|---|---|---|---|---|
| Payroll | Paystubs (synthetic) | 14 | 5 | `notebooks/hindsight_paystub_cba_v2.ipynb` |
| Receipts | SROIE | 4 | 3 | `notebooks/hindsight_sroie_cba.ipynb` |
| Receipts | CORD | 13 | 3 | `notebooks/hindsight_cord_cba.ipynb` |
| Tax forms | W-2 (synthetic) | 44 | 8 | `notebooks/hindsight_w2_cba.ipynb` |
| Legal contracts | LEDGAR | 15 | 6 | `notebooks/hindsight_ledgar_cba.ipynb` |
| Legal contracts | CUAD | 15 | 5 | `notebooks/hindsight_cuad_cba.ipynb` |
| Government forms | VRDU Registration | 6 | 4 | `notebooks/hindsight_vrdu_registration_cba.ipynb` |
| Government forms | VRDU Ad-buy | 9 | 4 | `notebooks/hindsight_vrdu_adbuy_cba.ipynb` |
| Invoices | GST Invoices | 13 | 4 | `notebooks/hindsight_gst_invoice_cba.ipynb` |

All nine notebooks are self-contained: data loading, API calls, scoring, and visualization in a single file. Two domains (Paystubs, W-2) are synthetic and author-generated; the remaining seven load their source data directly from the public datasets cited in the paper (SROIE, CORD, LEDGAR/LexGLUE, CUAD, VRDU, and the Roboflow GST Invoice dataset). `scripts/rescore_*.py` provide non-notebook, scriptable equivalents of the same extraction + scoring pipeline for each domain (see below).

### Ontology assets

Every domain has a standalone `data/<domain>/ontology.yaml` file (concepts, semantic families, and `confusable_with` relations), in addition to whatever ontology definition its notebook uses internally:

- `data/paystubs/ontology.yaml`, `data/w2/ontology.yaml`, `data/contracts/ontology.yaml`, `data/invoices/ontology.yaml`, `data/gst_invoices/ontology.yaml`, `data/vrdu_adbuy/ontology.yaml`, `data/vrdu_registration/ontology.yaml`
- SROIE, CORD, and LEDGAR define their ontology inline in their notebook / `scripts/rescore_*.py` module only.

Ground-truth annotations and rendered document images are checked in for the synthetic Paystubs domain under `data/paystubs/` (200 documents; 6 of the 200 image files are 0-byte placeholders, a pre-existing data gap that `scripts/rescore_paystubs.py` filters out automatically before sampling).

## Package: `hindsight_eval`

```python
from hindsight_eval import score_document, score_corpus, load_ontology, build_confusion_matrix

ontology = load_ontology("data/paystubs/ontology.yaml")
gt = {"gross_pay": "5000.00", "net_pay": "3800.00"}
pred = {"gross_pay": "5000.00", "net_pay": "3800.00"}
result = score_document(gt, pred, ontology)
```

- `metrics.py` — `cba_strict`, `cba_soft`, `cba_weighted`, and value normalization.
- `alignment.py` — value-first matching between ground truth and predictions, including a Hungarian-matching (`scipy.optimize.linear_sum_assignment`) `OptimalAssignmentAligner` for optimal duplicate-value resolution.
- `confusion.py` — per-family confusion matrix construction.
- `io.py` — ontology loading.
- `report.py` — corpus-level result aggregation.

Install locally with:

```bash
pip install -r requirements.txt
pip install -e .
```

### `cba_scorer.py` — the paper's reference scorer

The standalone reference implementation of Algorithm 1 (Value-First Bipartite Hungarian Matching Protocol) and the three-metric framework ($Rec_{val}$, $CBA_{cond}$, $ACC_{joint}$, $\Delta$), plus Precision, Duplicate Extraction Rate, and $F1_{joint}$ (Appendix F):

```bash
python cba_scorer.py path/to/corpus.json   # [{ground_truth: {...}, predictions: {...}}, ...]
```

or as a library:

```python
from cba_scorer import score_document, score_corpus

result = score_document(
    [("gross_pay", "$4,523.67"), ("net_pay", "$3,100.00")],
    [("net_pay", "$4,523.67"), ("gross_pay", "$3,100.00")],
)
result.rec_val, result.cba_cond, result.acc_joint, result.delta
```

All numbers reported in the paper's Table 4 and onward were produced with this scorer, replacing an earlier per-domain approach in which "CBA-strict"'s denominator was computed inconsistently across notebooks (some conditioned on recovered values only, others on all ground-truth fields) and duplicate predicted values were resolved by dictionary iteration order rather than an optimal assignment.

## `scripts/` — re-running the experiments

| Script | What it reproduces |
|---|---|
| `rescore_cuad.py`, `rescore_ledgar.py`, `rescore_sroie.py`, `rescore_cord.py`, `rescore_w2.py`, `rescore_vrdu.py`, `rescore_gst.py`, `rescore_paystubs.py` | Table 4 (main results): re-run extraction against Claude Haiku 4.5 and GPT-4o-mini and score with `cba_scorer.py`, for each of the 9 domains. |
| `ablate_cuad_schema.py`, `ablate_w2_free.py`, `ablate_paystub_schema.py` | Prompt-format ablation (paper Section "Prompt Format Ablation"): the free-JSON vs. schema-constrained comparison on CUAD, W-2, and Paystubs. |
| `c4_factorial_design.py`, `run_c4_factorial.py` | Controlled density × semantic-overlap factorial study on W-2 (20 cells). `c4_factorial_design.py` defines and validates the four concept orderings and can be run standalone to inspect the design; `run_c4_factorial.py` executes all 20 cells. |
| `local_qwen_client.py`, `eval_qwen_c5.py` | Open-weights model evaluation (Qwen2-VL-7B-Instruct, 4-bit) on CUAD, W-2, SROIE, VRDU Registration, VRDU Ad-buy — runs entirely locally, no API key needed. |
| `bootstrap_ci.py` | 95% bootstrap confidence intervals (10,000 paired document-level resamples) for every domain × model in Table 4. Pure local computation over already-saved results; no API calls. |

Every `rescore_*.py` / `ablate_*.py` / `run_c4_factorial.py` script expects an OpenAI-chat-completions-compatible endpoint and reads its API key and base URL from the environment:

```bash
export LLMGATEWAY_API_KEY=...            # or point at a direct OpenAI-compatible endpoint
python scripts/rescore_cuad.py --n-bundles 50 --out results/cuad_full.json
```

Each script's `--help` documents its arguments; smaller `--n-samples`/`--n-bundles` values are useful for a quick, low-cost smoke test before committing to a full run. See `../REPRODUCING.md` for the exact commands, arguments, and expected costs used to produce every number in the paper.

**Reproducing without spending API credits:** `results/rescored/` (below) already contains the raw per-document outputs from every run reported in the paper. `cba_scorer.py` and `bootstrap_ci.py` can be re-run over this saved data to independently re-derive every point estimate and confidence interval in Table 4 and the ablation tables without making any new model calls.

## `results/rescored/` — saved raw outputs

Raw per-document `(ground_truth, predicted)` pairs and computed metrics for every experiment reported in the paper:

- `{cuad,ledgar,sroie,cord,w2,vrdu_reg,vrdu_adbuy,gst,paystub}_full.json` — the main Table 4 run, one file per domain, both models.
- `cuad_schema_full.json`, `w2_free_full.json`, `paystub_schema_full.json` — the prompt-format ablation's new condition per domain (the complementary condition is the corresponding `_full.json` file above).
- `c4_factorial/*.json` — one file per (overlap, density) cell of the 20-cell factorial study, plus `_summary.json` aggregating all 20.
- `qwen_{cuad,w2,sroie,vrdu_reg,vrdu_adbuy}_full.json` — the open-weights model (Qwen2-VL-7B) evaluation.
- `bootstrap_results.json` — 95% CIs for every domain × model in Table 4.

## Repository layout

```
data/            per-domain ontologies, ground truth, and images
docs/            annotation guidelines, datasheet, domain-extension guide
hindsight_eval/  the CBA scoring library
cba_scorer.py    standalone Hungarian-matching reference scorer (Algorithm 1)
notebooks/       one self-contained evaluation notebook per dataset
scripts/         re-scoring, ablation, factorial-study, and bootstrap scripts (see table above)
results/         raw outputs backing every table/figure in the paper (see above)
tests/           unit tests for alignment, metrics, and cba_scorer
```

## Models evaluated

- Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)
- GPT-4o-mini (`gpt-4o-mini-2024-07-18`)
- Qwen2-VL-7B-Instruct (4-bit, `mlx-community/Qwen2-VL-7B-Instruct-4bit`), open-weights, evaluated locally

All proprietary-model evaluation is at `T=0` with `random.seed(42)` for sampling. See `docs/experiment_phases.md` and the paper's appendix for the full extraction protocol, rate limiting, and compute-cost breakdown. Note: the results in this release were obtained through an OpenAI-chat-completions-compatible gateway rather than the providers' native SDKs, which may report slightly different model-ID strings (e.g. `claude-haiku-4-5` without a dated snapshot suffix) than the pinned identifiers above; see the paper's Compute Resources appendix for details.

## Running the open-weights evaluation locally

`scripts/eval_qwen_c5.py` uses [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm), which requires Apple Silicon (M-series) hardware. On other platforms, an equivalent `transformers`-based loader for the same `Qwen/Qwen2-VL-7B-Instruct` weights (full-precision or a `bitsandbytes`/AWQ-quantized build) can be substituted; `local_qwen_client.py`'s `LocalQwenClient` interface (`chat.completions.create(model, messages, max_tokens)`) is the only integration point the rest of the pipeline depends on, so a compatible drop-in adapter is sufficient without changing any other script.

```bash
pip install mlx-vlm huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen2-VL-7B-Instruct-4bit', local_dir='models/Qwen2-VL-7B-Instruct-4bit')"
python scripts/eval_qwen_c5.py --domain cuad --n-samples 50 --out /tmp/qwen_cuad.json
```

## Running the test suite

```bash
pip install -e ".[dev]"
pytest
```

54 tests cover `hindsight_eval`'s alignment/metrics/confusion logic and `cba_scorer.py`'s Hungarian matching, three-metric identities, and the paper's worked example (Section 3.1).

## Adding a new domain

See `docs/adding_domains.md` for the four-step process: defining a concept ontology, sourcing/annotating a corpus, running extraction, and scoring with `hindsight_eval`.

## License

Released under the MIT license (see `LICENSE`). Third-party datasets retain their original licenses; see the paper's appendix for a full list of dataset licenses and sources.
