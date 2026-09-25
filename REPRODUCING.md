# Reproducing the paper's results

This document maps every table and figure in the paper to the exact code and (where applicable) raw data that produced it. All commands assume `cd SRC` first, and a Python 3.9+ environment with `pip install -r requirements.txt && pip install -e .` already run.

Two reproduction paths are supported for every result below:

1. **Re-derive from saved raw data (no API key, no cost, seconds)** — every experiment's raw per-document `(ground_truth, predicted)` pairs are checked into `SRC/results/rescored/`. Running the scorer over this saved data reproduces every reported number exactly.
2. **Re-run from scratch (requires an API key, incurs API cost)** — regenerates the raw data itself by calling the models again, then scores it. Costs given below are what the original run cost the authors; re-running will incur similar cost at current provider pricing.

---

## Table 4 — Main results (all 9 domains × 2 models)

**Path 1 (no cost):**
```bash
python scripts/bootstrap_ci.py --out /tmp/bootstrap_results.json
```
This loads `results/rescored/{cuad,ledgar,sroie,cord,w2,vrdu_reg,vrdu_adbuy,gst,paystub}_full.json`, re-scores each with `cba_scorer.py`, and computes the 95% bootstrap CIs shown in Table 4's brackets. Printed point estimates should match Table 4 to the printed decimal place; CIs should match exactly (deterministic given `random.seed(42)`).

**Path 2 (regenerate raw data; needs `LLMGATEWAY_API_KEY` or equivalent env var pointing at an OpenAI-chat-completions-compatible endpoint serving `claude-haiku-4-5` and `gpt-4o-mini`):**
```bash
python scripts/rescore_cuad.py     --n-bundles 50 --out results/rescored/cuad_full.json        # ~$0.22
python scripts/rescore_ledgar.py   --n-bundles 50 --out results/rescored/ledgar_full.json       # ~$0.25
python scripts/rescore_sroie.py    --n-samples 50 --out results/rescored/sroie_full.json        # ~$0.45
python scripts/rescore_cord.py     --n-samples 93 --out results/rescored/cord_full.json         # ~$0.68 (LLM-vision track only, see note below)
python scripts/rescore_w2.py       --n-samples 50 --out results/rescored/w2_full.json           # ~$0.49
python scripts/rescore_vrdu.py --domain registration --n-samples 50 --out results/rescored/vrdu_reg_full.json    # ~$0.62
python scripts/rescore_vrdu.py --domain adbuy        --n-samples 50 --out results/rescored/vrdu_adbuy_full.json  # ~$0.68
python scripts/rescore_gst.py      --n-gt-candidates 150 --n-final-docs 100 --out results/rescored/gst_full.json # ~$1.10 (also needs claude-sonnet-4-6 access, for guided ground-truth construction)
python scripts/rescore_paystubs.py --n-samples 50 --out results/rescored/paystub_full.json      # ~$0.16 (needs a local Tesseract OCR install)
```
Total: ~$4.68 for all 9 domains. Then re-run `bootstrap_ci.py` as in Path 1 pointed at the freshly generated files.

**Known limitations of full reproduction:**
- CORD's original evaluation used two extraction tracks (Tesseract-OCR text mapping, and direct LLM vision). `rescore_cord.py` reproduces the LLM-vision track only; the Tesseract track additionally needs an Indonesian-language Tesseract data pack, which was not available in the authors' environment.
- `rescore_paystubs.py` needs a local Tesseract OCR binary (`brew install tesseract` / `apt install tesseract-ocr`) plus the `pytesseract` Python package.
- 6 of 200 paystub images in `data/paystubs/images/` are 0-byte placeholder files (a pre-existing data gap, not introduced by this release); `rescore_paystubs.py` filters these out automatically before its stratified sampling.
- `rescore_gst.py` requires access to `claude-sonnet-4-6` (or an equivalent capable model) in addition to the two evaluated models, since GST's ground truth is constructed via a guided-extraction pass with a stronger model (see paper Appendix, "GST Invoices" domain description) before the two evaluated models are tested unguided.

---

## Figure 2 — Delta ($\Delta$) bar chart across domains

Directly derived from Table 4's $\Delta$ column; no separate script. The `\Delta \geq 0` property for every bar follows automatically from `cba_scorer.py`'s definition ($\Delta = Rec_{val} - Rec_{val}\times CBA_{cond} \geq 0$ since $CBA_{cond} \in [0,1]$) — see `SRC/tests/test_cba_scorer.py::TestThreeMetricFramework::test_delta_never_negative`.

---

## Figure 3 — W-2 concept confusion matrix

Rebuilt from `results/rescored/w2_full.json`'s Haiku predictions using `cba_scorer.match_document`. A minimal reproduction:

```python
import json, sys
sys.path.insert(0, ".")
from cba_scorer import match_document

d = json.load(open("results/rescored/w2_full.json"))["haiku"]["documents"]
misbindings = []
for doc in d:
    gt_items = [(c, v) for c, v in doc["ground_truth"].items() if v != "N/A"]
    pred_items = [(c, v) for c, v in doc["predicted"].items() if v not in (None, "", "N/A")]
    m = match_document(gt_items, pred_items)
    misbindings += [f for f in m.fields if f.is_misbinding]
print(len(misbindings))  # 194, matching the paper's reported Haiku misbinding count
```
The family/category groupings used for the figure's axes are defined in `scripts/rescore_w2.py::W2_ONTOLOGY_FAMILIES` (8-category ontology-family level, used for the paper's "89% within-family" claim) and inline in the figure-generation analysis (a finer 8-way visualization split of `compensation` into `wages`/`soc_sec`/`medicare`, used for the heatmap itself and the paper's "77% within-category at this finer granularity" claim).

---

## Table (Qwen2-VL-7B-Instruct open-weights results)

**Path 1 (no cost):** raw results are in `results/rescored/qwen_{cuad,w2,sroie,vrdu_reg,vrdu_adbuy}_full.json`; each file's `scorer_result` field is the reported row.

**Path 2 (re-run locally, no API key, needs an Apple Silicon Mac or a `transformers`-based substitute — see `SRC/README.md`):**
```bash
python scripts/eval_qwen_c5.py --domain cuad             --n-samples 50 --out /tmp/qwen_cuad.json
python scripts/eval_qwen_c5.py --domain w2               --n-samples 50 --out /tmp/qwen_w2.json
python scripts/eval_qwen_c5.py --domain sroie            --n-samples 50 --out /tmp/qwen_sroie.json
python scripts/eval_qwen_c5.py --domain vrdu_registration --n-samples 50 --out /tmp/qwen_vrdu_reg.json
python scripts/eval_qwen_c5.py --domain vrdu_adbuy       --n-samples 50 --out /tmp/qwen_vrdu_adbuy.json
```
Zero API cost (all inference is local); requires downloading `mlx-community/Qwen2-VL-7B-Instruct-4bit` (~4.4GB) from Hugging Face once.

---

## Table (Prompt Format Ablation: Free JSON vs. Schema-Constrained)

**Path 1 (no cost):** the "Free JSON" condition for CUAD and Paystubs, and the "Schema-Constrained" condition for W-2, are the same files as Table 4 above (`cuad_full.json`, `paystub_full.json`, `w2_full.json` — the domains' *original* prompt formats already differ, see `SRC/README.md`). The complementary condition for each domain is in `results/rescored/cuad_schema_full.json`, `results/rescored/w2_free_full.json`, `results/rescored/paystub_schema_full.json`.

**Path 2 (re-run):**
```bash
python scripts/ablate_cuad_schema.py    --n-bundles 50 --out results/rescored/cuad_schema_full.json    # ~$0.16
python scripts/ablate_w2_free.py        --n-samples 50 --out results/rescored/w2_free_full.json        # ~$0.72
python scripts/ablate_paystub_schema.py --n-samples 50 --out results/rescored/paystub_schema_full.json # ~$0.22
```
Total: ~$1.1.

---

## Table + Figure (Controlled Density × Semantic-Overlap Factorial Study)

**Path 1 (no cost):** `results/rescored/c4_factorial/_summary.json` contains all 20 cells' scorer results; `results/rescored/c4_factorial/{overlap}_N{density}.json` contains each cell's raw per-document data (e.g. `near_synonym_N4.json`).

To recompute the paper's main-effects table and heatmap from the saved summary:
```python
import json
summary = json.load(open("results/rescored/c4_factorial/_summary.json"))
overlaps = ["unrelated", "same_type", "same_family", "near_synonym"]
densities = [4, 8, 16, 32, 44]
for overlap in overlaps:
    vals = [summary[f"{overlap}_N{n}"][m]["cba_cond"] for n in densities for m in ["haiku", "gpt4o-mini"]]
    print(overlap, round(100 * sum(vals) / len(vals), 1))
```

**Path 2 (re-run all 20 cells):**
```bash
python scripts/c4_factorial_design.py                 # inspect/validate the 4 concept orderings (no API calls)
python scripts/run_c4_factorial.py --n-samples 50 --out-dir results/rescored/c4_factorial   # ~$6.80, 2,000 calls
```
`run_c4_factorial.py` saves one file per cell incrementally, so an interrupted run can be resumed (it skips cells whose output file already exists).

---

## Appendix table (CBA-soft vs. CBA-strict, "soft-strict gap")

Not exposed as a single library call in this release; computed with a short script built on `cba_scorer.match_document` plus each domain's family map (`scripts/rescore_w2.py::W2_ONTOLOGY_FAMILIES`, and the inline `CUAD_ONTOLOGY` / `LEDGAR_ONTOLOGY` family dicts in `scripts/rescore_cuad.py` / `scripts/rescore_ledgar.py`):

```python
import json, sys
sys.path.insert(0, ".")
from cba_scorer import match_document

def cba_soft_cond(path, kind, families, model):
    c2f = {c: f for f, cs in families.items() for c in cs}
    d = json.load(open(path))[model]
    items = d["bundles"] if kind == "bundles" else d["documents"]
    n_recovered = soft_sum = strict_sum = 0
    for it in items:
        if kind == "bundles":
            gt = [(c, str(n)) for n, c in it["gt_mapping"].items()]
            pr = [(c, str(n)) for n, c in it["pred_mapping"].items()]
        else:
            gt = [(c, v) for c, v in it["ground_truth"].items() if v != "N/A"]
            pr = [(c, v) for c, v in it["predicted"].items() if v not in (None, "", "N/A")]
        for f in match_document(gt, pr).fields:
            if f.joint_match:
                n_recovered += 1; strict_sum += 1; soft_sum += 1.0
            elif f.is_misbinding:
                n_recovered += 1
                if c2f.get(f.gt_concept) == c2f.get(f.pred_concept):
                    soft_sum += 0.5
    return strict_sum / n_recovered, soft_sum / n_recovered
```
Call with `("results/rescored/w2_full.json", "documents", W2_ONTOLOGY_FAMILIES, "haiku")` etc. for the six rows in the paper's table.

---

## Appendix tables (per-family misbinding count, top confusion pairs)

Both derived from the same `match_document` walk over `results/rescored/w2_full.json` shown under "Figure 3" above; group by `f.gt_concept`'s family (per-family table) or by `(f.gt_concept, f.pred_concept)` pairs (top-confusion-pairs table) instead of collecting a flat list.

---

## Unit tests

```bash
pytest -v
```
54 tests, including a reproduction of the paper's Section 3.1 worked example (`test_cba_scorer.py::TestWorkedExample`), the three-metric identities ($ACC_{joint} = Rec_{val} \times CBA_{cond}$, $\Delta \geq 0$ always), and a regression test for the Hungarian-matcher bug fixed in this release (`test_alignment.py::test_optimal_prefers_misbinding_over_unrelated_prediction`).
