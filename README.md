# Hindsight: Concept-Binding Accuracy for Document AI

This repository is the anonymized code and data release accompanying the ICLR 2027 submission **"Beyond Field F1: Concept-Binding Accuracy and the Hindsight Benchmark for Document AI."** It contains the manuscript source, the `hindsight_eval` evaluation library, the reference scorer (`cba_scorer.py`), all nine benchmark domains' ontologies and evaluation notebooks, and every script and raw result file needed to reproduce every table and figure in the paper.

**For the fastest path from a paper claim to the exact command that reproduces it, see [`REPRODUCING.md`](REPRODUCING.md).**

## What's in here

```
Paper/    LaTeX source and compiled PDF of the manuscript
SRC/      code, data, and results (see SRC/README.md)
```

- **`Paper/`** — `AT-ICLR-Hindsight-190926.tex` (source, with revision-tracking markup: text added or corrected during the review process is shown in blue), `references.bib`, `tables/`, and the compiled `AT-ICLR-Hindsight-190926.pdf`.
- **`SRC/`** — the `hindsight_eval` package, `cba_scorer.py` (the paper's reference scorer, Algorithm 1), nine per-domain evaluation notebooks, per-domain ontologies and ground truth, and `scripts/` for re-running every experiment reported in the paper (main results, prompt-format ablation, density × overlap factorial study, open-weights model evaluation, and bootstrap confidence intervals). Full detail in [`SRC/README.md`](SRC/README.md).

## Quick start

```bash
cd SRC
pip install -r requirements.txt
pip install -e .
pytest                                    # 54 tests, no API keys needed
```

To re-derive every point estimate and confidence interval in the paper **without spending any API credits**, run the scorer over the raw per-document results already included in `SRC/results/rescored/`:

```bash
cd SRC
python scripts/bootstrap_ci.py --out /tmp/bootstrap_results.json
```

This should reproduce Table 4's point estimates and 95% CIs exactly (see `REPRODUCING.md` for a full table-by-table breakdown, including the ablation and factorial-study tables).

To re-run extraction against live models (regenerating the raw results from scratch) requires an OpenAI-chat-completions-compatible API key; see `SRC/README.md` and `REPRODUCING.md` for per-experiment commands and approximate costs (the full re-scoring + ablation + factorial study reported in the paper cost approximately $12 in API credits total, at the rates available to the authors at the time).

## Compute requirements

- **Scoring / re-deriving results from saved data**: any machine with Python 3.9+, seconds to minutes, no GPU, no API key.
- **Re-running model extraction**: no local GPU needed (all proprietary-model calls are remote API calls); network access and an API key for Claude Haiku 4.5 and GPT-4o-mini (or an OpenAI-compatible gateway routing to them).
- **Open-weights model evaluation (Qwen2-VL-7B-Instruct, 4-bit)**: runs locally; used on an Apple Silicon Mac in the paper (`mlx-vlm`), but a `transformers`-based substitute works on any platform with ~5GB free disk and enough RAM/VRAM to hold a 7B 4-bit model (roughly 5-8GB); see `SRC/README.md`.

## License

`SRC/` (code) is released under the MIT license (see `SRC/LICENSE`). Third-party datasets used by the benchmark retain their original licenses; see the paper's appendix for a full list of dataset licenses and sources.

## Anonymity

This repository has been prepared for double-blind review: it contains no author names, institutional affiliations, or other identifying information. Please do not attempt to deanonymize the authors.
