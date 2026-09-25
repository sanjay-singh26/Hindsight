# `results/` — what's current vs. historical

**Current, authoritative pipeline (produced every number in the submitted paper):** `results/rescored/` (raw per-document outputs) together with `../cba_scorer.py` and `../scripts/bootstrap_ci.py`. See the repository root's `REPRODUCING.md` for exact commands.

**Historical / superseded, kept for provenance:**

- `hindsight_phase0_results.json` through `hindsight_phase3_revised_esults.json` — outputs from earlier development phases of the benchmark and scorer, predating the corrected three-metric framework ($Rec_{val}$, $CBA_{cond}$, $ACC_{joint}$, $\Delta$) described in the paper's Section 3.1. Several of these phases used different models (e.g. Claude Sonnet/Opus in exploratory paystub runs) or the earlier, since-fixed scoring logic (documented in `../hindsight_eval/metrics.py`'s module docstring and `../cba_scorer.py`'s module docstring).
- `results.json`, `results_paper.json`, `changed_cells.json` — a snapshot-and-patch mechanism used by `../scripts/gen_table_results.py` to generate an earlier version of `Paper/tables/tab_all_results.tex`. **This pipeline is superseded and will not reproduce the paper's current Table 4** if run as-is, since it reflects the pre-correction scoring approach. `../scripts/sensitivity_analysis.py` and `sensitivity_alignment.txt` are likewise tied to that earlier pipeline.

If you are trying to reproduce a number from the paper and it does not match one of these historical files, that is expected — check `results/rescored/` and `REPRODUCING.md` instead.
