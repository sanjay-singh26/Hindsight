"""Alignment-strategy sensitivity analysis.

Compares FirstMatchAligner (greedy, current paper methodology) vs
OptimalAssignmentAligner (Hungarian algorithm) on available raw-prediction
data to verify that the choice of alignment strategy does not materially
affect reported CBA scores.

Usage:
    python3 SRC/scripts/sensitivity_analysis.py

Output: a plain-text table printed to stdout and saved to
    SRC/results/sensitivity_alignment.txt

Data source:
    SRC/results/hindsight_phase2_results.json  (paystub adversarial, 10 docs,
    4 experiment conditions: haiku_full, haiku_stripped, sonnet_full,
    sonnet_stripped).

Note on W-2 and GST:
    Raw per-document predictions from the W-2 and GST experiments were not
    preserved in any stored JSON file (only aggregate metrics were exported
    from the Colab notebooks).  Re-running those experiments would be needed
    to extend this analysis to those domains.  The paystub data used here
    contains duplicate normalised values in every document (e.g. pay_date
    coincides with pay_period_end), making it a valid stress-test of the
    alignment strategies.
"""

import json, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SRC_DIR)

from hindsight_eval.alignment import FirstMatchAligner, OptimalAssignmentAligner

RESULTS_FILE = os.path.join(SRC_DIR, "results", "hindsight_phase2_results.json")
OUTPUT_FILE  = os.path.join(SRC_DIR, "results", "sensitivity_alignment.txt")

EMPTY_ONTOLOGY = {"concept_to_family": {}}


def score_corpus_with_aligner(raw_predictions, aligner):
    doc_recalls, doc_cbas, doc_mbs = [], [], []
    for entry in raw_predictions:
        gt   = entry["ground_truth"]
        pred = entry["predicted"]
        fields = aligner.align(gt, pred, EMPTY_ONTOLOGY)
        n = len(fields)
        if n == 0:
            continue
        doc_recalls.append(sum(f["field_recall_contribution"] for f in fields) / n)
        doc_cbas.append(sum(f["cba_strict"] for f in fields) / n)
        doc_mbs.append(sum(1 for f in fields if f["is_misbinding"]))
    macro_recall = sum(doc_recalls) / len(doc_recalls) if doc_recalls else 0.0
    macro_cba    = sum(doc_cbas)    / len(doc_cbas)    if doc_cbas    else 0.0
    total_mb     = sum(doc_mbs)
    return macro_recall, macro_cba, total_mb


def main():
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    fm_aligner  = FirstMatchAligner()
    oa_aligner  = OptimalAssignmentAligner()

    lines = []
    lines.append("=" * 76)
    lines.append("  ALIGNMENT STRATEGY SENSITIVITY ANALYSIS")
    lines.append("  Data: Paystub adversarial (10 docs × 4 conditions)")
    lines.append("=" * 76)
    lines.append("")
    lines.append(
        f"{'Experiment':<22} {'Aligner':<20} "
        f"{'Recall':>8} {'CBA-strict':>12} {'Misbindings':>12}"
    )
    lines.append("-" * 76)

    for exp_name, exp in data["experiments"].items():
        raw = exp.get("raw_predictions", [])
        if not raw:
            continue
        for aligner, label in [(fm_aligner, "FirstMatch"), (oa_aligner, "OptimalAssign")]:
            recall, cba, mb = score_corpus_with_aligner(raw, aligner)
            lines.append(
                f"{exp_name:<22} {label:<20} "
                f"{recall:>8.4f} {cba:>12.4f} {mb:>12d}"
            )
        lines.append("")

    # Summary
    lines.append("=" * 76)
    lines.append("  DELTA (|FirstMatch CBA - OptimalAssign CBA|) per experiment:")
    lines.append("-" * 76)

    for exp_name, exp in data["experiments"].items():
        raw = exp.get("raw_predictions", [])
        if not raw:
            continue
        _, cba_fm, _ = score_corpus_with_aligner(raw, fm_aligner)
        _, cba_oa, _ = score_corpus_with_aligner(raw, oa_aligner)
        diff = abs(cba_fm - cba_oa)
        lines.append(f"  {exp_name:<22}  |delta| = {diff:.6f}")

    lines.append("")
    lines.append("  Interpretation: |delta| < 0.001 means strategies are equivalent")
    lines.append("  for this corpus.  Larger values indicate ordering sensitivity.")
    lines.append("=" * 76)

    report = "\n".join(lines)
    print(report)

    with open(OUTPUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
