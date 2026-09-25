"""Category C6: paired document-level bootstrap confidence intervals.

Loads each domain's raw re-scored output (from the rescore_*.py scripts),
resamples documents (or bundles, for CUAD/LEDGAR) with replacement
10,000 times (random.seed(42), matching the paper's existing bootstrap
convention), and reports 95% percentile CIs for Rec_val, CBA_cond,
ACC_joint, and Delta.

Pure local computation — no API cost.

Usage:
    python3 scripts/bootstrap_ci.py --out /tmp/bootstrap_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cba_scorer import score_corpus, score_document  # noqa: E402

N_RESAMPLES = 10_000

# (result_file, kind) — kind is "bundles" (CUAD/LEDGAR, clause_num-as-value)
# or "documents" (everything else, concept:value dicts).
DOMAINS = {
    "CUAD": ("/tmp/cuad_full.json", "bundles"),
    "LEDGAR": ("/tmp/ledgar_full.json", "bundles"),
    "SROIE": ("/tmp/sroie_full.json", "documents_gt_na"),
    "CORD": ("/tmp/cord_full.json", "documents_cord"),
    "W-2": ("/tmp/w2_full.json", "documents_gt_na"),
    "VRDU Reg.": ("/tmp/vrdu_reg_full.json", "documents_raw"),
    "VRDU Ad-buy": ("/tmp/vrdu_adbuy_full.json", "documents_raw"),
    "GST": ("/tmp/gst_full.json", "documents_gst"),
    "Paystubs": ("/tmp/paystub_full.json", "documents_raw"),
}


def extract_doc_pairs(model_result: dict, kind: str) -> list:
    """Return a list of (gt_items, pred_items) pairs, one per document/bundle."""
    pairs = []
    if kind == "bundles":
        for b in model_result["bundles"]:
            gt_items = [(concept, str(cn)) for cn, concept in b["gt_mapping"].items()]
            pred_items = [(concept, str(cn)) for cn, concept in b["pred_mapping"].items()]
            pairs.append((gt_items, pred_items))
    elif kind == "documents_gt_na":
        for d in model_result["documents"]:
            gt_items = [(c, v) for c, v in d["ground_truth"].items() if v != "N/A"]
            pred_items = [(c, v) for c, v in d["predicted"].items() if v not in (None, "", "N/A")]
            pairs.append((gt_items, pred_items))
    elif kind == "documents_raw":
        for d in model_result["documents"]:
            gt_items = [(c, v) for c, v in d["ground_truth"].items()]
            pred_items = [(c, v) for c, v in d["predicted"].items() if v not in (None, "", "N/A")]
            pairs.append((gt_items, pred_items))
    elif kind == "documents_cord":
        # CORD applies Indonesian-thousands-separator normalization before
        # scoring (see rescore_cord.py normalize_cord_value); reapply it here
        # so bootstrap resamples match the original point estimate's matching.
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from rescore_cord import normalize_cord_value

        for d in model_result["documents"]:
            gt_items = [(c, normalize_cord_value(v)) for c, v in d["ground_truth"].items()]
            pred_items = [(c, normalize_cord_value(v)) for c, v in d["predicted"].items() if v not in (None, "")]
            pairs.append((gt_items, pred_items))
    elif kind == "documents_gst":
        from rescore_gst import normalize_gst_value

        for d in model_result["documents"]:
            gt_items = [(c, normalize_gst_value(v)) for c, v in d["ground_truth"].items()]
            pred_items = [(c, normalize_gst_value(v)) for c, v in d["predicted"].items() if v not in (None, "")]
            pairs.append((gt_items, pred_items))
    return pairs


def per_document_tallies(doc_pairs: list) -> list:
    """Run the Hungarian matcher once per real document and cache the counts
    needed to reconstruct Rec_val/CBA_cond/ACC_joint/Delta for any resample
    by simple summation, avoiding 10,000x redundant re-matching per domain.
    """
    tallies = []
    for gt_items, pred_items in doc_pairs:
        c = score_document(gt_items, pred_items)
        tallies.append((c.n_gt, c.n_recovered, c.n_joint_correct))
    return tallies


def bootstrap_metric(doc_pairs: list, n_resamples: int = N_RESAMPLES, seed: int = 42) -> dict:
    tallies = per_document_tallies(doc_pairs)
    rng = random.Random(seed)
    n = len(tallies)
    rec_vals, cba_conds, acc_joints, deltas = [], [], [], []

    for _ in range(n_resamples):
        n_gt = n_recovered = n_joint = 0
        for _i in range(n):
            g, r, j = tallies[rng.randrange(n)]
            n_gt += g
            n_recovered += r
            n_joint += j
        rec_val = n_recovered / n_gt if n_gt else 0.0
        cba_cond = n_joint / n_recovered if n_recovered else 0.0
        acc_joint = rec_val * cba_cond
        delta = rec_val - acc_joint
        rec_vals.append(rec_val)
        cba_conds.append(cba_cond)
        acc_joints.append(acc_joint)
        deltas.append(delta)

    def pct(values, lo=2.5, hi=97.5):
        s = sorted(values)
        lo_idx = int(len(s) * lo / 100)
        hi_idx = min(int(len(s) * hi / 100), len(s) - 1)
        return s[lo_idx], s[hi_idx]

    return {
        "rec_val_ci": pct(rec_vals),
        "cba_cond_ci": pct(cba_conds),
        "acc_joint_ci": pct(acc_joints),
        "delta_ci": pct(deltas),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/bootstrap_results.json")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    args = parser.parse_args()

    all_results = {}
    for domain, (path, kind) in DOMAINS.items():
        if not os.path.exists(path):
            print(f"SKIP {domain}: {path} not found")
            continue
        data = json.load(open(path))
        all_results[domain] = {}
        for model_name, model_result in data.items():
            doc_pairs = extract_doc_pairs(model_result, kind)
            point = model_result["scorer_result"]
            ci = bootstrap_metric(doc_pairs, n_resamples=args.n_resamples)
            all_results[domain][model_name] = {"point": point, "ci": ci}
            print(f"{domain:<14}{model_name:<12} "
                  f"Rec_val={point['rec_val']*100:5.1f}% [{ci['rec_val_ci'][0]*100:5.1f},{ci['rec_val_ci'][1]*100:5.1f}]  "
                  f"CBA_cond={point['cba_cond']*100:5.1f}% [{ci['cba_cond_ci'][0]*100:5.1f},{ci['cba_cond_ci'][1]*100:5.1f}]  "
                  f"Delta={point['delta']*100:5.1f}% [{ci['delta_ci'][0]*100:5.1f},{ci['delta_ci'][1]*100:5.1f}]")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
