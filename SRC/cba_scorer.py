"""cba_scorer.py — reference scorer for Concept-Binding Accuracy (CBA).

Implements the Value-First Bipartite Hungarian Matching Protocol described
as Algorithm 1 in the paper ("Beyond Field F1: Concept-Binding Accuracy and
the Hindsight Benchmark for Document AI") and the corrected three-metric
framework of Section 3.1:

    Rec_val   = fraction of ground-truth values recovered anywhere in the
                prediction set, irrespective of concept key.
    CBA_cond  = fraction of *recovered* values bound to the correct concept
                (conditional on recovery).
    ACC_joint = Rec_val * CBA_cond, the unconditional fraction of ground
                truth items with both value and concept correct.
    Delta     = Rec_val - ACC_joint = Rec_val * (1 - CBA_cond).

This module replaces an earlier, per-domain-inconsistent scoring approach
(some notebooks computed CBA-strict unconditionally over all ground-truth
fields, others conditionally over only the recovered subset, and duplicate
predicted values were resolved by dictionary iteration order rather than by
an optimal assignment). Every score in this module is computed from a
single Hungarian assignment per document, so CBA_cond and Rec_val always
have their documented, distinct denominators, and duplicate-value ties are
resolved optimally rather than arbitrarily.

Example
-------
>>> gt = [("gross_pay", "$4,523.67"), ("net_pay", "$3,100.00")]
>>> pred = [("net_pay", "$4,523.67"), ("gross_pay", "$3,100.00")]
>>> result = score_document(gt, pred)
>>> result.rec_val, result.cba_cond, result.acc_joint, result.delta
(1.0, 0.0, 0.0, 1.0)

Also exposes Precision, Duplicate Extraction Rate, and F1_joint (Appendix F
of the paper), computed from the same matching.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

Item = tuple[str, str]  # (concept, value)


# ---------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------

def normalize_value(v: object) -> str:
    """Normalize a value for comparison: case-insensitive, whitespace-
    stripped, currency/punctuation-normalized.

    Mirrors ``hindsight_eval.metrics.normalize_value`` so that scores
    produced by this script and by the library agree; kept as a local,
    dependency-free copy so this file can be run standalone.
    """
    s = str(v).lower().strip()
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"\s+", " ", s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s


# ---------------------------------------------------------------------------
# Per-document matching (Algorithm 1)
# ---------------------------------------------------------------------------

@dataclass
class FieldMatch:
    gt_concept: str
    gt_value: str
    pred_concept: str | None
    pred_value: str | None
    cost: float  # 0.0 joint match, 0.5 misbinding, 1.0 miss

    @property
    def value_match(self) -> bool:
        return bool(self.cost < 1.0)

    @property
    def joint_match(self) -> bool:
        return bool(self.cost == 0.0)

    @property
    def is_misbinding(self) -> bool:
        return bool(self.cost == 0.5)


@dataclass
class DocumentMatch:
    fields: list[FieldMatch]
    matched_pred_indices: set[int]
    n_pred: int

    @property
    def unmatched_pred_indices(self) -> list[int]:
        return [j for j in range(self.n_pred) if j not in self.matched_pred_indices]


def match_document(gt: Sequence[Item], pred: Sequence[Item]) -> DocumentMatch:
    """Run the value-first bipartite Hungarian matching protocol for one document.

    Steps (paper Algorithm 1):
      1. Normalize all values.
      2. Build the bipartite candidate graph implicitly via the cost matrix.
      3. Cost matrix: 0.0 (value + concept match), 0.5 (value match, wrong
         concept), 1.0 (no value match).
      4. Solve the optimal one-to-one assignment with
         ``scipy.optimize.linear_sum_assignment``.
      5. Classify each ground-truth item as a joint match, a misbinding, or
         a miss.
    """
    n_gt, n_pred = len(gt), len(pred)

    if n_gt == 0:
        return DocumentMatch(fields=[], matched_pred_indices=set(), n_pred=n_pred)

    norm_gt = [normalize_value(v) for _, v in gt]
    norm_pred = [normalize_value(v) for _, v in pred]

    # scipy requires a non-empty matrix; pad with a single dummy "no match" column.
    width = max(n_pred, 1)
    costs = np.ones((n_gt, width), dtype=float)
    for i, (gt_concept, _) in enumerate(gt):
        for j, (pred_concept, _) in enumerate(pred):
            if norm_gt[i] == norm_pred[j]:
                costs[i, j] = 0.0 if pred_concept == gt_concept else 0.5

    row_ind, col_ind = linear_sum_assignment(costs)

    fields: list[FieldMatch] = []
    matched_pred_indices: set[int] = set()
    assigned = {r: c for r, c in zip(row_ind, col_ind)}

    for i, (gt_concept, gt_value) in enumerate(gt):
        j = assigned.get(i)
        if j is not None and j < n_pred and norm_gt[i] == norm_pred[j]:
            pred_concept, pred_value = pred[j]
            fields.append(FieldMatch(gt_concept, gt_value, pred_concept, pred_value, float(costs[i, j])))
            matched_pred_indices.add(j)
        else:
            fields.append(FieldMatch(gt_concept, gt_value, None, None, 1.0))

    return DocumentMatch(fields=fields, matched_pred_indices=matched_pred_indices, n_pred=n_pred)


# ---------------------------------------------------------------------------
# Corpus-level metrics
# ---------------------------------------------------------------------------

@dataclass
class CorpusScore:
    n_gt: int
    n_pred: int
    n_recovered: int
    n_joint_correct: int
    n_misbindings: int
    n_duplicates: int
    rec_val: float
    cba_cond: float
    acc_joint: float
    delta: float
    precision: float
    dup_rate: float
    f1_joint: float
    misbindings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "n_recovered": self.n_recovered,
            "n_joint_correct": self.n_joint_correct,
            "n_misbindings": self.n_misbindings,
            "n_duplicates": self.n_duplicates,
            "rec_val": round(self.rec_val, 6),
            "cba_cond": round(self.cba_cond, 6),
            "acc_joint": round(self.acc_joint, 6),
            "delta": round(self.delta, 6),
            "precision": round(self.precision, 6),
            "dup_rate": round(self.dup_rate, 6),
            "f1_joint": round(self.f1_joint, 6),
        }


def score_corpus(documents: Iterable[tuple[Sequence[Item], Sequence[Item]]]) -> CorpusScore:
    """Score a corpus of (ground_truth, predictions) document pairs.

    ``ground_truth`` and ``predictions`` are each a sequence of
    ``(concept, value)`` pairs — duplicate concepts within one side are
    allowed (e.g. a model emitting the same concept twice).

    Aggregation is additive over ground-truth items and predictions
    (i.e. metrics are corpus-level, matching the paper's $GT=\\{(c_i,v_i)\\}_{i=1}^N$
    summed over the corpus), not a per-document macro-average.
    """
    n_gt = n_pred = n_recovered = n_joint_correct = n_misbindings = n_duplicates = 0
    misbindings: list[dict] = []

    for gt, pred in documents:
        gt, pred = list(gt), list(pred)
        n_gt += len(gt)
        n_pred += len(pred)

        m = match_document(gt, pred)
        for f in m.fields:
            if f.joint_match:
                n_recovered += 1
                n_joint_correct += 1
            elif f.is_misbinding:
                n_recovered += 1
                n_misbindings += 1
                misbindings.append({
                    "gt_concept": f.gt_concept,
                    "pred_concept": f.pred_concept,
                    "value": f.gt_value,
                })

        # Duplicate extraction rate: predictions whose normalized value was
        # never matched because a lower-cost claim on the same value already
        # exists (i.e. a repeated value the model emitted under >1 concept key).
        norm_pred_vals = [normalize_value(v) for _, v in pred]
        norm_matched_vals = {
            normalize_value(f.gt_value) for f in m.fields if f.value_match
        }
        for j in m.unmatched_pred_indices:
            if norm_pred_vals[j] in norm_matched_vals:
                n_duplicates += 1

    rec_val = n_recovered / n_gt if n_gt else 0.0
    cba_cond = n_joint_correct / n_recovered if n_recovered else 0.0
    acc_joint = rec_val * cba_cond
    delta = rec_val - acc_joint
    precision = n_joint_correct / n_pred if n_pred else 0.0
    dup_rate = n_duplicates / n_pred if n_pred else 0.0
    f1_joint = (
        2 * acc_joint * precision / (acc_joint + precision)
        if (acc_joint + precision) > 0
        else 0.0
    )

    return CorpusScore(
        n_gt=n_gt,
        n_pred=n_pred,
        n_recovered=n_recovered,
        n_joint_correct=n_joint_correct,
        n_misbindings=n_misbindings,
        n_duplicates=n_duplicates,
        rec_val=rec_val,
        cba_cond=cba_cond,
        acc_joint=acc_joint,
        delta=delta,
        precision=precision,
        dup_rate=dup_rate,
        f1_joint=f1_joint,
        misbindings=misbindings,
    )


def score_document(gt: Sequence[Item], pred: Sequence[Item]) -> CorpusScore:
    """Convenience wrapper: score a single document as a one-document corpus."""
    return score_corpus([(gt, pred)])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_items(obj) -> list[Item]:
    """Accept either a dict {concept: value} or a list of [concept, value] pairs."""
    if isinstance(obj, dict):
        return list(obj.items())
    return [(c, v) for c, v in obj]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Score a corpus of documents with the Value-First Bipartite "
            "Hungarian Matching Protocol. Input is a JSON file: a list of "
            "{\"ground_truth\": {...}, \"predictions\": {...}} objects."
        )
    )
    parser.add_argument("corpus_json", help="Path to a JSON file of documents")
    args = parser.parse_args()

    with open(args.corpus_json) as f:
        docs = json.load(f)

    documents = [
        (_load_items(d["ground_truth"]), _load_items(d["predictions"]))
        for d in docs
    ]
    result = score_corpus(documents)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
