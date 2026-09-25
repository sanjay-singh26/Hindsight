"""Core CBA (Concept-Binding Accuracy) metrics for document intelligence evaluation.

This module implements three scoring modes:
  - cba_strict:   binary match — did the system bind the value to the correct concept?
  - cba_soft:     partial credit via ontology families — same family gets 0.5
  - cba_weighted: severity-weighted strict score for domain-critical fields

Scoring follows the value-first protocol from §3.2 of the paper:
  i.   For each (c_i, v_i) in ground truth, search predictions for a matching value.
  ii.  Found under correct concept → correct extraction and binding (recall=1, CBA=1).
  iii. Found under wrong concept   → correct extraction, misbinding  (recall=1, CBA=0).
  iv.  Multiple value matches      → prefer exact concept match.
  v.   No value match, concept key present → concept matched, value wrong (recall=0, CBA=0).
  vi.  No match at all             → complete miss (recall=0, CBA=0).

Previously the code matched by concept key first (concept-first), which caused the
metric to invert: misbindings were scored as CBA=1, correct bindings as CBA=0.
This version matches by normalised value first, then checks concept alignment.

Note on field_recall: this quantity was previously called field_f1, but it has no
precision term and is strictly recall over ground-truth fields.  The key is now
named ``field_recall`` throughout.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Value normalisation
# ---------------------------------------------------------------------------

def normalize_value(v: str) -> str:
    """Normalize an extracted value for comparison.

    Steps:
      1. Convert to lowercase.
      2. Strip leading/trailing whitespace.
      3. Remove dollar signs and commas (currency formatting).
      4. Normalize decimal trailing zeros (e.g. "100.00" -> "100", "3.50" -> "3.5").
      5. Collapse internal whitespace runs to a single space.

    Args:
        v: Raw extracted value string.

    Returns:
        Cleaned, comparable string.
    """
    v = str(v).lower().strip()
    v = v.replace("$", "").replace(",", "")
    v = re.sub(r"\s+", " ", v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        v = v.rstrip("0").rstrip(".")
    return v


# ---------------------------------------------------------------------------
# Concept-level scoring functions
# ---------------------------------------------------------------------------

def cba_strict(predicted_concept: str, ground_truth_concept: str) -> float:
    """Binary concept-binding accuracy.

    Args:
        predicted_concept: Concept ID the system assigned.
        ground_truth_concept: Correct concept ID from the ground truth.

    Returns:
        1.0 if the concepts match exactly, 0.0 otherwise.
    """
    return 1.0 if predicted_concept == ground_truth_concept else 0.0


def cba_soft(predicted_concept: str, ground_truth_concept: str, ontology: dict) -> float:
    """Partial-credit concept-binding accuracy using ontology families.

    Scoring:
      - 1.0 if the predicted concept equals the ground-truth concept.
      - 0.5 if they belong to the same semantic family.
      - 0.0 otherwise (different family or unknown concept).

    Args:
        predicted_concept: Concept ID the system assigned.
        ground_truth_concept: Correct concept ID.
        ontology: Loaded ontology dict (must contain ``concept_to_family`` lookup).

    Returns:
        Float score in {0.0, 0.5, 1.0}.
    """
    if predicted_concept == ground_truth_concept:
        return 1.0
    c2f = ontology.get("concept_to_family", {})
    gt_family = c2f.get(ground_truth_concept)
    pred_family = c2f.get(predicted_concept)
    if gt_family is not None and gt_family == pred_family:
        return 0.5
    return 0.0


def cba_weighted(
    predicted_concept: str,
    ground_truth_concept: str,
    severity_weights: dict,
) -> float:
    """Severity-weighted concept-binding accuracy.

    Multiplies the strict CBA score by a domain-defined severity weight for
    the ground-truth concept.  Higher weights mark fields where misbinding
    is more costly (e.g. ``total_tax`` vs ``memo``).

    Args:
        predicted_concept: Concept ID the system assigned.
        ground_truth_concept: Correct concept ID.
        severity_weights: Mapping of concept_id -> float weight.

    Returns:
        ``cba_strict * weight`` for the ground-truth concept.
        Weight defaults to 1.0 when the concept is absent from the map.
    """
    strict = cba_strict(predicted_concept, ground_truth_concept)
    weight = severity_weights.get(ground_truth_concept, 1.0)
    return strict * weight


# ---------------------------------------------------------------------------
# Field-level scoring
# ---------------------------------------------------------------------------

def score_field(
    gt_value: str,
    pred_value: str,
    gt_concept: str,
    pred_concept: str,
    ontology: dict,
    severity_weights: Optional[dict] = None,
) -> dict:
    """Score a single field extraction under the value-first protocol.

    A *misbinding* is detected when the extracted value matches the ground
    truth but the concept label is wrong.  CBA scores are only non-zero when
    the value is correctly extracted; a concept match with a wrong value still
    yields CBA=0 (protocol step v).

    Args:
        gt_value: Ground-truth value string.
        pred_value: Predicted value string.
        gt_concept: Ground-truth concept ID.
        pred_concept: Predicted concept ID.
        ontology: Loaded ontology dict with ``concept_to_family``.
        severity_weights: Optional concept -> weight mapping.

    Returns:
        Dict containing:
          - value_match (bool)
          - field_recall_contribution (float, 1.0 or 0.0)
          - cba_strict (float)
          - cba_soft (float)
          - cba_weighted (float or None)
          - is_misbinding (bool)
          - gt_concept, pred_concept, gt_value, pred_value
    """
    norm_gt = normalize_value(gt_value)
    norm_pred = normalize_value(pred_value)
    value_match = norm_gt == norm_pred

    # CBA is only meaningful when the value was correctly extracted (steps ii–iv).
    # A concept match with a wrong value must yield CBA=0 (step v).
    if value_match:
        strict = cba_strict(pred_concept, gt_concept)
        soft = cba_soft(pred_concept, gt_concept, ontology)
        weighted = (
            cba_weighted(pred_concept, gt_concept, severity_weights)
            if severity_weights is not None
            else None
        )
    else:
        strict = 0.0
        soft = 0.0
        weighted = 0.0 if severity_weights is not None else None

    return {
        "value_match": value_match,
        "field_recall_contribution": 1.0 if value_match else 0.0,
        "cba_strict": strict,
        "cba_soft": soft,
        "cba_weighted": weighted,
        "is_misbinding": value_match and strict == 0.0,
        "gt_concept": gt_concept,
        "pred_concept": pred_concept,
        "gt_value": gt_value,
        "pred_value": pred_value,
    }


# ---------------------------------------------------------------------------
# Document-level scoring
# ---------------------------------------------------------------------------

def score_document(
    ground_truth: dict,
    predictions: dict,
    ontology: dict,
    severity_weights: Optional[dict] = None,
) -> dict:
    """Score all field extractions for a single document (value-first protocol).

    For every ground-truth (concept, value) pair the function applies the
    six-step value-first protocol from §3.2:

      i.   Search all unused predictions for a normalised-value match.
      ii.  If exactly correct: recall=1, CBA=1.
      iii. If value found under wrong concept: recall=1, CBA=0, misbinding.
      iv.  If multiple value matches, prefer the prediction whose concept matches.
      v.   If no value match but concept key present: recall=0, CBA=0 (wrong value).
      vi.  Otherwise: complete miss, recall=0, CBA=0.

    Fabricated concept pairs (old Strategy 3) are no longer created; an
    unmatched ground-truth field is simply a miss.

    Args:
        ground_truth: Mapping concept_id -> value (the gold standard).
        predictions: Mapping concept_id -> value (system output).
        ontology: Loaded ontology dict with family lookups.
        severity_weights: Optional concept -> weight mapping.

    Returns:
        Dict with aggregate scores and per-field detail.
    """
    per_field: list[dict] = []
    misbindings: list[dict] = []
    used_pred_keys: set = set()

    for gt_concept, gt_value in ground_truth.items():
        norm_gt = normalize_value(gt_value)

        # Steps (i) + (iv): collect all predictions whose value matches; prefer
        # the one that also matches the concept.
        value_matches = [
            pk for pk, pv in predictions.items()
            if pk not in used_pred_keys and normalize_value(pv) == norm_gt
        ]

        if value_matches:
            # Step (iv): prefer exact concept match when there are multiple candidates
            pred_concept = (
                gt_concept if gt_concept in value_matches else value_matches[0]
            )
            pred_value = predictions[pred_concept]
            used_pred_keys.add(pred_concept)

        elif gt_concept in predictions and gt_concept not in used_pred_keys:
            # Step (v): concept key present but value does not match
            pred_concept = gt_concept
            pred_value = predictions[gt_concept]
            used_pred_keys.add(gt_concept)

        else:
            # Step (vi): complete miss
            pred_concept = "__MISSING__"
            pred_value = ""

        field_score = score_field(
            gt_value, pred_value, gt_concept, pred_concept, ontology, severity_weights
        )
        per_field.append(field_score)

        if field_score["is_misbinding"]:
            misbindings.append(
                {
                    "gt_concept": gt_concept,
                    "pred_concept": pred_concept,
                    "value": gt_value,
                }
            )

    n = len(per_field)
    if n == 0:
        return {
            "field_recall": 0.0,
            "cba_strict": 0.0,
            "cba_soft": 0.0,
            "cba_weighted": None,
            "delta": 0.0,
            "n_fields": 0,
            "n_misbindings": 0,
            "misbindings": [],
            "per_field": [],
        }

    field_recall = sum(f["field_recall_contribution"] for f in per_field) / n
    cba_s = sum(f["cba_strict"] for f in per_field) / n
    cba_so = sum(f["cba_soft"] for f in per_field) / n

    if severity_weights is not None:
        total_weight = sum(
            severity_weights.get(f["gt_concept"], 1.0) for f in per_field
        )
        cba_w = (
            sum(f["cba_weighted"] for f in per_field) / total_weight
            if total_weight > 0
            else 0.0
        )
    else:
        cba_w = None

    return {
        "field_recall": round(field_recall, 6),
        "cba_strict": round(cba_s, 6),
        "cba_soft": round(cba_so, 6),
        "cba_weighted": round(cba_w, 6) if cba_w is not None else None,
        "delta": round(field_recall - cba_s, 6),
        "n_fields": n,
        "n_misbindings": len(misbindings),
        "misbindings": misbindings,
        "per_field": per_field,
    }


# ---------------------------------------------------------------------------
# Corpus-level scoring
# ---------------------------------------------------------------------------

def score_corpus(
    documents: list,
    ontology: dict,
    severity_weights: Optional[dict] = None,
) -> dict:
    """Score multiple documents and aggregate.

    Args:
        documents: List of (ground_truth_dict, predictions_dict) tuples.
        ontology: Loaded ontology dict.
        severity_weights: Optional concept -> weight mapping.

    Returns:
        Dict with:
          - field_recall, cba_strict, cba_soft, cba_weighted, delta:
            macro-averages across documents.
          - n_documents: number of documents scored.
          - total_fields: total GT fields across all documents.
          - total_misbindings: total misbinding count.
          - per_document: list of per-document score dicts.
    """
    per_doc: list[dict] = []

    for gt, pred in documents:
        doc_score = score_document(gt, pred, ontology, severity_weights)
        per_doc.append(doc_score)

    n_docs = len(per_doc)
    if n_docs == 0:
        return {
            "field_recall": 0.0,
            "cba_strict": 0.0,
            "cba_soft": 0.0,
            "cba_weighted": None,
            "delta": 0.0,
            "n_documents": 0,
            "total_fields": 0,
            "total_misbindings": 0,
            "per_document": [],
        }

    field_recall = sum(d["field_recall"] for d in per_doc) / n_docs
    cba_s = sum(d["cba_strict"] for d in per_doc) / n_docs
    cba_so = sum(d["cba_soft"] for d in per_doc) / n_docs
    delta = sum(d["delta"] for d in per_doc) / n_docs

    if severity_weights is not None:
        vals = [d["cba_weighted"] for d in per_doc if d["cba_weighted"] is not None]
        cba_w = sum(vals) / len(vals) if vals else None
    else:
        cba_w = None

    return {
        "field_recall": round(field_recall, 6),
        "cba_strict": round(cba_s, 6),
        "cba_soft": round(cba_so, 6),
        "cba_weighted": round(cba_w, 6) if cba_w is not None else None,
        "delta": round(delta, 6),
        "n_documents": n_docs,
        "total_fields": sum(d["n_fields"] for d in per_doc),
        "total_misbindings": sum(d["n_misbindings"] for d in per_doc),
        "per_document": per_doc,
    }
