"""Alignment strategies for matching GT fields to predicted fields.

Two strategies are provided:

- FirstMatchAligner: greedy, iterates GT fields in dictionary order and
  assigns the first value-matching unused prediction.  This matches the
  behaviour of score_document() prior to the abstraction.

- OptimalAssignmentAligner: builds a bipartite weight matrix and uses the
  Hungarian algorithm (scipy.optimize.linear_sum_assignment) to maximise
  total CBA-strict across all field assignments simultaneously.

Both strategies apply the value-first protocol: a GT field may only be
matched to a prediction whose normalised value equals the GT normalised
value.  Concept alignment is checked *after* value matching.

The aligners differ only when two or more GT fields share the same
normalised value; in that case the greedy strategy is order-dependent
while the optimal strategy picks the assignment that maximises the total
CBA score.
"""

from __future__ import annotations

from typing import Optional

from .metrics import normalize_value, score_field


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Aligner:
    """Abstract base class for field-to-field aligners."""

    def align(
        self,
        ground_truth: dict,
        predictions: dict,
        ontology: dict,
        severity_weights: Optional[dict] = None,
    ) -> list[dict]:
        """Return a list of per-field score dicts (one per GT field).

        Args:
            ground_truth: Mapping concept_id -> value (gold standard).
            predictions: Mapping concept_id -> value (system output).
            ontology: Loaded ontology dict with ``concept_to_family``.
            severity_weights: Optional concept -> weight mapping.

        Returns:
            List of dicts as returned by :func:`~hindsight_eval.metrics.score_field`.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FirstMatchAligner
# ---------------------------------------------------------------------------

class FirstMatchAligner(Aligner):
    """Greedy value-first aligner (dictionary-iteration order).

    For each GT field (in dict iteration order) the aligner searches unused
    predictions for one whose normalised value equals the GT normalised
    value.  When multiple predictions match the value it prefers the one
    whose concept key equals the GT concept (protocol step iv), then picks
    the first remaining candidate.

    This is the same logic as :func:`~hindsight_eval.metrics.score_document`
    and represents the baseline behaviour of the library.
    """

    def align(
        self,
        ground_truth: dict,
        predictions: dict,
        ontology: dict,
        severity_weights: Optional[dict] = None,
    ) -> list[dict]:
        per_field: list[dict] = []
        used: set = set()

        for gt_concept, gt_value in ground_truth.items():
            norm_gt = normalize_value(gt_value)

            value_matches = [
                pk for pk, pv in predictions.items()
                if pk not in used and normalize_value(pv) == norm_gt
            ]

            if value_matches:
                pred_concept = (
                    gt_concept if gt_concept in value_matches else value_matches[0]
                )
                pred_value = predictions[pred_concept]
                used.add(pred_concept)

            elif gt_concept in predictions and gt_concept not in used:
                pred_concept = gt_concept
                pred_value = predictions[gt_concept]
                used.add(gt_concept)

            else:
                pred_concept = "__MISSING__"
                pred_value = ""

            per_field.append(
                score_field(
                    gt_value, pred_value, gt_concept, pred_concept,
                    ontology, severity_weights,
                )
            )

        return per_field


# ---------------------------------------------------------------------------
# OptimalAssignmentAligner
# ---------------------------------------------------------------------------

class OptimalAssignmentAligner(Aligner):
    """Optimal value-first aligner using the Hungarian algorithm.

    Builds a bipartite *cost* matrix (paper Algorithm 1, "Value-First
    Bipartite Hungarian Matching Protocol") where entry (i, j) is:

      - 0.0 if GT field *i* and prediction *j* have matching normalised
        values AND matching concepts (a joint match),
      - 0.5 if the values match but the concepts differ (a misbinding),
      - 1.0 otherwise (no value match).

    Using three distinct cost tiers, rather than a single 0/1 weight, is
    the fix for a real bug in an earlier version of this aligner: with only
    two tiers, "misbinding" (value matches, concept doesn't) and "complete
    miss" (no value match at all) were indistinguishable to the solver
    (both scored 0), so it could arbitrarily pair a GT field with an
    unrelated prediction instead of the value-matching-but-misbound one,
    silently undercounting misbindings whenever a GT field had no
    perfectly-correct candidate available. The 3-tier cost matrix, minimised
    via :func:`scipy.optimize.linear_sum_assignment`, ensures the solver
    always prefers a value-only match over no match at all.

    This aligner differs from :class:`FirstMatchAligner` only when two or
    more GT fields share the same normalised value; in that case the greedy
    strategy is order-dependent, while the optimal strategy picks the
    assignment that minimises total cost (equivalently, maximises correct
    joint matches, then misbindings, then leaves the rest unmatched).
    """

    def align(
        self,
        ground_truth: dict,
        predictions: dict,
        ontology: dict,
        severity_weights: Optional[dict] = None,
    ) -> list[dict]:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise ImportError(
                "OptimalAssignmentAligner requires scipy. "
                "Install it with: pip install scipy"
            ) from exc

        import numpy as np

        gt_items = list(ground_truth.items())
        pred_items = list(predictions.items())

        n_gt = len(gt_items)
        n_pred = len(pred_items)

        if n_gt == 0:
            return []

        # Build cost matrix: 0.0 (joint match), 0.5 (misbinding), 1.0 (no value match)
        costs = np.ones((n_gt, max(n_pred, 1)), dtype=float)
        for i, (gt_concept, gt_value) in enumerate(gt_items):
            norm_gt = normalize_value(gt_value)
            for j, (pred_concept, pred_value) in enumerate(pred_items):
                if normalize_value(pred_value) == norm_gt:
                    costs[i, j] = 0.0 if pred_concept == gt_concept else 0.5

        # Solve minimum-cost bipartite matching
        row_ind, col_ind = linear_sum_assignment(costs)

        # Build assignment map: gt_index -> pred_index (or None)
        assignment: dict[int, Optional[int]] = {i: None for i in range(n_gt)}
        for r, c in zip(row_ind, col_ind):
            if n_pred == 0 or c >= n_pred:
                continue
            gt_concept, gt_value = gt_items[r]
            pred_concept, pred_value = pred_items[c]
            norm_gt = normalize_value(gt_value)
            if normalize_value(pred_value) == norm_gt:
                assignment[r] = c
            # If value doesn't match, also check the step-v case below

        # Build per-field scores
        used_pred_indices: set[int] = {c for c in assignment.values() if c is not None}
        per_field: list[dict] = []

        for i, (gt_concept, gt_value) in enumerate(gt_items):
            j = assignment[i]

            if j is not None:
                pred_concept, pred_value = pred_items[j]
            elif gt_concept in predictions:
                # Step (v): concept key present, value doesn't match
                # Only valid if that prediction index isn't already used
                pred_idx = next(
                    (idx for idx, (pk, _) in enumerate(pred_items)
                     if pk == gt_concept and idx not in used_pred_indices),
                    None,
                )
                if pred_idx is not None:
                    pred_concept, pred_value = pred_items[pred_idx]
                    used_pred_indices.add(pred_idx)
                else:
                    pred_concept = "__MISSING__"
                    pred_value = ""
            else:
                pred_concept = "__MISSING__"
                pred_value = ""

            per_field.append(
                score_field(
                    gt_value, pred_value, gt_concept, pred_concept,
                    ontology, severity_weights,
                )
            )

        return per_field
