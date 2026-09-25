"""Concept confusion matrix construction and analysis.

Builds an NxN matrix of ground-truth vs. predicted concept IDs from scored
documents, enabling diagnosis of systematic misbindings.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional


def build_confusion_matrix(scored_documents: list, ontology: dict) -> dict:
    """Build an NxN concept confusion matrix from scored documents.

    Each ``scored_document`` is the dict returned by
    :func:`hindsight_eval.metrics.score_document` (must include ``per_field``).

    Args:
        scored_documents: List of document-level score dicts.
        ontology: Loaded ontology with ``concept_to_family`` and
            ``all_concepts`` lookups.

    Returns:
        Dict with:
          - matrix: 2D list of ints (rows = GT concept, cols = predicted concept).
          - concepts: Ordered list of concept IDs (index matches matrix axes).
          - families: Mapping concept_id -> family_name.
    """
    # Collect every concept that appears in scores
    concept_set: set[str] = set()
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for doc in scored_documents:
        for field in doc.get("per_field", []):
            gt_c = field["gt_concept"]
            pred_c = field["pred_concept"]
            concept_set.add(gt_c)
            if pred_c != "__MISSING__":
                concept_set.add(pred_c)
            pair_counts[(gt_c, pred_c)] += 1

    # Merge with all_concepts from ontology so the matrix is complete
    all_concepts = ontology.get("all_concepts", [])
    concept_set.update(all_concepts)

    # Sort for deterministic ordering
    concepts = sorted(concept_set)
    idx = {c: i for i, c in enumerate(concepts)}
    n = len(concepts)

    # Build the matrix
    matrix = [[0] * n for _ in range(n)]
    for (gt_c, pred_c), count in pair_counts.items():
        if gt_c in idx and pred_c in idx:
            matrix[idx[gt_c]][idx[pred_c]] += count

    # Build family mapping
    c2f = ontology.get("concept_to_family", {})
    families = {c: c2f.get(c, "unknown") for c in concepts}

    return {
        "matrix": matrix,
        "concepts": concepts,
        "families": families,
    }


def top_confusions(matrix_data: dict, n: int = 10) -> list:
    """Return the top *n* most frequent concept confusions (off-diagonal).

    Args:
        matrix_data: Dict returned by :func:`build_confusion_matrix`.
        n: Number of confusions to return.

    Returns:
        List of ``(gt_concept, pred_concept, count)`` tuples sorted by count
        descending.  Diagonal entries (correct bindings) are excluded.
    """
    concepts = matrix_data["concepts"]
    matrix = matrix_data["matrix"]
    confusions: list[tuple[str, str, int]] = []

    for i, gt_c in enumerate(concepts):
        for j, pred_c in enumerate(concepts):
            if i != j and matrix[i][j] > 0:
                confusions.append((gt_c, pred_c, matrix[i][j]))

    confusions.sort(key=lambda x: x[2], reverse=True)
    return confusions[:n]
