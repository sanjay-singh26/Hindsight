"""Human-readable reporting utilities for CBA evaluation results."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .confusion import top_confusions


def generate_report(
    corpus_scores: dict,
    confusion: dict,
    ontology: dict,
) -> str:
    """Generate a plain-text summary report of evaluation results.

    The report includes:
      - Headline metrics (Field F1, CBA-strict, CBA-soft, delta).
      - Misbinding statistics.
      - Per-family CBA breakdown.
      - Top concept confusions.

    Args:
        corpus_scores: Dict returned by :func:`~hindsight_eval.metrics.score_corpus`.
        confusion: Dict returned by :func:`~hindsight_eval.confusion.build_confusion_matrix`.
        ontology: Loaded ontology dict.

    Returns:
        Multi-line report string.
    """
    lines: list[str] = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("  HINDSIGHT CBA EVALUATION REPORT")
    lines.append(sep)
    lines.append("")

    # --- Headline metrics ---
    lines.append("HEADLINE METRICS")
    lines.append("-" * 40)
    lines.append(f"  Documents scored  : {corpus_scores['n_documents']}")
    lines.append(f"  Total fields      : {corpus_scores['total_fields']}")
    lines.append(f"  Field Recall      : {corpus_scores['field_recall']:.4f}")
    lines.append(f"  CBA-strict        : {corpus_scores['cba_strict']:.4f}")
    lines.append(f"  CBA-soft          : {corpus_scores['cba_soft']:.4f}")
    if corpus_scores.get("cba_weighted") is not None:
        lines.append(f"  CBA-weighted      : {corpus_scores['cba_weighted']:.4f}")
    lines.append(f"  Delta (F1 - CBA)  : {corpus_scores['delta']:.4f}")
    lines.append(f"  Total misbindings : {corpus_scores['total_misbindings']}")
    lines.append("")

    # --- Delta interpretation ---
    delta = corpus_scores["delta"]
    if delta > 0.05:
        lines.append(
            f"  WARNING: Delta of {delta:.4f} indicates significant concept misbinding."
        )
        lines.append(
            "  The model extracts correct text but attaches it to wrong fields."
        )
    elif delta > 0.01:
        lines.append(
            f"  NOTE: Delta of {delta:.4f} shows moderate misbinding. Review confusions below."
        )
    else:
        lines.append(
            f"  OK: Delta of {delta:.4f} -- concept bindings are well-aligned with values."
        )
    lines.append("")

    # --- Per-family breakdown ---
    family_breakdown = per_family_breakdown(corpus_scores, ontology)
    if family_breakdown:
        lines.append("PER-FAMILY CBA-STRICT BREAKDOWN")
        lines.append("-" * 40)
        for family_name, info in sorted(
            family_breakdown.items(), key=lambda x: x[1]["cba_strict"]
        ):
            lines.append(
                f"  {family_name:30s}  CBA={info['cba_strict']:.4f}  "
                f"(n={info['n_fields']})"
            )
            if info.get("top_confusion"):
                gt_c, pred_c, cnt = info["top_confusion"]
                lines.append(
                    f"    top confusion: {gt_c} -> {pred_c} ({cnt}x)"
                )
        lines.append("")

    # --- Top confusions ---
    top = top_confusions(confusion, n=10)
    if top:
        lines.append("TOP CONCEPT CONFUSIONS")
        lines.append("-" * 40)
        for gt_c, pred_c, count in top:
            gt_fam = ontology.get("concept_to_family", {}).get(gt_c, "?")
            pred_fam = ontology.get("concept_to_family", {}).get(pred_c, "?")
            same = "SAME-FAM" if gt_fam == pred_fam else "CROSS-FAM"
            lines.append(
                f"  {gt_c:25s} -> {pred_c:25s}  {count:3d}x  [{same}]"
            )
        lines.append("")

    lines.append(sep)
    lines.append("  End of report")
    lines.append(sep)

    return "\n".join(lines)


def per_family_breakdown(corpus_scores: dict, ontology: dict) -> dict:
    """Compute CBA-strict per semantic family.

    Args:
        corpus_scores: Dict from :func:`~hindsight_eval.metrics.score_corpus`.
        ontology: Loaded ontology dict with ``concept_to_family``.

    Returns:
        Dict mapping family_name -> {cba_strict, n_fields, top_confusion}.
        ``top_confusion`` is the single most frequent off-diagonal pair
        within that family (or None).
    """
    c2f = ontology.get("concept_to_family", {})

    # Accumulate per-family field scores
    family_scores: dict[str, list[float]] = defaultdict(list)
    # Track confusions within each family
    family_confusions: dict[str, dict[tuple[str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for doc in corpus_scores.get("per_document", []):
        for field in doc.get("per_field", []):
            gt_c = field["gt_concept"]
            pred_c = field["pred_concept"]
            family = c2f.get(gt_c, "unknown")
            family_scores[family].append(field["cba_strict"])
            if gt_c != pred_c:
                family_confusions[family][(gt_c, pred_c)] += 1

    result: dict[str, dict] = {}
    for family, scores in family_scores.items():
        n = len(scores)
        cba = sum(scores) / n if n > 0 else 0.0

        # Find top confusion for this family
        conf_map = family_confusions.get(family, {})
        top_conf = None
        if conf_map:
            best_pair = max(conf_map, key=conf_map.get)
            top_conf = (best_pair[0], best_pair[1], conf_map[best_pair])

        result[family] = {
            "cba_strict": round(cba, 6),
            "n_fields": n,
            "top_confusion": top_conf,
        }

    return result
