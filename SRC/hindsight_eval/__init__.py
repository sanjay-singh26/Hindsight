"""hindsight_eval -- Concept-Binding Accuracy evaluation for document intelligence.

This package provides metrics to measure whether document-AI systems bind
extracted values to the *correct semantic concept*, not just whether they
extract the right text.

Quick start::

    from hindsight_eval import score_document, score_corpus, load_ontology, build_confusion_matrix

    ontology = load_ontology("ontology.yaml")
    gt = {"gross_pay": "5000.00", "net_pay": "3800.00"}
    pred = {"gross_pay": "5000.00", "net_pay": "3800.00"}
    result = score_document(gt, pred, ontology)
"""

from .metrics import score_document, score_corpus
from .io import load_ontology
from .confusion import build_confusion_matrix

__all__ = [
    "score_document",
    "score_corpus",
    "load_ontology",
    "build_confusion_matrix",
]

__version__ = "0.1.0"
