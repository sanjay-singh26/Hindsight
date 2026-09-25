"""I/O utilities for loading ontologies, predictions, and ground truth.

Supports YAML ontology files and JSON data files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import yaml


def load_ontology(path: str) -> dict:
    """Load an ontology YAML file and compute convenience lookups.

    Expected YAML structure::

        domain: "paystubs"
        families:
          - name: "earnings"
            concepts:
              - id: "gross_pay"
                label: "Gross Pay"
              - id: "net_pay"
                label: "Net Pay"
          - name: "taxes"
            concepts:
              - id: "federal_tax"
                label: "Federal Income Tax"

    Args:
        path: Path to the YAML ontology file.

    Returns:
        The parsed dict with two additional computed keys:
          - ``concept_to_family``: mapping concept_id -> family name.
          - ``all_concepts``: flat list of all concept IDs.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ontology file not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    # Build lookups
    concept_to_family: dict[str, str] = {}
    all_concepts: list[str] = []

    for family in data.get("families", []):
        family_name = family.get("name", "unknown")
        for concept in family.get("concepts", []):
            cid = concept.get("id", "")
            if cid:
                concept_to_family[cid] = family_name
                all_concepts.append(cid)

    data["concept_to_family"] = concept_to_family
    data["all_concepts"] = all_concepts

    return data


def load_predictions(path: str) -> dict:
    """Load predictions from a JSON file.

    The file should contain a JSON object mapping concept_id -> predicted value,
    or a list of such objects (one per document).

    Args:
        path: Path to the JSON predictions file.

    Returns:
        Parsed JSON content (dict or list).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth(path: str) -> dict:
    """Load ground truth from a JSON file.

    Same format expectations as :func:`load_predictions`.

    Args:
        path: Path to the JSON ground truth file.

    Returns:
        Parsed JSON content (dict or list).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def export_results(results: dict, path: str, format: str = "json") -> None:
    """Export evaluation results to a file.

    Supported formats:
      - ``json``: Pretty-printed JSON.
      - ``csv``:  Flattened key-value CSV (top-level scalars only;
        nested structures are JSON-serialised into the value column).

    Args:
        results: Evaluation results dict.
        path: Output file path.
        format: ``"json"`` (default) or ``"csv"``.

    Raises:
        ValueError: If the format is not supported.
    """
    fmt = format.lower()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

    elif fmt == "csv":
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for key, value in results.items():
                if isinstance(value, (dict, list)):
                    writer.writerow([key, json.dumps(value, default=str)])
                else:
                    writer.writerow([key, value])

    else:
        raise ValueError(f"Unsupported export format: {format!r}. Use 'json' or 'csv'.")
