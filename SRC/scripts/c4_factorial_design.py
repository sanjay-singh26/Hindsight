"""Category C4: controlled density x semantic-overlap factorial design for W-2.

Defines four orderings of the 44 W-2 concepts, one per "overlap" condition.
For a given density N, taking the first N concepts of an ordering gives
that ordering's N-field subset. This lets density and overlap be varied
independently while both hold |active fields| = N fixed.

- unrelated: round-robin across all 8 ontology families, maximizing
  family spread at low N (at N=8, exactly one concept per family).
- same_type: all monetary ($) fields first (spanning multiple families
  but sharing surface type), then non-monetary fields.
- same_family: whole families in descending size order, so low-to-moderate
  N stays within a single family (state_local, the largest, fills N<=14).
- near_synonym: known highly-confusable pairs/clusters first (SSN/EIN,
  state_1_*/state_2_* mirrored pairs, local/state income tax cross-pairs,
  box_12 code/value slots), so low N concentrates on the most ambiguous
  fields identified in the main W-2 re-score (Section 4.2).

Each ordering is validated to be a permutation of the same 44 concepts.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import rescore_w2 as base  # noqa: E402

ALL_CONCEPTS = base.ALL_CONCEPTS
FAMILIES = base.W2_ONTOLOGY_FAMILIES

DENSITY_LEVELS = [4, 8, 16, 32, 44]

MONETARY = {
    "wages_tips_compensation", "social_security_wages", "medicare_wages",
    "social_security_tips", "allocated_tips", "federal_tax_withheld",
    "social_security_tax", "medicare_tax", "dependent_care_benefits",
    "nonqualified_plans", "box_12a_value", "box_12b_value", "box_12c_value",
    "box_12d_value", "state_1_wages", "state_1_income_tax", "local_1_wages",
    "local_1_income_tax", "state_2_wages", "state_2_income_tax",
    "local_2_wages", "local_2_income_tax",
}


def _round_robin(families: dict) -> list:
    pools = {f: list(cs) for f, cs in families.items()}
    order = []
    while any(pools.values()):
        for fam in pools:
            if pools[fam]:
                order.append(pools[fam].pop(0))
    return order


def _unrelated_order() -> list:
    return _round_robin(FAMILIES)


def _same_type_order() -> list:
    monetary = [c for c in ALL_CONCEPTS if c in MONETARY]
    non_monetary = [c for c in ALL_CONCEPTS if c not in MONETARY]
    return monetary + non_monetary


def _same_family_order() -> list:
    fams_by_size = sorted(FAMILIES.items(), key=lambda kv: -len(kv[1]))
    order = []
    for _fam, concepts in fams_by_size:
        order.extend(concepts)
    return order


def _near_synonym_order() -> list:
    priority = [
        "employee_ssn", "employer_ein",
        "state_1", "state_2",
        "state_1_wages", "state_2_wages",
        "state_1_income_tax", "state_2_income_tax",
        "local_1_income_tax", "local_2_income_tax",
        "social_security_wages", "social_security_tips",
        "medicare_wages", "allocated_tips",
        "box_12a_code", "box_12b_code",
        "box_12a_value", "box_12b_value",
        "box_12c_code", "box_12d_code",
        "box_12c_value", "box_12d_value",
        "local_1_wages", "local_2_wages",
        "state_1_employer_id", "state_2_employer_id",
        "local_1_name", "local_2_name",
        "federal_tax_withheld", "social_security_tax", "medicare_tax",
        "wages_tips_compensation",
        "dependent_care_benefits", "nonqualified_plans",
        "statutory_employee", "retirement_plan", "third_party_sick_pay",
        "employer_name", "employee_name",
        "employer_address", "employee_address",
        "employer_city_state_zip", "employee_city_state_zip",
        "control_number",
    ]
    assert set(priority) == set(ALL_CONCEPTS), (
        set(ALL_CONCEPTS) - set(priority), set(priority) - set(ALL_CONCEPTS)
    )
    return priority


OVERLAP_ORDERINGS = {
    "unrelated": _unrelated_order(),
    "same_type": _same_type_order(),
    "same_family": _same_family_order(),
    "near_synonym": _near_synonym_order(),
}

for name, order in OVERLAP_ORDERINGS.items():
    assert len(order) == 44, f"{name}: expected 44, got {len(order)}"
    assert set(order) == set(ALL_CONCEPTS), f"{name}: not a permutation of ALL_CONCEPTS"


def active_fields(overlap: str, density: int) -> list:
    """The N active concepts for a given (overlap, density) cell."""
    return OVERLAP_ORDERINGS[overlap][:density]


if __name__ == "__main__":
    for overlap in OVERLAP_ORDERINGS:
        print(f"=== {overlap} ===")
        for n in DENSITY_LEVELS:
            fields = active_fields(overlap, n)
            fams = sorted({base.concept_to_family(c) if hasattr(base, "concept_to_family") else None for c in fields}) if False else None
            print(f"  N={n:2d}: {fields[:6]}{'...' if n > 6 else ''}")
