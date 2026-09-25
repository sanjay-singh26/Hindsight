"""Tests for FirstMatchAligner and OptimalAssignmentAligner.

Both aligners must agree on all cases where no two GT fields share
the same normalised value.  They may differ only in the duplicate-value
case, where OptimalAssignment chooses the max-CBA assignment.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from hindsight_eval.alignment import FirstMatchAligner, OptimalAssignmentAligner

EMPTY_ONTOLOGY = {"concept_to_family": {}}


def _score(aligner_cls, gt, pred, ontology=None):
    ont = ontology or EMPTY_ONTOLOGY
    aligner = aligner_cls()
    fields = aligner.align(gt, pred, ont)
    n = len(fields)
    if n == 0:
        return {"field_recall": 0.0, "cba_strict": 0.0, "n_misbindings": 0}
    return {
        "field_recall": sum(f["field_recall_contribution"] for f in fields) / n,
        "cba_strict":   sum(f["cba_strict"] for f in fields) / n,
        "n_misbindings": sum(1 for f in fields if f["is_misbinding"]),
    }


ALIGNERS = [FirstMatchAligner, OptimalAssignmentAligner]


class TestAgreeOnNonAmbiguousCases:
    """Both aligners must return identical results when values are unique."""

    @pytest.mark.parametrize("cls", ALIGNERS)
    def test_all_correct(self, cls):
        r = _score(cls, {"a": "1", "b": "2"}, {"a": "1", "b": "2"})
        assert r["field_recall"] == 1.0
        assert r["cba_strict"] == 1.0

    @pytest.mark.parametrize("cls", ALIGNERS)
    def test_all_swapped(self, cls):
        r = _score(cls, {"a": "1", "b": "2"}, {"b": "1", "a": "2"})
        assert r["field_recall"] == 1.0
        assert r["cba_strict"] == 0.0
        assert r["n_misbindings"] == 2

    @pytest.mark.parametrize("cls", ALIGNERS)
    def test_figure1_example(self, cls):
        """Figure 1 of the paper: two wage values swapped."""
        ontology = {
            "concept_to_family": {
                "gross_pay": "wages",
                "net_pay": "wages",
                "employee_ssn": "identifiers",
            }
        }
        gt = {"gross_pay": "$4,523.67", "net_pay": "$3,100.00", "employee_ssn": "123-45-6789"}
        pr = {"net_pay": "$4,523.67", "gross_pay": "$3,100.00", "employee_ssn": "123-45-6789"}
        r = _score(cls, gt, pr, ontology)
        assert r["field_recall"] == pytest.approx(1.0)
        assert r["cba_strict"] == pytest.approx(1 / 3, abs=1e-4)
        assert r["n_misbindings"] == 2

    @pytest.mark.parametrize("cls", ALIGNERS)
    def test_complete_miss(self, cls):
        r = _score(cls, {"a": "alpha"}, {"b": "beta"})
        assert r["field_recall"] == 0.0
        assert r["cba_strict"] == 0.0

    @pytest.mark.parametrize("cls", ALIGNERS)
    def test_concept_match_wrong_value(self, cls):
        r = _score(cls, {"a": "correct"}, {"a": "wrong"})
        assert r["field_recall"] == 0.0
        assert r["cba_strict"] == 0.0

    @pytest.mark.parametrize("cls", ALIGNERS)
    def test_empty_gt(self, cls):
        r = _score(cls, {}, {"a": "1"})
        assert r["field_recall"] == 0.0
        assert r["cba_strict"] == 0.0

    @pytest.mark.parametrize("cls", ALIGNERS)
    def test_extra_prediction_ignored(self, cls):
        r = _score(cls, {"a": "1"}, {"a": "1", "b": "99"})
        assert r["field_recall"] == 1.0
        assert r["cba_strict"] == 1.0


class TestOptimalBeatsGreedyOnDuplicateValues:
    """When two GT fields share a value, optimal and greedy can diverge.

    The optimal aligner should prefer the assignment that maximises CBA.
    """

    def test_optimal_prefers_correct_concept(self):
        """GT: {a:'1', b:'1'}, pred: {a:'1', b:'1'} — both aligners agree."""
        gt = {"a": "1", "b": "1"}
        pr = {"a": "1", "b": "1"}
        r_fm  = _score(FirstMatchAligner,         gt, pr)
        r_oa  = _score(OptimalAssignmentAligner,  gt, pr)
        # Both should give cba=1.0 since both correct concepts appear
        assert r_fm["cba_strict"]  == pytest.approx(1.0)
        assert r_oa["cba_strict"] == pytest.approx(1.0)

    def test_optimal_handles_partial_match(self):
        """GT: {a:'1', b:'1'}, pred: {a:'1', x:'1'}.
        Greedy (dict order): a→a (CBA=1), b→x (CBA=0)  → avg CBA=0.5
        Optimal: same — a→a maximises, b→x is the only remaining match.
        """
        gt = {"a": "1", "b": "1"}
        pr = {"a": "1", "x": "1"}
        r_fm = _score(FirstMatchAligner,        gt, pr)
        r_oa = _score(OptimalAssignmentAligner, gt, pr)
        assert r_fm["field_recall"]  == pytest.approx(1.0)
        assert r_oa["field_recall"] == pytest.approx(1.0)
        assert r_fm["cba_strict"]  == pytest.approx(0.5)
        assert r_oa["cba_strict"] == pytest.approx(0.5)

    def test_optimal_maximises_over_greedy_counterexample(self):
        """Construct case where greedy is suboptimal.

        GT:  {a:'1', b:'1'}
        Pred: {b:'1', a:'1'}   # dict order: b first

        If greedy picks b→a (CBA=0 mismatch), then a→b (CBA=0), total=0.
        But optimal picks a→a (CBA=1) and b→b (CBA=1), total=2.

        NOTE: Python dicts (3.7+) preserve insertion order. We control pred
        order to force the greedy sub-optimal case.
        """
        gt = {"a": "1", "b": "1"}
        # Pred dict where b is listed before a, so greedy assigns b first
        pr = {"b": "1", "a": "1"}

        fields_fm = FirstMatchAligner().align(gt, pr, EMPTY_ONTOLOGY)
        fields_oa = OptimalAssignmentAligner().align(gt, pr, EMPTY_ONTOLOGY)

        cba_fm = sum(f["cba_strict"] for f in fields_fm) / len(fields_fm)
        cba_oa = sum(f["cba_strict"] for f in fields_oa) / len(fields_oa)

        # Optimal must be at least as good as greedy
        assert cba_oa >= cba_fm - 1e-9

        # In this specific case optimal should find the perfect assignment
        assert cba_oa == pytest.approx(1.0)

    def test_optimal_prefers_misbinding_over_unrelated_prediction(self):
        """Regression test for the 2-tier weight-matrix bug (paper Algorithm 1).

        GT: {a: '1', b: '2'}
        Pred: {x: '1', y: '99'}   # x has a's value under the wrong concept

        Under the old 2-tier scheme, weight(a,x)=0 (concept mismatch) and
        weight(a,y)=0 (value mismatch) were indistinguishable, so the solver
        could arbitrarily assign a->y, hiding the misbinding as a complete
        miss instead. The 3-tier cost matrix (0.0 / 0.5 / 1.0) must always
        prefer the value-matching-but-misbound pairing (cost 0.5 < 1.0).
        """
        gt = {"a": "1", "b": "2"}
        pr = {"x": "1", "y": "99"}
        fields = OptimalAssignmentAligner().align(gt, pr, EMPTY_ONTOLOGY)
        by_concept = {f["gt_concept"]: f for f in fields}
        assert by_concept["a"]["value_match"] is True
        assert by_concept["a"]["is_misbinding"] is True
        assert by_concept["a"]["cba_strict"] == 0.0
        assert by_concept["b"]["value_match"] is False
