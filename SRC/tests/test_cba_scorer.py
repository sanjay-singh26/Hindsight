"""Tests for cba_scorer.py — the Hungarian-matching reference scorer.

Covers the paper's worked example (Section 3.1) and the three-metric
framework (Rec_val, CBA_cond, ACC_joint, Delta), plus Precision / duplicate
rate / F1_joint (Appendix F).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from cba_scorer import score_document, score_corpus, match_document, normalize_value


class TestNormalizeValue:
    def test_case_and_whitespace(self):
        assert normalize_value("  Gross Pay  ") == "gross pay"

    def test_currency_and_commas(self):
        assert normalize_value("$4,523.67") == "4523.67"

    def test_trailing_zeros(self):
        assert normalize_value("100.00") == "100"
        assert normalize_value("3.50") == "3.5"


class TestWorkedExample:
    """Section 3.1 worked example: 3 W-2 fields, first two swapped."""

    def test_three_field_example(self):
        gt = [
            ("gross_pay", "$4,523.67"),
            ("net_pay", "$3,100.00"),
            ("employee_ssn", "123-45-6789"),
        ]
        pred = [
            ("net_pay", "$4,523.67"),
            ("gross_pay", "$3,100.00"),
            ("employee_ssn", "123-45-6789"),
        ]
        r = score_document(gt, pred)
        assert r.rec_val == pytest.approx(1.0)
        assert r.cba_cond == pytest.approx(1 / 3, abs=1e-6)
        assert r.acc_joint == pytest.approx(1 / 3, abs=1e-6)
        assert r.delta == pytest.approx(2 / 3, abs=1e-6)
        assert r.n_misbindings == 2


class TestThreeMetricFramework:
    def test_all_correct(self):
        gt = pred = [("a", "1"), ("b", "2")]
        r = score_document(gt, pred)
        assert r.rec_val == 1.0
        assert r.cba_cond == 1.0
        assert r.acc_joint == 1.0
        assert r.delta == 0.0

    def test_complete_miss(self):
        r = score_document([("a", "alpha")], [("b", "beta")])
        assert r.rec_val == 0.0
        assert r.cba_cond == 0.0  # no recovered items -> defined as 0.0
        assert r.acc_joint == 0.0
        assert r.delta == 0.0

    def test_partial_recovery_high_conditional_accuracy(self):
        """Rec_val < 1 but every recovered value is correctly bound:
        CBA_cond should be 1.0, legitimately exceeding Rec_val (Table 4's
        GST/Paystubs pattern), while ACC_joint stays <= Rec_val."""
        gt = [("a", "1"), ("b", "2"), ("c", "3")]
        pred = [("a", "1")]  # only "a" recovered, and correctly bound
        r = score_document(gt, pred)
        assert r.rec_val == pytest.approx(1 / 3)
        assert r.cba_cond == pytest.approx(1.0)
        assert r.acc_joint == pytest.approx(1 / 3)
        assert r.acc_joint <= r.rec_val + 1e-9

    def test_delta_never_negative(self):
        cases = [
            ([("a", "1"), ("b", "2")], [("a", "2"), ("b", "1")]),  # full swap
            ([("a", "1")], []),  # nothing recovered
            ([("a", "1"), ("b", "1")], [("x", "1")]),  # duplicate GT value, one pred
        ]
        for gt, pred in cases:
            r = score_document(gt, pred)
            assert r.delta >= -1e-9

    def test_acc_joint_equals_recval_times_cbacond(self):
        gt = [("a", "1"), ("b", "2"), ("c", "3")]
        pred = [("a", "1"), ("x", "2"), ("c", "9")]
        r = score_document(gt, pred)
        assert r.acc_joint == pytest.approx(r.rec_val * r.cba_cond)


class TestOptimalMatchingResolvesDuplicates:
    def test_prefers_misbinding_over_unrelated_miss(self):
        """Regression: value match under wrong concept must not be treated
        as a complete miss just because another prediction shares no value."""
        gt = [("a", "1"), ("b", "2")]
        pred = [("x", "1"), ("y", "99")]
        m = match_document(gt, pred)
        by_concept = {f.gt_concept: f for f in m.fields}
        assert by_concept["a"].value_match is True
        assert by_concept["a"].is_misbinding is True
        assert by_concept["b"].value_match is False

    def test_duplicate_value_optimal_assignment(self):
        """Two GT fields share a value; predictions can satisfy only one
        correctly. The optimal assignment must not leave a joint match
        unclaimed in favor of a lower-value pairing."""
        gt = [("a", "1"), ("b", "1")]
        pred = [("b", "1"), ("a", "1")]  # order shouldn't matter
        r = score_document(gt, pred)
        assert r.rec_val == 1.0
        assert r.cba_cond == 1.0  # optimal assignment finds both joint matches


class TestPrecisionDuplicatesF1Joint:
    def test_precision_penalizes_overgeneration(self):
        gt = [("a", "1")]
        pred = [("a", "1"), ("b", "2"), ("c", "3")]  # 2 spurious predictions
        r = score_document(gt, pred)
        assert r.rec_val == 1.0
        assert r.precision == pytest.approx(1 / 3)

    def test_duplicate_extraction_rate(self):
        """Model emits the same value under two different concept keys."""
        gt = [("a", "1")]
        pred = [("a", "1"), ("b", "1")]  # "b" duplicates a's already-claimed value
        r = score_document(gt, pred)
        assert r.n_duplicates == 1
        assert r.dup_rate == pytest.approx(0.5)

    def test_f1_joint_is_harmonic_mean(self):
        gt = [("a", "1"), ("b", "2")]
        pred = [("a", "1"), ("b", "9"), ("c", "3")]
        r = score_document(gt, pred)
        expected = (
            2 * r.acc_joint * r.precision / (r.acc_joint + r.precision)
            if (r.acc_joint + r.precision) > 0
            else 0.0
        )
        assert r.f1_joint == pytest.approx(expected)


class TestCorpusAggregation:
    def test_additive_over_documents(self):
        docs = [
            ([("a", "1")], [("a", "1")]),
            ([("a", "1"), ("b", "2")], [("a", "9"), ("b", "2")]),
        ]
        r = score_corpus(docs)
        assert r.n_gt == 3
        assert r.n_recovered == 2
        assert r.n_joint_correct == 2
