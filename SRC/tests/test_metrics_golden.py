"""Golden tests for the value-first CBA scorer.

Figure 1 example (§3.1): three W-2 fields, two misbound.
Expected under the value-first protocol:
  field_recall = 100%   (all values present in output)
  cba_strict   = 33.3%  (only employee_ssn correctly bound)
  cba_soft     = 66.7%  (two within-family get 0.5 credit)
  delta        = +66.7%
  misbindings  = 2
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from hindsight_eval.metrics import score_document, score_field, normalize_value

# Shared ontology for Figure 1 example
ONTOLOGY = {
    "concept_to_family": {
        "gross_pay":    "wages",
        "net_pay":      "wages",
        "employee_ssn": "identifiers",
    }
}

GROUND_TRUTH = {
    "gross_pay":    "$4,523.67",
    "net_pay":      "$3,100.00",
    "employee_ssn": "123-45-6789",
}

# Model swapped the two wage values; SSN is correct
PREDICTIONS = {
    "net_pay":      "$4,523.67",
    "gross_pay":    "$3,100.00",
    "employee_ssn": "123-45-6789",
}


class TestGoldenFigure1:
    def setup_method(self):
        self.result = score_document(GROUND_TRUTH, PREDICTIONS, ONTOLOGY)

    def test_field_recall_is_100_percent(self):
        assert self.result["field_recall"] == pytest.approx(1.0, abs=1e-4)

    def test_cba_strict_is_33_percent(self):
        assert self.result["cba_strict"] == pytest.approx(1 / 3, abs=1e-4)

    def test_cba_soft_is_67_percent(self):
        assert self.result["cba_soft"] == pytest.approx(2 / 3, abs=1e-4)

    def test_delta_is_positive_67_percent(self):
        assert self.result["delta"] == pytest.approx(2 / 3, abs=1e-4)

    def test_misbinding_count_is_2(self):
        assert self.result["n_misbindings"] == 2


class TestProtocolSteps:
    """One test per protocol step (i)–(vi)."""

    def test_step_ii_correct_value_and_concept(self):
        """Value found under correct concept → F1=1, CBA=1, not a misbinding."""
        gt = {"field_a": "hello"}
        pr = {"field_a": "hello"}
        r = score_document(gt, pr, {"concept_to_family": {}})
        assert r["field_recall"] == 1.0
        assert r["cba_strict"] == 1.0
        assert r["n_misbindings"] == 0

    def test_step_iii_value_found_wrong_concept(self):
        """Value found under wrong concept → F1=1, CBA=0, 1 misbinding."""
        gt = {"field_a": "hello"}
        pr = {"field_b": "hello"}
        r = score_document(gt, pr, {"concept_to_family": {}})
        assert r["field_recall"] == 1.0
        assert r["cba_strict"] == 0.0
        assert r["n_misbindings"] == 1

    def test_step_iv_prefer_exact_concept_when_multiple_value_matches(self):
        """When two predictions share the GT value, prefer the one with the correct concept."""
        gt = {"field_a": "42"}
        pr = {"field_b": "42", "field_a": "42"}
        r = score_document(gt, pr, {"concept_to_family": {}})
        assert r["cba_strict"] == 1.0
        assert r["n_misbindings"] == 0

    def test_step_v_concept_match_wrong_value(self):
        """Concept key matched but value wrong → F1=0, CBA=0."""
        gt = {"field_a": "correct"}
        pr = {"field_a": "wrong"}
        r = score_document(gt, pr, {"concept_to_family": {}})
        assert r["field_recall"] == 0.0
        assert r["cba_strict"] == 0.0
        assert r["n_misbindings"] == 0

    def test_step_vi_complete_miss(self):
        """No value match, no concept match → F1=0, CBA=0."""
        gt = {"field_a": "alpha"}
        pr = {"field_b": "beta"}
        r = score_document(gt, pr, {"concept_to_family": {}})
        assert r["field_recall"] == 0.0
        assert r["cba_strict"] == 0.0
        assert r["n_misbindings"] == 0

    def test_all_correct(self):
        gt = {"a": "1", "b": "2", "c": "3"}
        pr = {"a": "1", "b": "2", "c": "3"}
        r = score_document(gt, pr, {"concept_to_family": {}})
        assert r["field_recall"] == 1.0
        assert r["cba_strict"] == 1.0
        assert r["n_misbindings"] == 0

    def test_all_swapped(self):
        gt = {"a": "1", "b": "2"}
        pr = {"b": "1", "a": "2"}
        r = score_document(gt, pr, {"concept_to_family": {}})
        assert r["field_recall"] == 1.0
        assert r["cba_strict"] == 0.0
        assert r["n_misbindings"] == 2

    def test_missing_prediction(self):
        gt = {"a": "1", "b": "2"}
        pr = {"a": "1"}
        r = score_document(gt, pr, {"concept_to_family": {}})
        assert r["field_recall"] == pytest.approx(0.5)
        assert r["cba_strict"] == pytest.approx(0.5)

    def test_extra_prediction_ignored(self):
        gt = {"a": "1"}
        pr = {"a": "1", "b": "99"}
        r = score_document(gt, pr, {"concept_to_family": {}})
        assert r["field_recall"] == 1.0
        assert r["cba_strict"] == 1.0

    def test_duplicate_identical_values_no_double_count(self):
        """Two GT fields with the same value: each should consume one prediction."""
        gt = {"a": "100", "b": "100"}
        pr = {"x": "100", "y": "100"}
        r = score_document(gt, pr, {"concept_to_family": {}})
        assert r["field_recall"] == 1.0
        assert r["n_misbindings"] == 2


class TestScoreFieldValueCondition:
    """score_field must give CBA=0 whenever the value is wrong."""

    def test_concept_match_value_wrong_gives_cba_zero(self):
        r = score_field("correct", "wrong", "field_a", "field_a",
                        {"concept_to_family": {}})
        assert r["cba_strict"] == 0.0
        assert r["cba_soft"] == 0.0
        assert r["field_recall_contribution"] == 0.0
        assert r["is_misbinding"] is False

    def test_concept_mismatch_value_correct_gives_cba_zero_and_misbinding(self):
        r = score_field("hello", "hello", "field_a", "field_b",
                        {"concept_to_family": {}})
        assert r["field_recall_contribution"] == 1.0
        assert r["cba_strict"] == 0.0
        assert r["is_misbinding"] is True


class TestNormalizeValue:
    def test_strips_currency(self):
        assert normalize_value("$1,234.50") == "1234.5"

    def test_lowercase(self):
        assert normalize_value("John DOE") == "john doe"

    def test_trailing_zeros(self):
        assert normalize_value("100.00") == "100"
        assert normalize_value("3.50") == "3.5"

    def test_collapses_whitespace(self):
        assert normalize_value("  hello   world  ") == "hello world"
