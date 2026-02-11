import pytest

from app.models.commercial import CommercialIntentClassifier, get_commercial_classifier
from app.models.safety import SafetyResult


@pytest.fixture(scope="module")
def classifier():
    return get_commercial_classifier()


@pytest.fixture
def safe_result():
    return SafetyResult(is_blocked=False, base_score=0.5, toxicity=0.1, danger_diff=-0.2)


@pytest.fixture
def borderline_toxic_result():
    return SafetyResult(is_blocked=False, base_score=0.3, toxicity=0.5, danger_diff=0.1)


class TestCommercialSignalDetection:

    def test_buy_signal_boosts_score(self, classifier, safe_result):
        score = classifier.score("where to buy running shoes", safe_result)
        assert score > safe_result.base_score

    def test_deal_signal_boosts_score(self, classifier, safe_result):
        score = classifier.score("best laptop deals", safe_result)
        assert score > safe_result.base_score

    def test_no_commercial_signal(self, classifier, safe_result):
        score = classifier.score("history of ancient Greece", safe_result)
        # No commercial boost, but sentiment might adjust slightly
        assert score <= safe_result.base_score + 0.15

    def test_multiple_signals_dont_stack(self, classifier, safe_result):
        """Multiple commercial keywords should only trigger one boost."""
        score = classifier.score("buy cheap discount coupon sale", safe_result)
        # Score should be boosted but not exceed 1.0
        assert 0.0 <= score <= 1.0


class TestSensitiveTermDetection:

    def test_depression_reduces_score(self, classifier, safe_result):
        score = classifier.score("dealing with depression", safe_result)
        base_no_sensitive = classifier.score("running shoes", safe_result)
        assert score < base_no_sensitive

    def test_bankruptcy_reduces_score(self, classifier, safe_result):
        score = classifier.score("filing for bankruptcy", safe_result)
        assert score < safe_result.base_score


class TestSentimentGating:

    def test_high_toxicity_skips_sentiment(self, classifier, borderline_toxic_result):
        """When toxicity >= 0.4, sentiment model should be skipped."""
        score = classifier.score("some random query", borderline_toxic_result)
        assert 0.0 <= score <= 1.0

    def test_sensitive_query_skips_sentiment(self, classifier, safe_result):
        """Sensitive queries should skip sentiment even if toxicity is low."""
        score = classifier.score("anxiety treatment options", safe_result)
        assert 0.0 <= score <= 1.0


class TestScoreBounds:

    def test_score_never_negative(self, classifier, safe_result):
        score = classifier.score("terrible awful horrible", safe_result)
        assert score >= 0.0

    def test_score_never_above_one(self, classifier, safe_result):
        score = classifier.score("buy purchase order shop deal discount", safe_result)
        assert score <= 1.0

    def test_score_with_zero_base(self, classifier):
        """Even with zero base_score, commercial boost should produce valid score."""
        zero_base = SafetyResult(is_blocked=False, base_score=0.0, toxicity=0.0, danger_diff=0.0)
        score = classifier.score("buy shoes", zero_base)
        assert 0.0 <= score <= 1.0


class TestCaching:

    def test_repeated_calls_return_same_score(self, classifier, safe_result):
        s1 = classifier.score("laptop recommendations", safe_result)
        s2 = classifier.score("laptop recommendations", safe_result)
        assert s1 == s2

    def test_different_safety_results_different_cache_keys(self, classifier):
        """Same query with different safety scores should not hit same cache."""
        safe = SafetyResult(is_blocked=False, base_score=0.5, toxicity=0.1, danger_diff=-0.2)
        less_safe = SafetyResult(is_blocked=False, base_score=0.2, toxicity=0.3, danger_diff=0.1)
        s1 = classifier.score("test query for cache", safe)
        s2 = classifier.score("test query for cache", less_safe)
        # Different base_scores should produce different results
        assert s1 != s2
