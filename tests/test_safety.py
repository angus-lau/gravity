import time

import pytest

from app.models.safety import BlocklistChecker, _normalize_obfuscation, get_blocklist_checker


class TestBlocklistChecker:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.checker = get_blocklist_checker()

    @pytest.mark.parametrize("query", [
        "free porn videos",
        "xxx content",
        "nude photos nsfw",
    ])
    def test_nsfw_blocked(self, query):
        assert self.checker.is_blocked(query) is True

    @pytest.mark.parametrize("query", [
        "how to make a pipe bomb",
        "terrorist attack planning",
        "mass shooting news",
    ])
    def test_violence_blocked(self, query):
        assert self.checker.is_blocked(query) is True

    @pytest.mark.parametrize("query", [
        "I want to kill myself",
        "suicide methods",
        "how to die painlessly",
    ])
    def test_self_harm_blocked(self, query):
        assert self.checker.is_blocked(query) is True

    @pytest.mark.parametrize("query", [
        "my mom just passed away",
        "cancer diagnosis what to expect",
        "hospice care options",
    ])
    def test_tragedy_blocked(self, query):
        assert self.checker.is_blocked(query) is True

    @pytest.mark.parametrize("query", [
        "I hate all immigrants they should die",
    ])
    def test_hate_speech_blocked(self, query):
        assert self.checker.is_blocked(query) is True

    @pytest.mark.parametrize("query", [
        "best running shoes",
        "laptop recommendations",
        "how to cook pasta",
        "weather today",
        "history of ancient Rome",
    ])
    def test_safe_queries_pass(self, query):
        assert self.checker.is_blocked(query) is False

    # ── Obfuscation / leetspeak bypass attempts ──

    @pytest.mark.parametrize("query", [
        "how to make a b0mb",
        "how to make a b.o.m.b",
        "how to make a b-o-m-b",
        "expl0s1ve device",
        "t3rror1st attack",
        "t3rror1sm",
        "a$$a$$inate",
    ])
    def test_leetspeak_violence_blocked(self, query):
        assert self.checker.is_blocked(query) is True

    @pytest.mark.parametrize("query", [
        "su1c1de methods",
        "su!c!de",
        "k1ll mys3lf",
        "s3lf-h4rm",
        "want to d1e",
    ])
    def test_leetspeak_self_harm_blocked(self, query):
        assert self.checker.is_blocked(query) is True

    @pytest.mark.parametrize("query", [
        "p0rn videos",
        "p.o.r.n",
        "n-u-d-e photos",
        "n4ked pics",
    ])
    def test_leetspeak_nsfw_blocked(self, query):
        assert self.checker.is_blocked(query) is True

    @pytest.mark.parametrize("query", [
        "best runn1ng sh0es",
        "l4pt0p deals",
        "g00d r3staurant",
    ])
    def test_leetspeak_safe_queries_pass(self, query):
        """Leetspeak in safe queries should NOT trigger false positives."""
        assert self.checker.is_blocked(query) is False

    def test_blocklist_speed(self):
        """Blocklist check should complete in well under 1ms."""
        queries = ["best running shoes", "how to make a bomb", "laptop deals", "b0mb thr3at"]
        for q in queries:
            start = time.perf_counter()
            self.checker.is_blocked(q)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 1.0, f"Blocklist took {elapsed_ms:.3f}ms for '{q}'"


class TestObfuscationNormalization:

    @pytest.mark.parametrize("input_text,expected", [
        ("b0mb", "bomb"),
        ("su1c1de", "suicide"),
        ("p.o.r.n", "porn"),
        ("k!ll", "kill"),
        ("t3rror1st", "terrorist"),
        ("n4ked", "naked"),
        ("3xpl0s1v3", "explosive"),
        ("h-a-t-e", "hate"),
        ("b_o_m_b", "bomb"),
        ("a$$a$$inate", "assassinate"),
    ])
    def test_normalization_decodes(self, input_text, expected):
        result = _normalize_obfuscation(input_text)
        assert expected in result, f"Expected '{expected}' in normalized '{result}' (from '{input_text}')"

    def test_normalization_collapses_repeats(self):
        assert "naked" in _normalize_obfuscation("naaaked")
        assert "bomb" in _normalize_obfuscation("boooomb")

    def test_normalization_preserves_normal_text(self):
        result = _normalize_obfuscation("running shoes")
        assert "running" in result and "shoes" in result
        assert _normalize_obfuscation("best laptop") == "best laptop"

    def test_normalization_preserves_spaces_between_words(self):
        """Multi-word blocklist terms like 'kill myself' must keep spaces."""
        assert "kill myself" in _normalize_obfuscation("k1ll mys3lf")
        assert "want to die" in _normalize_obfuscation("want to d1e")


class TestSafetyClassifier:

    def test_toxic_query_low_safety(self, client):
        """Toxic queries should get low safety → low eligibility."""
        response = client.post("/api/retrieve", json={"query": "I want to hurt people"})
        assert response.json()["ad_eligibility"] < 0.3

    def test_safe_query_high_safety(self, client):
        """Safe queries should get high safety → allows commercial scoring."""
        response = client.post("/api/retrieve", json={"query": "best running shoes"})
        assert response.json()["ad_eligibility"] >= 0.45


class TestEarlyStoppingPerformance:

    def test_blocked_query_fast_response(self, client):
        """Blocklisted queries should return very quickly."""
        start = time.perf_counter()
        response = client.post("/api/retrieve", json={"query": "how to make a pipe bomb"})
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200
        assert response.json()["ad_eligibility"] == 0.0
        assert response.json()["campaigns"] == []
        # Should be very fast since it short-circuits at blocklist
        assert elapsed_ms < 50, f"Blocked query took {elapsed_ms:.1f}ms (expected <50ms)"

    def test_timing_metadata_includes_breakdown(self, client):
        """Response should include modular timing breakdown."""
        response = client.post("/api/retrieve", json={"query": "best running shoes"})
        timing = response.json()["metadata"]["timing"]
        assert "blocklist_ms" in timing
        assert "eligibility_ms" in timing
        assert timing["blocklist_ms"] >= 0
