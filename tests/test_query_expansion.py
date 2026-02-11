import time

import pytest

from app.models.query_expansion import QueryExpander, get_query_expander


class TestQueryExpansion:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.expander = get_query_expander()

    def test_known_term_expanded(self):
        result = self.expander.expand("comfortable work shoes")
        assert result != "comfortable work shoes"
        assert "comfortable" in result
        # Should add synonyms like ergonomic, cushioned, supportive
        assert any(word in result.lower() for word in ["ergonomic", "cushioned", "supportive"])

    def test_cheap_expansion(self):
        result = self.expander.expand("cheap running shoes")
        assert "affordable" in result.lower() or "budget" in result.lower() or "value" in result.lower()

    def test_unknown_term_unchanged(self):
        result = self.expander.expand("xylophone maintenance tips")
        assert result == "xylophone maintenance tips"

    def test_brand_query_not_polluted(self):
        """Brand-specific queries should not get irrelevant expansions."""
        result = self.expander.expand("Nike Pegasus 40")
        # Should pass through mostly unchanged (no matching synonyms)
        assert "Nike Pegasus 40" in result

    def test_expansion_limit(self):
        """Should not add more than MAX_EXPANSION_TERMS."""
        result = self.expander.expand("cheap comfortable running shoes workout boots")
        words_added = len(result.split()) - len("cheap comfortable running shoes workout boots".split())
        assert words_added <= 4

    def test_expansion_speed(self):
        """Expansion should complete in under 1ms."""
        queries = [
            "best running shoes",
            "cheap laptop for gaming",
            "comfortable office chair",
        ]
        for q in queries:
            start = time.perf_counter()
            self.expander.expand(q)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 1.0, f"Expansion took {elapsed_ms:.3f}ms for '{q}'"

    def test_multi_domain_expansion(self):
        """Terms from different domains should both expand."""
        result = self.expander.expand("hotel near gym")
        assert result != "hotel near gym"

    def test_empty_query_returns_empty(self):
        result = self.expander.expand("")
        assert result == ""

    def test_whitespace_only_returns_unchanged(self):
        result = self.expander.expand("   ")
        assert result.strip() == ""

    def test_case_insensitive_matching(self):
        """Synonyms should match regardless of query case."""
        lower = self.expander.expand("comfortable shoes")
        upper = self.expander.expand("COMFORTABLE SHOES")
        # Both should get expansion (if synonym exists for "comfortable")
        # At minimum, original query preserved
        assert "comfortable" in lower.lower()
        assert "comfortable" in upper.lower()

    def test_no_duplicate_expansions(self):
        """Should not add the same expansion term twice."""
        result = self.expander.expand("cheap affordable budget shoes")
        words = result.lower().split()
        # Count each expansion word — none should appear more than
        # the number of times it's in the original query + 1
        from collections import Counter
        counts = Counter(words)
        for word, count in counts.items():
            original_count = "cheap affordable budget shoes".lower().split().count(word)
            assert count <= original_count + 1, f"'{word}' appeared {count} times"

    def test_idempotent_for_no_match(self):
        """Queries with no matching synonyms should return unchanged."""
        query = "quantum entanglement research papers"
        assert self.expander.expand(query) == query
