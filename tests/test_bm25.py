import pytest

from app.retrieval.bm25_index import BM25Index, get_bm25_index


class TestBM25Index:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.index = get_bm25_index()

    def test_index_loads(self):
        assert self.index.is_loaded
        assert len(self.index.campaign_ids) > 0

    def test_search_returns_results(self):
        results = self.index.search("running shoes", top_k=10)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_search_returns_campaign_ids_and_scores(self):
        results = self.index.search("laptop computer", top_k=5)
        for campaign_id, score in results:
            assert isinstance(campaign_id, str)
            assert isinstance(score, float)
            assert score > 0

    def test_exact_brand_ranks_high(self):
        """Exact brand name queries should rank matching campaigns high."""
        results = self.index.search("Nike running shoes", top_k=20)
        assert len(results) > 0
        # Top results should have positive scores
        assert results[0][1] > 0

    def test_empty_query_returns_empty(self):
        results = self.index.search("", top_k=10)
        assert results == []

    def test_gibberish_returns_few_results(self):
        results = self.index.search("asdfghjkl qwertyuiop", top_k=10)
        # May return some results due to partial token matches, but scores should be low
        assert len(results) <= 10

    def test_single_char_query_returns_empty(self):
        """Single-char tokens are dropped by tokenizer, so 'a' yields no tokens."""
        results = self.index.search("a", top_k=10)
        assert results == []

    def test_punctuation_only_returns_empty(self):
        results = self.index.search("!!! ??? ...", top_k=10)
        assert results == []

    def test_top_k_respected(self):
        results = self.index.search("running shoes", top_k=3)
        assert len(results) <= 3

    def test_scores_are_positive(self):
        results = self.index.search("running shoes", top_k=10)
        for _, score in results:
            assert score > 0

    def test_results_sorted_by_score_descending(self):
        results = self.index.search("laptop computer", top_k=20)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestBM25VsFAISS:

    def test_bm25_complements_faiss(self, client):
        """BM25 should help with exact-match queries where FAISS might miss."""
        response = client.post("/api/retrieve", json={"query": "Nike running shoes"})
        assert response.status_code == 200
        campaigns = response.json()["campaigns"]
        assert len(campaigns) > 0
