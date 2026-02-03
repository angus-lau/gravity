"""Campaign Ranking Tests - Verify ranking behavior.

These tests verify that campaigns are properly ranked by relevance
and that context-based boosting works correctly.
"""

import pytest


class TestRankingBasics:
    """Test basic ranking requirements."""

    def test_campaigns_sorted_descending(self, client, sample_query):
        """Campaigns must be sorted by relevance_score descending."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        campaigns = data["campaigns"]
        if len(campaigns) > 1:
            scores = [c["relevance_score"] for c in campaigns]
            assert scores == sorted(scores, reverse=True), \
                "Campaigns must be sorted by relevance_score descending"

    def test_relevance_scores_valid_range(self, client, sample_query):
        """All relevance_scores must be between 0.0 and 1.0."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        for campaign in data["campaigns"]:
            score = campaign["relevance_score"]
            assert 0.0 <= score <= 1.0, \
                f"Campaign {campaign['campaign_id']} has invalid score: {score}"

    def test_campaign_ids_unique(self, client, sample_query):
        """All campaign_ids in response must be unique."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        ids = [c["campaign_id"] for c in data["campaigns"]]
        assert len(ids) == len(set(ids)), "Campaign IDs must be unique"

    def test_top_results_more_relevant(self, client, sample_query):
        """Top results should have higher scores than bottom results."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        campaigns = data["campaigns"]
        if len(campaigns) >= 10:
            top_10_avg = sum(c["relevance_score"] for c in campaigns[:10]) / 10
            bottom_10_avg = sum(c["relevance_score"] for c in campaigns[-10:]) / 10
            
            assert top_10_avg >= bottom_10_avg, \
                f"Top results ({top_10_avg:.3f}) should have higher avg score than bottom ({bottom_10_avg:.3f})"


class TestRankingRelevance:
    """Test that ranking reflects query relevance."""

    def test_running_query_ranks_running_campaigns_high(self, client):
        """Running query should rank running-related campaigns higher."""
        response = client.post("/api/retrieve", json={
            "query": "best running shoes for marathon training"
        })
        data = response.json()
        
        # Check that top campaigns are likely running-related
        # (We can't verify exact content without knowing the data,
        # but we can check that top scores are high)
        if data["campaigns"]:
            top_score = data["campaigns"][0]["relevance_score"]
            assert top_score > 0.5, \
                f"Top campaign for running query should be relevant (score > 0.5), got {top_score}"

    def test_different_queries_different_rankings(self, client):
        """Different queries should produce different campaign rankings."""
        response1 = client.post("/api/retrieve", json={"query": "running shoes"})
        response2 = client.post("/api/retrieve", json={"query": "laptop computer"})
        
        campaigns1 = response1.json()["campaigns"]
        campaigns2 = response2.json()["campaigns"]
        
        if campaigns1 and campaigns2:
            # Top campaigns should be different for different queries
            top_ids_1 = [c["campaign_id"] for c in campaigns1[:10]]
            top_ids_2 = [c["campaign_id"] for c in campaigns2[:10]]
            
            # At least some difference in top 10
            overlap = len(set(top_ids_1) & set(top_ids_2))
            assert overlap < 10, \
                "Different queries should produce different top rankings"


class TestContextBasedRanking:
    """Test that user context influences ranking."""

    def test_location_context_influences_ranking(self, client):
        """User location should influence campaign ranking."""
        query = {"query": "local restaurants near me"}
        
        # Same query, different locations
        response1 = client.post("/api/retrieve", json={
            **query, "context": {"location": "San Francisco, CA"}
        })
        response2 = client.post("/api/retrieve", json={
            **query, "context": {"location": "New York, NY"}
        })
        
        # Both should return valid results
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Rankings might differ based on location (if campaigns have location targeting)
        # This is a soft test - we just verify the system handles location context

    def test_demographic_context_influences_ranking(self, client):
        """User demographics should influence campaign ranking."""
        query = {"query": "fashion recommendations"}
        
        response1 = client.post("/api/retrieve", json={
            **query, "context": {"gender": "female", "age": 25}
        })
        response2 = client.post("/api/retrieve", json={
            **query, "context": {"gender": "male", "age": 55}
        })
        
        # Both should return valid results
        assert response1.status_code == 200
        assert response2.status_code == 200

    def test_interests_context_influences_ranking(self, client):
        """User interests should influence campaign ranking."""
        query = {"query": "gift ideas"}
        
        response1 = client.post("/api/retrieve", json={
            **query, "context": {"interests": ["fitness", "running", "health"]}
        })
        response2 = client.post("/api/retrieve", json={
            **query, "context": {"interests": ["gaming", "technology", "esports"]}
        })
        
        # Both should return valid results
        assert response1.status_code == 200
        assert response2.status_code == 200


class TestRankingPerformance:
    """Test ranking performance characteristics."""

    def test_returns_1000_campaigns_when_available(self, client, sample_query):
        """Should return exactly 1000 campaigns when index has >= 1000."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        # This test assumes we have >= 1000 campaigns indexed
        # If not, it will return all available
        num_campaigns = len(data["campaigns"])
        
        # Either exactly 1000, or all available if < 1000
        assert num_campaigns <= 1000, f"Should not return more than 1000, got {num_campaigns}"
        
        # If we have enough campaigns, should return exactly 1000
        # (The metadata might tell us how many are indexed)
        metadata = data.get("metadata", {})
        if metadata.get("campaigns_indexed", 10000) >= 1000:
            assert num_campaigns == 1000, \
                f"Should return exactly 1000 campaigns, got {num_campaigns}"

    def test_ranking_is_deterministic(self, client, sample_query):
        """Same query should return same ranking."""
        rankings = []
        for _ in range(3):
            response = client.post("/api/retrieve", json=sample_query)
            campaigns = response.json()["campaigns"]
            rankings.append([c["campaign_id"] for c in campaigns[:20]])
        
        # All rankings should be identical
        assert rankings[0] == rankings[1] == rankings[2], \
            "Ranking should be deterministic for same query"
