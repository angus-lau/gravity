import pytest


class TestRankingBasics:

    def test_campaigns_sorted_descending(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        campaigns = response.json()["campaigns"]
        if len(campaigns) > 1:
            scores = [c["relevance_score"] for c in campaigns]
            assert scores == sorted(scores, reverse=True)

    def test_relevance_scores_valid_range(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        for campaign in response.json()["campaigns"]:
            assert 0.0 <= campaign["relevance_score"] <= 1.0

    def test_campaign_ids_unique(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        ids = [c["campaign_id"] for c in response.json()["campaigns"]]
        assert len(ids) == len(set(ids))

    def test_top_results_more_relevant(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        campaigns = response.json()["campaigns"]
        if len(campaigns) >= 10:
            top_avg = sum(c["relevance_score"] for c in campaigns[:10]) / 10
            bottom_avg = sum(c["relevance_score"] for c in campaigns[-10:]) / 10
            assert top_avg >= bottom_avg


class TestRankingRelevance:

    def test_running_query_ranks_relevant_high(self, client):
        response = client.post("/api/retrieve", json={
            "query": "best running shoes for marathon training"
        })
        campaigns = response.json()["campaigns"]
        if campaigns:
            assert campaigns[0]["relevance_score"] > 0.5

    def test_different_queries_different_rankings(self, client):
        r1 = client.post("/api/retrieve", json={"query": "running shoes"})
        r2 = client.post("/api/retrieve", json={"query": "laptop computer"})

        top1 = [c["campaign_id"] for c in r1.json()["campaigns"][:10]]
        top2 = [c["campaign_id"] for c in r2.json()["campaigns"][:10]]

        if top1 and top2:
            overlap = len(set(top1) & set(top2))
            assert overlap < 10


class TestContextBasedRanking:

    def test_location_context_accepted(self, client):
        for loc in ["San Francisco, CA", "New York, NY"]:
            response = client.post("/api/retrieve", json={
                "query": "local restaurants near me",
                "context": {"location": loc}
            })
            assert response.status_code == 200

    def test_demographic_context_accepted(self, client):
        for ctx in [{"gender": "female", "age": 25}, {"gender": "male", "age": 55}]:
            response = client.post("/api/retrieve", json={
                "query": "fashion recommendations",
                "context": ctx
            })
            assert response.status_code == 200


class TestRankingPerformance:

    def test_returns_up_to_1000_campaigns(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        assert len(response.json()["campaigns"]) <= 1000

    def test_ranking_is_deterministic(self, client, sample_query):
        rankings = [
            [c["campaign_id"] for c in client.post("/api/retrieve", json=sample_query).json()["campaigns"][:20]]
            for _ in range(3)
        ]
        assert rankings[0] == rankings[1] == rankings[2]
