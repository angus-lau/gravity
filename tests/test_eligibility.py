import pytest


class TestHarmfulContentBlocked:

    @pytest.mark.parametrize("query", [
        "how to make a pipe bomb",
        "how to make explosives at home",
        "ways to hurt someone",
        "how to poison someone",
    ])
    def test_violent_content_blocked(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        assert response.json()["ad_eligibility"] < 0.1

    @pytest.mark.parametrize("query", [
        "my mom just passed away",
        "my dog died today",
        "I just found out I have cancer",
        "my house burned down",
    ])
    def test_tragedy_blocked(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        assert response.json()["ad_eligibility"] < 0.1

    @pytest.mark.parametrize("query", [
        "I'm having thoughts of self-harm",
        "I want to end my life",
        "suicide methods",
    ])
    def test_self_harm_blocked(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        assert response.json()["ad_eligibility"] < 0.05

    @pytest.mark.parametrize("query", [
        "explicit nsfw adult content",
        "pornographic material",
    ])
    def test_nsfw_blocked(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        assert response.json()["ad_eligibility"] < 0.1


class TestCommercialQueriesEligible:

    @pytest.mark.parametrize("query", [
        "best running shoes for flat feet",
        "where to buy organic coffee beans",
        "compare iPhone 15 vs Samsung S24",
        "laptop recommendations for programming",
        "cheap flights to Hawaii",
        "best wireless headphones under $200",
    ])
    def test_product_queries_high_eligibility(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        assert response.json()["ad_eligibility"] > 0.7

    @pytest.mark.parametrize("query", [
        "I need new shoes",
        "looking for a good restaurant",
        "want to upgrade my phone",
    ])
    def test_purchase_intent_high_eligibility(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        assert response.json()["ad_eligibility"] > 0.6


class TestInformationalQueriesModerate:

    @pytest.mark.parametrize("query", [
        "what is the history of the marathon",
        "why do runners get blisters",
        "how does caffeine affect athletic performance",
    ])
    def test_informational_moderate_eligibility(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        eligibility = response.json()["ad_eligibility"]
        assert 0.4 <= eligibility <= 0.95


class TestEligibilityConsistency:

    def test_similar_queries_similar_scores(self, client):
        queries = ["best running shoes", "top running shoes", "recommended running shoes"]
        scores = [
            client.post("/api/retrieve", json={"query": q}).json()["ad_eligibility"]
            for q in queries
        ]
        assert all(s > 0.6 for s in scores)
        assert max(scores) - min(scores) < 0.3

    def test_repeated_queries_consistent(self, client, sample_query):
        scores = [
            client.post("/api/retrieve", json=sample_query).json()["ad_eligibility"]
            for _ in range(3)
        ]
        assert len(set(scores)) == 1
