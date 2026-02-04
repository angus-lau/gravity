import pytest


class TestHarmfulContentBlocked:
    """
    Test that clearly harmful/toxic content gets low eligibility scores.
    Uses ML-based toxicity detection which generalizes beyond hardcoded patterns.
    """

    @pytest.mark.parametrize("query", [
        "how to make a pipe bomb",
        "how to make explosives at home",
        "I want to kill someone",
        "how to poison someone",
    ])
    def test_violent_content_blocked(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        assert response.json()["ad_eligibility"] < 0.3

    @pytest.mark.parametrize("query", [
        "I'm having thoughts of self-harm",
        "I want to end my life",
        "suicide methods",
    ])
    def test_self_harm_blocked(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        assert response.json()["ad_eligibility"] < 0.3

    @pytest.mark.parametrize("query", [
        "I hate all immigrants they should die",
        "kill all the jews",
    ])
    def test_hate_speech_blocked(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        # Hate speech gets very low scores due to toxicity model
        assert response.json()["ad_eligibility"] < 0.2


class TestSensitiveContentLowerEligibility:
    """
    Test that sensitive/negative content gets lower (but not blocked) eligibility.
    The ML model uses sentiment to reduce scores for negative contexts.
    """

    @pytest.mark.parametrize("query", [
        "my mom just passed away",
        "I just found out I have cancer",
        "my house burned down",
        "I'm really depressed and struggling",
    ])
    def test_negative_sentiment_lower_eligibility(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        # Negative sentiment should result in lower eligibility
        assert response.json()["ad_eligibility"] < 0.5


class TestCommercialQueriesEligible:
    """
    Test that commercial/positive queries get higher eligibility scores.
    """

    @pytest.mark.parametrize("query", [
        "best running shoes for flat feet",
        "where to buy organic coffee beans",
        "compare iPhone 15 vs Samsung S24",
        "laptop recommendations for programming",
        "cheap flights to Hawaii",
        "best wireless headphones under $200",
    ])
    def test_product_queries_high_eligibility(self, client, query):
        # Commercial queries should have moderate-to-high eligibility (>= 0.45)
        response = client.post("/api/retrieve", json={"query": query})
        assert response.json()["ad_eligibility"] >= 0.45

    @pytest.mark.parametrize("query", [
        "I need new shoes",
        "looking for a good restaurant",
        "want to upgrade my phone",
    ])
    def test_purchase_intent_moderate_eligibility(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        # These neutral-phrased queries get moderate scores
        assert response.json()["ad_eligibility"] >= 0.3


class TestInformationalQueriesModerate:
    """
    Test that neutral informational queries get moderate eligibility.
    """

    @pytest.mark.parametrize("query", [
        "what is the history of the marathon",
        "why do runners get blisters",
        "how does caffeine affect athletic performance",
    ])
    def test_informational_moderate_eligibility(self, client, query):
        response = client.post("/api/retrieve", json={"query": query})
        eligibility = response.json()["ad_eligibility"]
        # Neutral/informational queries should have moderate eligibility
        assert 0.25 <= eligibility <= 0.8


class TestEligibilityConsistency:
    """
    Test that the ML model produces consistent and coherent scores.
    """

    def test_similar_queries_similar_scores(self, client):
        queries = ["best running shoes", "top running shoes", "recommended running shoes"]
        scores = [
            client.post("/api/retrieve", json={"query": q}).json()["ad_eligibility"]
            for q in queries
        ]
        # All should be moderate-to-high eligibility
        assert all(s > 0.4 for s in scores)
        # Scores should be relatively close to each other
        assert max(scores) - min(scores) < 0.3

    def test_repeated_queries_consistent(self, client, sample_query):
        scores = [
            client.post("/api/retrieve", json=sample_query).json()["ad_eligibility"]
            for _ in range(3)
        ]
        assert len(set(scores)) == 1

    def test_toxic_vs_safe_ordering(self, client):
        """Toxic content should always score lower than safe commercial content."""
        toxic_score = client.post(
            "/api/retrieve", json={"query": "I want to hurt people"}
        ).json()["ad_eligibility"]
        safe_score = client.post(
            "/api/retrieve", json={"query": "best running shoes"}
        ).json()["ad_eligibility"]
        assert toxic_score < safe_score
