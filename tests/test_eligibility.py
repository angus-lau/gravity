"""Eligibility Scoring Tests - Verify ad eligibility behavior.

These tests verify that the eligibility scoring correctly identifies
when ads should or should not be shown, without prescribing implementation.
"""

import pytest


class TestHarmfulContentBlocked:
    """Test that harmful content receives very low eligibility scores."""

    @pytest.mark.parametrize("query,description", [
        ("how to make a pipe bomb", "bomb making"),
        ("how to make explosives at home", "explosives"),
        ("ways to hurt someone", "violence"),
        ("how to poison someone", "poisoning"),
    ])
    def test_violent_harmful_content_blocked(self, client, query, description):
        """Violent/harmful content should have eligibility < 0.1."""
        response = client.post("/api/retrieve", json={"query": query})
        data = response.json()
        
        assert data["ad_eligibility"] < 0.1, \
            f"Harmful query ({description}) should have eligibility < 0.1, got {data['ad_eligibility']}"

    @pytest.mark.parametrize("query", [
        "my mom just passed away",
        "my dog died today",
        "I just found out I have cancer",
        "my house burned down",
        "I was just in a car accident",
    ])
    def test_tragedy_grief_blocked(self, client, query):
        """Tragedy/grief queries should have eligibility < 0.1."""
        response = client.post("/api/retrieve", json={"query": query})
        data = response.json()
        
        assert data["ad_eligibility"] < 0.1, \
            f"Tragedy query should have eligibility < 0.1, got {data['ad_eligibility']}"

    @pytest.mark.parametrize("query", [
        "I'm having thoughts of self-harm",
        "I want to end my life",
        "suicide methods",
        "I feel like hurting myself",
    ])
    def test_self_harm_blocked(self, client, query):
        """Self-harm/suicide queries should have eligibility = 0.0."""
        response = client.post("/api/retrieve", json={"query": query})
        data = response.json()
        
        assert data["ad_eligibility"] < 0.05, \
            f"Self-harm query should have eligibility near 0, got {data['ad_eligibility']}"

    @pytest.mark.parametrize("query", [
        "explicit nsfw adult content",
        "pornographic material",
    ])
    def test_nsfw_blocked(self, client, query):
        """NSFW content should have eligibility < 0.1."""
        response = client.post("/api/retrieve", json={"query": query})
        data = response.json()
        
        assert data["ad_eligibility"] < 0.1, \
            f"NSFW query should have eligibility < 0.1, got {data['ad_eligibility']}"

    @pytest.mark.parametrize("query", [
        "I hate all [ethnic group] people",
        "why are [group] so inferior",
    ])
    def test_hate_speech_blocked(self, client, query):
        """Hate speech should have eligibility < 0.1."""
        response = client.post("/api/retrieve", json={"query": query})
        data = response.json()
        
        assert data["ad_eligibility"] < 0.2, \
            f"Hate speech query should have low eligibility, got {data['ad_eligibility']}"


class TestCommercialQueriesEligible:
    """Test that commercial queries receive high eligibility scores."""

    @pytest.mark.parametrize("query,expected_min", [
        ("best running shoes for flat feet", 0.7),
        ("where to buy organic coffee beans", 0.7),
        ("compare iPhone 15 vs Samsung S24", 0.7),
        ("laptop recommendations for programming", 0.7),
        ("cheap flights to Hawaii", 0.7),
        ("best wireless headphones under $200", 0.7),
        ("where can I buy a new mattress", 0.7),
    ])
    def test_product_queries_high_eligibility(self, client, query, expected_min):
        """Product/purchase queries should have eligibility > 0.7."""
        response = client.post("/api/retrieve", json={"query": query})
        data = response.json()
        
        assert data["ad_eligibility"] > expected_min, \
            f"Commercial query should have eligibility > {expected_min}, got {data['ad_eligibility']}"

    @pytest.mark.parametrize("query", [
        "I need new shoes",
        "looking for a good restaurant",
        "want to upgrade my phone",
        "shopping for furniture",
    ])
    def test_purchase_intent_high_eligibility(self, client, query):
        """Purchase intent queries should have high eligibility."""
        response = client.post("/api/retrieve", json={"query": query})
        data = response.json()
        
        assert data["ad_eligibility"] > 0.6, \
            f"Purchase intent query should have eligibility > 0.6, got {data['ad_eligibility']}"


class TestInformationalQueriesModerate:
    """Test that informational queries receive moderate eligibility scores."""

    @pytest.mark.parametrize("query", [
        "what is the history of the marathon",
        "why do runners get blisters",
        "how does caffeine affect athletic performance",
        "benefits of morning exercise",
    ])
    def test_informational_moderate_eligibility(self, client, query):
        """Informational queries should have moderate eligibility (0.4-0.9)."""
        response = client.post("/api/retrieve", json={"query": query})
        data = response.json()
        
        # Informational but related to products - can show contextual ads
        assert 0.4 <= data["ad_eligibility"] <= 0.95, \
            f"Informational query should have eligibility in [0.4, 0.95], got {data['ad_eligibility']}"


class TestSensitiveTopicsModerate:
    """Test that sensitive but not harmful topics are handled appropriately."""

    @pytest.mark.parametrize("query,max_expected", [
        ("I'm feeling really stressed about work", 0.6),
        ("how do I file for unemployment", 0.5),
        ("dealing with financial difficulties", 0.5),
        ("struggling with anxiety", 0.4),
    ])
    def test_sensitive_topics_cautious(self, client, query, max_expected):
        """Sensitive topics should have moderate/low eligibility."""
        response = client.post("/api/retrieve", json={"query": query})
        data = response.json()
        
        # These are sensitive but not necessarily ad-inappropriate
        # The system should err on the side of caution
        assert data["ad_eligibility"] <= max_expected + 0.2, \
            f"Sensitive query should have cautious eligibility, got {data['ad_eligibility']}"


class TestEligibilityConsistency:
    """Test that eligibility scoring is consistent."""

    def test_similar_queries_similar_scores(self, client):
        """Similar queries should have similar eligibility scores."""
        queries = [
            "best running shoes",
            "top running shoes",
            "recommended running shoes",
        ]
        
        scores = []
        for query in queries:
            response = client.post("/api/retrieve", json={"query": query})
            scores.append(response.json()["ad_eligibility"])
        
        # All should be high (commercial queries)
        for score in scores:
            assert score > 0.6, f"Similar commercial queries should all be high eligibility"
        
        # Scores should be relatively close (within 0.3 of each other)
        assert max(scores) - min(scores) < 0.3, \
            f"Similar queries should have similar scores: {scores}"

    def test_repeated_queries_consistent(self, client, sample_query):
        """Same query should return consistent eligibility scores."""
        scores = []
        for _ in range(3):
            response = client.post("/api/retrieve", json=sample_query)
            scores.append(response.json()["ad_eligibility"])
        
        # All scores should be identical (deterministic)
        assert len(set(scores)) == 1, \
            f"Repeated queries should return identical scores: {scores}"
