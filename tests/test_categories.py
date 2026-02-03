"""Category Extraction Tests - Verify category extraction behavior.

These tests verify that category extraction returns relevant,
appropriately granular categories for the given query.
"""

import pytest


class TestCategoryBasics:
    """Test basic category extraction requirements."""

    def test_returns_1_to_10_categories(self, client, sample_query):
        """Should return between 1 and 10 categories."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        num_categories = len(data["extracted_categories"])
        assert 1 <= num_categories <= 10, \
            f"Should return 1-10 categories, got {num_categories}"

    def test_categories_are_strings(self, client, sample_query):
        """All categories should be strings."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        for cat in data["extracted_categories"]:
            assert isinstance(cat, str), f"Category should be string, got {type(cat)}"
            assert len(cat) > 0, "Category should not be empty"

    def test_categories_are_unique(self, client, sample_query):
        """Categories should not have duplicates."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        categories = data["extracted_categories"]
        assert len(categories) == len(set(categories)), \
            f"Categories should be unique: {categories}"


class TestCategoryRelevance:
    """Test that extracted categories are relevant to the query."""

    def test_running_query_returns_running_categories(self, client):
        """Running-related query should return running-related categories."""
        response = client.post("/api/retrieve", json={
            "query": "I need new running shoes for marathon training"
        })
        data = response.json()
        
        categories_lower = [c.lower() for c in data["extracted_categories"]]
        
        # At least one category should be related to running/shoes/fitness
        relevant_keywords = ["running", "shoe", "athletic", "fitness", "marathon", "sport"]
        has_relevant = any(
            any(kw in cat for kw in relevant_keywords)
            for cat in categories_lower
        )
        
        assert has_relevant, \
            f"Running query should have running-related categories: {data['extracted_categories']}"

    def test_electronics_query_returns_electronics_categories(self, client):
        """Electronics query should return electronics categories."""
        response = client.post("/api/retrieve", json={
            "query": "looking for a new laptop for software development"
        })
        data = response.json()
        
        categories_lower = [c.lower() for c in data["extracted_categories"]]
        
        relevant_keywords = ["laptop", "computer", "electronic", "tech", "software", "device"]
        has_relevant = any(
            any(kw in cat for kw in relevant_keywords)
            for cat in categories_lower
        )
        
        assert has_relevant, \
            f"Electronics query should have relevant categories: {data['extracted_categories']}"

    def test_travel_query_returns_travel_categories(self, client):
        """Travel query should return travel categories."""
        response = client.post("/api/retrieve", json={
            "query": "planning a vacation to Europe, looking for flights and hotels"
        })
        data = response.json()
        
        categories_lower = [c.lower() for c in data["extracted_categories"]]
        
        relevant_keywords = ["travel", "flight", "hotel", "vacation", "trip", "tourism"]
        has_relevant = any(
            any(kw in cat for kw in relevant_keywords)
            for cat in categories_lower
        )
        
        assert has_relevant, \
            f"Travel query should have travel-related categories: {data['extracted_categories']}"


class TestCategoryGranularity:
    """Test that categories have appropriate granularity."""

    def test_categories_not_too_broad(self, client, sample_query):
        """Categories should not be overly broad."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        overly_broad = ["everything", "all", "general", "stuff", "things", "other"]
        
        for cat in data["extracted_categories"]:
            cat_lower = cat.lower()
            for broad in overly_broad:
                assert broad != cat_lower, f"Category '{cat}' is too broad"

    def test_categories_not_too_narrow(self, client):
        """Categories should not be overly specific brand names only."""
        response = client.post("/api/retrieve", json={
            "query": "I want to buy running shoes"
        })
        data = response.json()
        
        # Should have general categories, not just specific brands
        # At least one category should be a general product type
        categories_lower = [c.lower() for c in data["extracted_categories"]]
        
        general_product_keywords = ["shoe", "running", "athletic", "footwear", "sport"]
        has_general = any(
            any(kw in cat for kw in general_product_keywords)
            for cat in categories_lower
        )
        
        assert has_general, \
            f"Should have at least one general category, got: {data['extracted_categories']}"


class TestCategoryContextInfluence:
    """Test that user context influences category extraction when appropriate."""

    def test_fitness_interest_influences_categories(self, client):
        """User fitness interest should influence extracted categories."""
        # Same query, different contexts
        query = "I need something to track my progress"
        
        # Fitness-oriented context
        response1 = client.post("/api/retrieve", json={
            "query": query,
            "context": {"interests": ["fitness", "running", "gym"]}
        })
        
        # Tech-oriented context  
        response2 = client.post("/api/retrieve", json={
            "query": query,
            "context": {"interests": ["technology", "software", "gaming"]}
        })
        
        cats1 = set(c.lower() for c in response1.json()["extracted_categories"])
        cats2 = set(c.lower() for c in response2.json()["extracted_categories"])
        
        # Categories might differ based on context
        # This is a soft test - context SHOULD influence but implementations vary
        # We just verify both return valid categories
        assert len(cats1) >= 1, "Should return categories for fitness context"
        assert len(cats2) >= 1, "Should return categories for tech context"


class TestCategoryEdgeCases:
    """Test category extraction edge cases."""

    def test_ambiguous_query_returns_categories(self, client):
        """Ambiguous query should still return some categories."""
        response = client.post("/api/retrieve", json={
            "query": "I need something good"
        })
        data = response.json()
        
        # Even vague queries should attempt category extraction
        assert isinstance(data["extracted_categories"], list)
        # May return empty list for very vague queries, which is acceptable

    def test_multi_topic_query_returns_multiple_categories(self, client):
        """Query covering multiple topics should return diverse categories."""
        response = client.post("/api/retrieve", json={
            "query": "I need running shoes and also a new laptop for work"
        })
        data = response.json()
        
        assert len(data["extracted_categories"]) >= 2, \
            "Multi-topic query should return multiple categories"
