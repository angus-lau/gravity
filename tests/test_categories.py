import pytest


class TestCategoryBasics:

    def test_returns_1_to_10_categories(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        num_categories = len(response.json()["extracted_categories"])
        assert 1 <= num_categories <= 10

    def test_categories_are_strings(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        for cat in response.json()["extracted_categories"]:
            assert isinstance(cat, str)
            assert len(cat) > 0

    def test_categories_are_unique(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        categories = response.json()["extracted_categories"]
        assert len(categories) == len(set(categories))


class TestCategoryRelevance:

    def test_running_query_returns_relevant_categories(self, client):
        response = client.post("/api/retrieve", json={
            "query": "I need new running shoes for marathon training"
        })
        categories = [c.lower() for c in response.json()["extracted_categories"]]
        relevant = ["running", "shoe", "athletic", "fitness", "marathon", "sport"]
        assert any(any(kw in cat for kw in relevant) for cat in categories)

    def test_electronics_query_returns_relevant_categories(self, client):
        response = client.post("/api/retrieve", json={
            "query": "looking for a new laptop for software development"
        })
        categories = [c.lower() for c in response.json()["extracted_categories"]]
        relevant = ["laptop", "computer", "electronic", "tech"]
        assert any(any(kw in cat for kw in relevant) for cat in categories)

    def test_travel_query_returns_relevant_categories(self, client):
        response = client.post("/api/retrieve", json={
            "query": "planning a vacation to Europe, looking for flights and hotels"
        })
        categories = [c.lower() for c in response.json()["extracted_categories"]]
        relevant = ["travel", "flight", "hotel", "vacation", "trip"]
        assert any(any(kw in cat for kw in relevant) for cat in categories)


class TestCategoryGranularity:

    def test_categories_not_too_broad(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        overly_broad = ["everything", "all", "general", "stuff", "things"]
        for cat in response.json()["extracted_categories"]:
            assert cat.lower() not in overly_broad


class TestCategoryEdgeCases:

    def test_ambiguous_query_returns_categories(self, client):
        response = client.post("/api/retrieve", json={"query": "I need something good"})
        assert isinstance(response.json()["extracted_categories"], list)

    def test_multi_topic_query_returns_multiple_categories(self, client):
        response = client.post("/api/retrieve", json={
            "query": "I need running shoes and also a new laptop for work"
        })
        assert len(response.json()["extracted_categories"]) >= 2
