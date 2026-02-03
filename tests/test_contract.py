import pytest


class TestAPIContract:

    def test_endpoint_exists(self, client):
        response = client.post("/api/retrieve", json={"query": "test query"})
        assert response.status_code != 404

    def test_returns_required_fields(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        assert response.status_code == 200
        data = response.json()
        for field in ["ad_eligibility", "extracted_categories", "campaigns", "latency_ms", "metadata"]:
            assert field in data

    def test_eligibility_in_valid_range(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        assert 0.0 <= data["ad_eligibility"] <= 1.0

    def test_categories_is_list_of_strings(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        assert isinstance(data["extracted_categories"], list)
        for cat in data["extracted_categories"]:
            assert isinstance(cat, str)

    def test_returns_up_to_1000_campaigns(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        assert len(data["campaigns"]) <= 1000

    def test_campaigns_have_required_fields(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        for campaign in data["campaigns"][:10]:
            assert "campaign_id" in campaign
            assert "relevance_score" in campaign

    def test_campaign_relevance_scores_valid(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        for campaign in data["campaigns"][:100]:
            assert 0.0 <= campaign["relevance_score"] <= 1.0

    def test_latency_ms_is_positive(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        assert isinstance(data["latency_ms"], (int, float))
        assert data["latency_ms"] >= 0

    def test_metadata_is_dict(self, client, sample_query):
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        assert isinstance(data["metadata"], dict)


class TestLatencyRequirement:

    @pytest.mark.slow
    def test_latency_under_100ms(self, client, sample_query):
        import time
        start = time.perf_counter()
        response = client.post("/api/retrieve", json=sample_query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200
        assert response.json()["latency_ms"] < 100
        assert elapsed_ms < 200  # allow buffer for test overhead


class TestRequestValidation:

    def test_query_is_required(self, client):
        response = client.post("/api/retrieve", json={})
        assert response.status_code == 422

    def test_context_is_optional(self, client):
        response = client.post("/api/retrieve", json={"query": "test"})
        assert response.status_code == 200

    def test_partial_context_accepted(self, client):
        response = client.post("/api/retrieve", json={
            "query": "running shoes",
            "context": {"age": 25}
        })
        assert response.status_code == 200

    def test_extra_context_fields_accepted(self, client):
        response = client.post("/api/retrieve", json={
            "query": "running shoes",
            "context": {"age": 25, "custom_field": "value"}
        })
        assert response.status_code == 200


class TestHealthEndpoint:

    def test_health_endpoint_exists(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
