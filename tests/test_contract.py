"""API Contract Tests - Verify the API meets its specification.

These tests verify that the API response format matches the contract
defined in the take-home specification, regardless of implementation details.
"""

import pytest


class TestAPIContract:
    """Test that the API response matches the required contract."""

    def test_endpoint_exists(self, client):
        """POST /api/retrieve endpoint must exist."""
        response = client.post("/api/retrieve", json={"query": "test query"})
        assert response.status_code != 404, "Endpoint /api/retrieve not found"

    def test_returns_required_fields(self, client, sample_query):
        """Response must contain all required fields."""
        response = client.post("/api/retrieve", json=sample_query)
        assert response.status_code == 200
        
        data = response.json()
        required_fields = ["ad_eligibility", "extracted_categories", "campaigns", "latency_ms", "metadata"]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_eligibility_in_valid_range(self, client, sample_query):
        """ad_eligibility must be between 0.0 and 1.0."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        assert 0.0 <= data["ad_eligibility"] <= 1.0, \
            f"ad_eligibility {data['ad_eligibility']} not in range [0.0, 1.0]"

    def test_categories_is_list_of_strings(self, client, sample_query):
        """extracted_categories must be a list of strings."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        assert isinstance(data["extracted_categories"], list), \
            "extracted_categories must be a list"
        
        for cat in data["extracted_categories"]:
            assert isinstance(cat, str), f"Category {cat} is not a string"

    def test_returns_up_to_1000_campaigns(self, client, sample_query):
        """Should return at most 1000 campaigns."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        assert len(data["campaigns"]) <= 1000, \
            f"Returned {len(data['campaigns'])} campaigns, expected <= 1000"

    def test_campaigns_have_required_fields(self, client, sample_query):
        """Each campaign must have campaign_id and relevance_score."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        for i, campaign in enumerate(data["campaigns"][:10]):  # Check first 10
            assert "campaign_id" in campaign, f"Campaign {i} missing campaign_id"
            assert "relevance_score" in campaign, f"Campaign {i} missing relevance_score"

    def test_campaign_relevance_scores_valid(self, client, sample_query):
        """Campaign relevance_score must be between 0.0 and 1.0."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        for campaign in data["campaigns"][:100]:  # Check first 100
            score = campaign["relevance_score"]
            assert 0.0 <= score <= 1.0, \
                f"Campaign {campaign['campaign_id']} has invalid score: {score}"

    def test_latency_ms_is_positive_number(self, client, sample_query):
        """latency_ms must be a positive number."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        assert isinstance(data["latency_ms"], (int, float)), \
            "latency_ms must be a number"
        assert data["latency_ms"] >= 0, "latency_ms must be non-negative"

    def test_metadata_is_dict(self, client, sample_query):
        """metadata must be a dictionary."""
        response = client.post("/api/retrieve", json=sample_query)
        data = response.json()
        
        assert isinstance(data["metadata"], dict), "metadata must be a dictionary"


class TestLatencyRequirement:
    """Test the critical latency requirement."""

    @pytest.mark.slow
    def test_latency_under_100ms(self, client, sample_query):
        """Response must complete in under 100ms (p95 requirement).
        
        Note: This tests a single request. The benchmark script tests p95.
        """
        import time
        
        start = time.perf_counter()
        response = client.post("/api/retrieve", json=sample_query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert response.status_code == 200
        
        # Internal latency (from response)
        internal_latency = response.json()["latency_ms"]
        assert internal_latency < 100, \
            f"Internal latency {internal_latency}ms exceeds 100ms limit"
        
        # Total round-trip (for local testing, should be very fast)
        # Allow some buffer for test overhead
        assert elapsed_ms < 200, \
            f"Round-trip {elapsed_ms:.1f}ms is too slow (internal: {internal_latency}ms)"


class TestRequestValidation:
    """Test request validation behavior."""

    def test_query_is_required(self, client):
        """Request without query should fail."""
        response = client.post("/api/retrieve", json={})
        assert response.status_code == 422, "Should reject request without query"

    def test_context_is_optional(self, client):
        """Request without context should succeed."""
        response = client.post("/api/retrieve", json={"query": "test"})
        assert response.status_code == 200, "Should accept request without context"

    def test_partial_context_accepted(self, client):
        """Request with partial context should succeed."""
        response = client.post("/api/retrieve", json={
            "query": "running shoes",
            "context": {"age": 25}  # Only age, no other fields
        })
        assert response.status_code == 200, "Should accept partial context"

    def test_extra_context_fields_accepted(self, client):
        """Request with extra context fields should be accepted."""
        response = client.post("/api/retrieve", json={
            "query": "running shoes",
            "context": {
                "age": 25,
                "custom_field": "custom_value",
                "device": "mobile"
            }
        })
        assert response.status_code == 200, "Should accept extra context fields"


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint_exists(self, client):
        """GET /health endpoint must exist."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Health endpoint returns status information."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
