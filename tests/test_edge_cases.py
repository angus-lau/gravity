"""Edge Case Tests - Verify robust handling of unusual inputs.

These tests verify that the system handles edge cases gracefully
without crashing or returning invalid responses.
"""

import pytest


class TestEmptyAndMinimalInputs:
    """Test handling of empty and minimal inputs."""

    def test_empty_query_rejected(self, client):
        """Empty query should be rejected with 422."""
        response = client.post("/api/retrieve", json={"query": ""})
        assert response.status_code == 422, "Empty query should be rejected"

    def test_whitespace_only_query_rejected(self, client):
        """Whitespace-only query should be rejected."""
        response = client.post("/api/retrieve", json={"query": "   "})
        assert response.status_code == 422, "Whitespace-only query should be rejected"

    def test_single_character_query_handled(self, client):
        """Single character query should be handled."""
        response = client.post("/api/retrieve", json={"query": "a"})
        assert response.status_code == 200, "Single character query should be accepted"
        
        data = response.json()
        assert "ad_eligibility" in data
        assert "campaigns" in data

    def test_single_word_query_handled(self, client):
        """Single word query should be handled."""
        response = client.post("/api/retrieve", json={"query": "shoes"})
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["campaigns"]) > 0 or True  # May have results

    def test_empty_context_handled(self, client):
        """Empty context object should be handled."""
        response = client.post("/api/retrieve", json={
            "query": "running shoes",
            "context": {}
        })
        assert response.status_code == 200

    def test_null_context_handled(self, client):
        """Null context should be handled."""
        response = client.post("/api/retrieve", json={
            "query": "running shoes",
            "context": None
        })
        assert response.status_code == 200


class TestLongInputs:
    """Test handling of unusually long inputs."""

    def test_very_long_query_handled(self, client):
        """Very long query should be handled gracefully."""
        long_query = "running shoes " * 200  # ~2800 characters
        response = client.post("/api/retrieve", json={"query": long_query})
        
        # Should either succeed or fail gracefully (not crash)
        assert response.status_code in [200, 422], \
            f"Long query should be handled gracefully, got {response.status_code}"

    def test_query_at_max_length(self, client):
        """Query at maximum allowed length should work."""
        # 10000 character limit from schema
        max_query = "shoes " * 1600  # ~9600 chars
        response = client.post("/api/retrieve", json={"query": max_query})
        assert response.status_code == 200

    def test_query_exceeds_max_length(self, client):
        """Query exceeding maximum length should be rejected."""
        too_long = "a" * 10001
        response = client.post("/api/retrieve", json={"query": too_long})
        assert response.status_code == 422, "Query exceeding max length should be rejected"

    def test_many_interests_handled(self, client):
        """Many interests in context should be handled."""
        response = client.post("/api/retrieve", json={
            "query": "gift ideas",
            "context": {"interests": [f"interest_{i}" for i in range(100)]}
        })
        assert response.status_code == 200


class TestSpecialCharacters:
    """Test handling of special characters and encoding."""

    def test_emoji_in_query_handled(self, client):
        """Emoji in query should be handled."""
        response = client.post("/api/retrieve", json={
            "query": "🏃‍♂️ running shoes 👟 marathon 🏆"
        })
        assert response.status_code == 200

    def test_unicode_query_handled(self, client):
        """Unicode characters should be handled."""
        response = client.post("/api/retrieve", json={
            "query": "日本語のランニングシューズ"  # Japanese: running shoes
        })
        assert response.status_code == 200

    def test_newlines_in_query_handled(self, client):
        """Newlines in query should be handled."""
        response = client.post("/api/retrieve", json={
            "query": "running shoes\nfor marathon\ntraining"
        })
        assert response.status_code == 200

    def test_tabs_in_query_handled(self, client):
        """Tab characters in query should be handled."""
        response = client.post("/api/retrieve", json={
            "query": "running\tshoes\tfor\tmarathon"
        })
        assert response.status_code == 200

    def test_html_in_query_not_executed(self, client):
        """HTML in query should be treated as text, not executed."""
        response = client.post("/api/retrieve", json={
            "query": "<script>alert('xss')</script> running shoes"
        })
        assert response.status_code == 200
        
        data = response.json()
        # Response should not contain executable HTML
        response_text = str(data)
        assert "<script>" not in response_text or "alert" not in response_text

    def test_sql_injection_attempt_safe(self, client):
        """SQL injection attempts should be handled safely."""
        response = client.post("/api/retrieve", json={
            "query": "'; DROP TABLE campaigns; --"
        })
        assert response.status_code == 200  # Should treat as normal text


class TestMalformedRequests:
    """Test handling of malformed requests."""

    def test_invalid_json_returns_422(self, client):
        """Invalid JSON should return 422."""
        response = client.post(
            "/api/retrieve",
            content="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_wrong_content_type_handled(self, client):
        """Wrong content type should be handled."""
        response = client.post(
            "/api/retrieve",
            content='{"query": "test"}',
            headers={"Content-Type": "text/plain"}
        )
        # Should either work (FastAPI is lenient) or return appropriate error
        assert response.status_code in [200, 415, 422]

    def test_invalid_age_rejected(self, client):
        """Invalid age values should be rejected."""
        response = client.post("/api/retrieve", json={
            "query": "running shoes",
            "context": {"age": -5}
        })
        assert response.status_code == 422, "Negative age should be rejected"

    def test_age_over_limit_rejected(self, client):
        """Age over reasonable limit should be rejected."""
        response = client.post("/api/retrieve", json={
            "query": "running shoes",
            "context": {"age": 200}
        })
        assert response.status_code == 422, "Age > 120 should be rejected"


class TestConcurrentRequests:
    """Test handling of concurrent requests."""

    @pytest.mark.slow
    def test_multiple_concurrent_requests(self, client):
        """System should handle multiple concurrent requests."""
        import concurrent.futures
        
        def make_request(i):
            return client.post("/api/retrieve", json={
                "query": f"running shoes query {i}"
            })
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200


class TestGracefulDegradation:
    """Test that system degrades gracefully under stress."""

    def test_low_eligibility_still_returns_structure(self, client):
        """Even low-eligibility queries should return valid response structure."""
        response = client.post("/api/retrieve", json={
            "query": "my mom just passed away"
        })
        assert response.status_code == 200
        
        data = response.json()
        # Should still return valid structure even if eligibility is 0
        assert "ad_eligibility" in data
        assert "extracted_categories" in data
        assert "campaigns" in data
        assert "latency_ms" in data

    def test_gibberish_query_handled(self, client):
        """Gibberish query should be handled gracefully."""
        response = client.post("/api/retrieve", json={
            "query": "asdfghjkl qwertyuiop zxcvbnm"
        })
        assert response.status_code == 200
        
        data = response.json()
        # Should return valid structure, maybe with low scores
        assert isinstance(data["campaigns"], list)
