import pytest


class TestEmptyInputs:

    def test_empty_query_rejected(self, client):
        response = client.post("/api/retrieve", json={"query": ""})
        assert response.status_code == 422

    def test_whitespace_only_rejected(self, client):
        response = client.post("/api/retrieve", json={"query": "   "})
        assert response.status_code == 422

    def test_single_character_accepted(self, client):
        response = client.post("/api/retrieve", json={"query": "a"})
        assert response.status_code == 200

    def test_empty_context_accepted(self, client):
        response = client.post("/api/retrieve", json={"query": "shoes", "context": {}})
        assert response.status_code == 200

    def test_null_context_accepted(self, client):
        response = client.post("/api/retrieve", json={"query": "shoes", "context": None})
        assert response.status_code == 200


class TestLongInputs:

    def test_very_long_query_handled(self, client):
        response = client.post("/api/retrieve", json={"query": "shoes " * 200})
        assert response.status_code in [200, 422]

    def test_query_exceeds_max_rejected(self, client):
        response = client.post("/api/retrieve", json={"query": "a" * 10001})
        assert response.status_code == 422

    def test_many_interests_handled(self, client):
        response = client.post("/api/retrieve", json={
            "query": "gift ideas",
            "context": {"interests": [f"interest_{i}" for i in range(100)]}
        })
        assert response.status_code == 200


class TestSpecialCharacters:

    def test_emoji_handled(self, client):
        response = client.post("/api/retrieve", json={"query": "🏃 running shoes 👟"})
        assert response.status_code == 200

    def test_unicode_handled(self, client):
        response = client.post("/api/retrieve", json={"query": "日本語のランニングシューズ"})
        assert response.status_code == 200

    def test_newlines_handled(self, client):
        response = client.post("/api/retrieve", json={"query": "running\nshoes\nmarathon"})
        assert response.status_code == 200

    def test_html_not_executed(self, client):
        response = client.post("/api/retrieve", json={
            "query": "<script>alert('xss')</script> shoes"
        })
        assert response.status_code == 200

    def test_sql_injection_safe(self, client):
        response = client.post("/api/retrieve", json={"query": "'; DROP TABLE campaigns; --"})
        assert response.status_code == 200


class TestMalformedRequests:

    def test_invalid_json_rejected(self, client):
        response = client.post(
            "/api/retrieve",
            content="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_invalid_age_rejected(self, client):
        response = client.post("/api/retrieve", json={
            "query": "shoes",
            "context": {"age": -5}
        })
        assert response.status_code == 422

    def test_age_over_limit_rejected(self, client):
        response = client.post("/api/retrieve", json={
            "query": "shoes",
            "context": {"age": 200}
        })
        assert response.status_code == 422


class TestConcurrentRequests:

    @pytest.mark.slow
    def test_concurrent_requests_succeed(self, client):
        import concurrent.futures

        def make_request(i):
            return client.post("/api/retrieve", json={"query": f"shoes {i}"})

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            responses = list(ex.map(make_request, range(10)))

        assert all(r.status_code == 200 for r in responses)


class TestGracefulDegradation:

    def test_low_eligibility_still_valid_structure(self, client):
        response = client.post("/api/retrieve", json={"query": "my mom just passed away"})
        assert response.status_code == 200
        data = response.json()
        assert "ad_eligibility" in data
        assert "campaigns" in data

    def test_gibberish_handled(self, client):
        response = client.post("/api/retrieve", json={"query": "asdfghjkl qwertyuiop"})
        assert response.status_code == 200
        assert isinstance(response.json()["campaigns"], list)
