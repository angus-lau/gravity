import pytest

from app.retrieval.ranker import (
    _age_boost, _gender_boost, _interest_boost, _location_boost,
    AGE_BOOST, GENDER_BOOST, INTEREST_BOOST, LOCATION_BOOST,
)


class TestLocationAliasNormalization:

    def test_nyc_matches_new_york(self):
        assert _location_boost("NYC", ["New York, NY"]) == LOCATION_BOOST

    def test_la_matches_los_angeles(self):
        assert _location_boost("LA", ["Los Angeles, CA"]) == LOCATION_BOOST

    def test_sf_matches_san_francisco(self):
        assert _location_boost("SF", ["San Francisco, CA"]) == LOCATION_BOOST

    def test_unknown_city_falls_back_to_substring(self):
        assert _location_boost("San Diego", ["San Diego, CA"]) == LOCATION_BOOST

    def test_normalization_is_case_insensitive(self):
        assert _location_boost("nyc", ["New York, NY"]) == LOCATION_BOOST
        assert _location_boost("Nyc", ["New York, NY"]) == LOCATION_BOOST
        assert _location_boost("NYC", ["new york, ny"]) == LOCATION_BOOST

    def test_no_match_returns_zero(self):
        assert _location_boost("NYC", ["Los Angeles, CA"]) == 0.0

    def test_empty_inputs_return_zero(self):
        assert _location_boost(None, ["New York, NY"]) == 0.0
        assert _location_boost("NYC", []) == 0.0

    def test_whitespace_padding_user_input(self):
        assert _location_boost("  NYC  ", ["New York, NY"]) == LOCATION_BOOST

    def test_whitespace_padding_campaign_input(self):
        assert _location_boost("NYC", ["  New York, NY  "]) == LOCATION_BOOST

    def test_whitespace_padding_both_sides(self):
        assert _location_boost("  SF  ", ["  San Francisco, CA  "]) == LOCATION_BOOST

    def test_state_abbreviation_matches(self):
        assert _location_boost("IL", ["Chicago, IL"]) == LOCATION_BOOST
        assert _location_boost("FL", ["Miami, FL"]) == LOCATION_BOOST
        assert _location_boost("GA", ["Atlanta, GA"]) == LOCATION_BOOST

    def test_state_full_name_matches(self):
        assert _location_boost("Florida", ["Miami, FL"]) == LOCATION_BOOST
        assert _location_boost("Illinois", ["Chicago, IL"]) == LOCATION_BOOST
        assert _location_boost("Colorado", ["Denver, CO"]) == LOCATION_BOOST

    def test_state_abbreviation_matches_multi_city_states(self):
        assert _location_boost("CA", ["San Francisco, CA"]) == LOCATION_BOOST
        assert _location_boost("CA", ["Los Angeles, CA"]) == LOCATION_BOOST
        assert _location_boost("TX", ["Austin, TX"]) == LOCATION_BOOST

    def test_state_full_name_matches_multi_city_states(self):
        assert _location_boost("California", ["San Francisco, CA"]) == LOCATION_BOOST
        assert _location_boost("California", ["Los Angeles, CA"]) == LOCATION_BOOST
        assert _location_boost("Texas", ["Austin, TX"]) == LOCATION_BOOST

    def test_state_mismatch_returns_zero(self):
        assert _location_boost("CA", ["Chicago, IL"]) == 0.0
        assert _location_boost("Texas", ["Miami, FL"]) == 0.0

    def test_state_extraction_case_insensitive(self):
        assert _location_boost("california", ["San Francisco, CA"]) == LOCATION_BOOST
        assert _location_boost("CALIFORNIA", ["San Francisco, CA"]) == LOCATION_BOOST
        assert _location_boost("ca", ["San Francisco, CA"]) == LOCATION_BOOST


class TestAgeBoost:

    def test_age_in_range(self):
        assert _age_boost(25, [18, 35]) == AGE_BOOST

    def test_age_at_lower_bound(self):
        assert _age_boost(18, [18, 35]) == AGE_BOOST

    def test_age_at_upper_bound(self):
        assert _age_boost(35, [18, 35]) == AGE_BOOST

    def test_age_below_range(self):
        assert _age_boost(17, [18, 35]) == 0.0

    def test_age_above_range(self):
        assert _age_boost(36, [18, 35]) == 0.0

    def test_age_none(self):
        assert _age_boost(None, [18, 35]) == 0.0

    def test_age_range_none(self):
        assert _age_boost(25, None) == 0.0

    def test_age_range_empty(self):
        assert _age_boost(25, []) == 0.0

    def test_age_range_malformed_single(self):
        assert _age_boost(25, [18]) == 0.0

    def test_age_range_malformed_triple(self):
        assert _age_boost(25, [18, 35, 50]) == 0.0

    def test_age_range_reversed(self):
        # [65, 18] is malformed — no age satisfies 65 <= age <= 18
        assert _age_boost(25, [65, 18]) == 0.0


class TestGenderBoost:

    def test_gender_match_in_list(self):
        assert _gender_boost("female", ["female", "male"]) == GENDER_BOOST

    def test_gender_no_match_in_list(self):
        assert _gender_boost("female", ["male"]) == 0.0

    def test_gender_all_string(self):
        assert _gender_boost("female", "all") == GENDER_BOOST

    def test_gender_all_case_insensitive(self):
        assert _gender_boost("female", "ALL") == GENDER_BOOST
        assert _gender_boost("female", "All") == GENDER_BOOST

    def test_gender_match_single_string(self):
        assert _gender_boost("male", "male") == GENDER_BOOST

    def test_gender_no_match_single_string(self):
        assert _gender_boost("female", "male") == 0.0

    def test_gender_case_insensitive(self):
        assert _gender_boost("Female", ["FEMALE", "MALE"]) == GENDER_BOOST

    def test_gender_none_user(self):
        assert _gender_boost(None, ["female"]) == 0.0

    def test_gender_none_campaign(self):
        assert _gender_boost("female", None) == 0.0

    def test_gender_both_none(self):
        assert _gender_boost(None, None) == 0.0

    def test_gender_empty_list(self):
        assert _gender_boost("female", []) == 0.0


class TestInterestBoost:

    def test_single_overlap(self):
        user = frozenset(["fitness", "cooking"])
        assert _interest_boost(user, ["fitness", "travel"]) == INTEREST_BOOST

    def test_multiple_overlaps(self):
        user = frozenset(["fitness", "cooking", "travel"])
        assert _interest_boost(user, ["fitness", "travel", "music"]) == 2 * INTEREST_BOOST

    def test_no_overlap(self):
        user = frozenset(["fitness"])
        assert _interest_boost(user, ["cooking", "travel"]) == 0.0

    def test_capped_at_015(self):
        user = frozenset(["a", "b", "c", "d", "e"])
        assert _interest_boost(user, ["a", "b", "c", "d", "e"]) == 0.15

    def test_case_insensitive(self):
        user = frozenset(["fitness"])
        assert _interest_boost(user, ["FITNESS"]) == INTEREST_BOOST

    def test_none_user(self):
        assert _interest_boost(None, ["fitness"]) == 0.0

    def test_empty_campaign(self):
        user = frozenset(["fitness"])
        assert _interest_boost(user, []) == 0.0

    def test_empty_frozenset(self):
        assert _interest_boost(frozenset(), ["fitness"]) == 0.0


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
