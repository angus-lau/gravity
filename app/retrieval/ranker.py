from app.retrieval.faiss_index import get_campaign_index
from app.schemas import Campaign, UserContext

LOCATION_BOOST = 0.10
AGE_BOOST = 0.05
INTEREST_BOOST = 0.05
GENDER_BOOST = 0.02

_LOCATION_ALIASES = {
    # New York
    "nyc": "new york, ny", "new york city": "new york, ny", "manhattan": "new york, ny",
    "new york": "new york, ny", "brooklyn": "new york, ny",
    # Los Angeles
    "la": "los angeles, ca", "los angeles": "los angeles, ca",
    # San Francisco
    "sf": "san francisco, ca", "san fran": "san francisco, ca",
    "san francisco": "san francisco, ca",
    # Chicago
    "chi-town": "chicago, il", "chicago": "chicago, il",
    # Miami
    "miami": "miami, fl",
    # Seattle
    "seattle": "seattle, wa",
    # Austin
    "austin": "austin, tx",
    # Denver
    "denver": "denver, co",
    # Boston
    "boston": "boston, ma",
    # Atlanta
    "atl": "atlanta, ga", "atlanta": "atlanta, ga",
    # Phoenix
    "phx": "phoenix, az", "phoenix": "phoenix, az",
    # Portland
    "pdx": "portland, or", "portland": "portland, or",
    # Additional cities
    "las vegas": "las vegas, nv", "vegas": "las vegas, nv",
    "dc": "washington, dc", "washington dc": "washington, dc",
    "philly": "philadelphia, pa", "philadelphia": "philadelphia, pa",
    "dallas": "dallas, tx", "dfw": "dallas, tx",
    "houston": "houston, tx",
}

_STATE_NAMES = {
    "california": "ca", "texas": "tx", "illinois": "il",
    "florida": "fl", "washington": "wa", "colorado": "co",
    "massachusetts": "ma", "georgia": "ga", "arizona": "az",
    "oregon": "or", "nevada": "nv", "new york state": "ny",
    "pennsylvania": "pa",
}


def _normalize_location(loc: str) -> str:
    loc_lower = loc.lower().strip()
    return _LOCATION_ALIASES.get(loc_lower, loc_lower)


def _extract_state(loc: str) -> str | None:
    loc_lower = loc.lower().strip()
    if loc_lower in _STATE_NAMES:
        return _STATE_NAMES[loc_lower]
    if len(loc_lower) == 2 and loc_lower.isalpha():
        return loc_lower
    parts = loc_lower.split(",")
    if len(parts) == 2:
        return parts[1].strip()
    return None


def rerank(
    candidates: list[tuple[str, float]],
    context: UserContext | None,
    top_k: int = 1000,
) -> list[Campaign]:
    index = get_campaign_index()

    # Batch-fetch all campaign data upfront instead of one-by-one
    campaign_data_map: dict[str, dict] = {}
    for campaign_id, _ in candidates:
        if campaign_id not in campaign_data_map:
            data = index.get_campaign(campaign_id)
            if data is not None:
                campaign_data_map[campaign_id] = data

    # Pre-compute user interests set once (avoids rebuilding per candidate)
    user_interest_set: frozenset[str] | None = None
    if context and context.interests:
        user_interest_set = frozenset(i.lower() for i in context.interests)

    scored = []
    for campaign_id, base_score in candidates:
        campaign_data = campaign_data_map.get(campaign_id)
        if not campaign_data:
            continue

        boost = 0.0
        if context:
            targeting = campaign_data.get("targeting", {})
            boost += _location_boost(context.location, targeting.get("locations", []))
            boost += _age_boost(context.age, targeting.get("age_range"))
            boost += _interest_boost(user_interest_set, targeting.get("interests", []))
            boost += _gender_boost(context.gender, targeting.get("genders"))

        scored.append((campaign_data, base_score + boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Normalize scores to [0, 1] after sorting to preserve ranking signal
    max_score = scored[0][1] if scored else 1.0
    if max_score <= 0:
        max_score = 1.0

    # Use model_construct to skip Pydantic validation for performance
    return [
        Campaign.model_construct(
            campaign_id=c["campaign_id"],
            relevance_score=round(score / max_score, 4),
            advertiser=c.get("advertiser"),
            title=c.get("title"),
            categories=c.get("categories"),
        )
        for c, score in scored[:top_k]
    ]


def _location_boost(user_loc: str | None, campaign_locs: list[str]) -> float:
    if not user_loc or not campaign_locs:
        return 0.0
    user_normalized = _normalize_location(user_loc)
    for loc in campaign_locs:
        campaign_normalized = _normalize_location(loc)
        if user_normalized == campaign_normalized:
            return LOCATION_BOOST
    # State-level matching (more precise than substring)
    user_state = _extract_state(user_loc)
    if user_state:
        for loc in campaign_locs:
            campaign_state = _extract_state(loc)
            if campaign_state and user_state == campaign_state:
                return LOCATION_BOOST
        # User input is a state name/abbrev — don't fall through to substring
        # matching, which would produce false positives (e.g. "CA" in "Chicago")
        return 0.0
    # Fallback: substring matching for partial matches
    for loc in campaign_locs:
        campaign_normalized = _normalize_location(loc)
        if user_normalized in campaign_normalized or campaign_normalized in user_normalized:
            return LOCATION_BOOST
    return 0.0


def _age_boost(user_age: int | None, age_range: list[int] | None) -> float:
    if user_age is None or not age_range or len(age_range) != 2:
        return 0.0
    if age_range[0] <= user_age <= age_range[1]:
        return AGE_BOOST
    return 0.0


def _interest_boost(user_interest_set: frozenset[str] | None, campaign_interests: list[str]) -> float:
    if not user_interest_set or not campaign_interests:
        return 0.0
    campaign_set = {i.lower() for i in campaign_interests}
    overlap = len(user_interest_set & campaign_set)
    return min(overlap * INTEREST_BOOST, 0.15)


def _gender_boost(user_gender: str | None, campaign_genders: str | list[str] | None) -> float:
    if not user_gender or not campaign_genders:
        return 0.0
    if isinstance(campaign_genders, str) and campaign_genders.lower() == "all":
        return GENDER_BOOST
    if isinstance(campaign_genders, list):
        if user_gender.lower() in [g.lower() for g in campaign_genders]:
            return GENDER_BOOST
    elif user_gender.lower() == campaign_genders.lower():
        return GENDER_BOOST
    return 0.0
