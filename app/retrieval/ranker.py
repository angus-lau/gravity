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
    campaigns_db = index.campaigns

    has_context = context is not None
    user_loc: str | None = None
    user_loc_norm: str | None = None
    user_loc_state: str | None = None
    user_age: int | None = None
    user_interest_set: frozenset[str] | None = None
    user_gender_lower: str | None = None

    if has_context:
        if context.location:
            user_loc = context.location
            user_loc_norm = _normalize_location(user_loc)
            user_loc_state = _extract_state(user_loc)
        user_age = context.age
        if context.interests:
            user_interest_set = frozenset(i.lower() for i in context.interests)
        if context.gender:
            user_gender_lower = context.gender.lower()

    scored = []
    scored_append = scored.append

    for campaign_id, base_score in candidates:
        campaign_data = campaigns_db.get(campaign_id)
        if campaign_data is None:
            continue

        boost = 0.0

        if has_context:
            targeting = campaign_data.get("targeting")
            if targeting:
                if user_loc_norm:
                    clocs = targeting.get("_locations_normalized")
                    if clocs:
                        if user_loc_norm in clocs:
                            boost += LOCATION_BOOST
                        elif not user_loc_state:
                            for cl in clocs:
                                if user_loc_norm in cl or cl in user_loc_norm:
                                    boost += LOCATION_BOOST
                                    break

                if user_age is not None:
                    age_range = targeting.get("age_range")
                    if age_range and len(age_range) == 2 and age_range[0] <= user_age <= age_range[1]:
                        boost += AGE_BOOST

                if user_interest_set:
                    ci = targeting.get("_interests_lowered")
                    if ci:
                        overlap = len(user_interest_set & ci)
                        if overlap:
                            boost += min(overlap * INTEREST_BOOST, 0.15)

                if user_gender_lower:
                    gl = targeting.get("_genders_lowered")
                    if gl:
                        if user_gender_lower in gl:
                            boost += GENDER_BOOST
                    else:
                        gs = targeting.get("_genders_lowered_str")
                        if gs and (gs == "all" or gs == user_gender_lower):
                            boost += GENDER_BOOST

        scored_append((campaign_data, base_score + boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    max_score = scored[0][1] if scored else 1.0
    if max_score <= 0:
        max_score = 1.0
    inv_max = 1.0 / max_score

    return [
        Campaign.model_construct(
            campaign_id=c["campaign_id"],
            relevance_score=round(score * inv_max, 4),
            advertiser=c.get("advertiser"),
            title=c.get("title"),
            categories=c.get("categories"),
        )
        for c, score in scored[:top_k]
    ]


def _location_boost(
    user_loc: str | None,
    campaign_locs: list[str],
    user_normalized: str | None = None,
    campaign_locs_normalized: list[str] | None = None,
) -> float:
    if not user_loc or not campaign_locs:
        return 0.0
    if user_normalized is None:
        user_normalized = _normalize_location(user_loc)
    if campaign_locs_normalized is None:
        campaign_locs_normalized = [_normalize_location(loc) for loc in campaign_locs]
    for campaign_normalized in campaign_locs_normalized:
        if user_normalized == campaign_normalized:
            return LOCATION_BOOST
    user_state = _extract_state(user_loc)
    if user_state:
        for loc in campaign_locs:
            campaign_state = _extract_state(loc)
            if campaign_state and user_state == campaign_state:
                return LOCATION_BOOST
        return 0.0
    for campaign_normalized in campaign_locs_normalized:
        if user_normalized in campaign_normalized or campaign_normalized in user_normalized:
            return LOCATION_BOOST
    return 0.0


def _age_boost(user_age: int | None, age_range: list[int] | None) -> float:
    if user_age is None or not age_range or len(age_range) != 2:
        return 0.0
    if age_range[0] <= user_age <= age_range[1]:
        return AGE_BOOST
    return 0.0


def _interest_boost(
    user_interest_set: frozenset[str] | None,
    campaign_interests: list[str],
    campaign_interests_lowered: frozenset[str] | None = None,
) -> float:
    if not user_interest_set or not campaign_interests:
        return 0.0
    campaign_set = campaign_interests_lowered if campaign_interests_lowered is not None else frozenset(i.lower() for i in campaign_interests)
    overlap = len(user_interest_set & campaign_set)
    return min(overlap * INTEREST_BOOST, 0.15)


def _gender_boost(
    user_gender: str | None,
    campaign_genders: str | list[str] | None,
    campaign_genders_lowered: list[str] | None = None,
    campaign_genders_lowered_str: str | None = None,
) -> float:
    if not user_gender or not campaign_genders:
        return 0.0
    user_lower = user_gender.lower()
    if isinstance(campaign_genders, str):
        g_lower = campaign_genders_lowered_str if campaign_genders_lowered_str is not None else campaign_genders.lower()
        if g_lower == "all":
            return GENDER_BOOST
        if user_lower == g_lower:
            return GENDER_BOOST
    elif isinstance(campaign_genders, list):
        g_list = campaign_genders_lowered if campaign_genders_lowered is not None else [g.lower() for g in campaign_genders]
        if user_lower in g_list:
            return GENDER_BOOST
    return 0.0
