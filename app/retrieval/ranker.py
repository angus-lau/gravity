from app.retrieval.faiss_index import get_campaign_index
from app.schemas import Campaign, UserContext

LOCATION_BOOST = 0.10
AGE_BOOST = 0.05
INTEREST_BOOST = 0.05
GENDER_BOOST = 0.02


def rerank(
    candidates: list[tuple[str, float]],
    context: UserContext | None,
    top_k: int = 1000,
) -> list[Campaign]:
    index = get_campaign_index()
    scored = []

    for campaign_id, base_score in candidates:
        campaign_data = index.get_campaign(campaign_id)
        if not campaign_data:
            continue

        boost = 0.0
        if context:
            targeting = campaign_data.get("targeting", {})
            boost += _location_boost(context.location, targeting.get("locations", []))
            boost += _age_boost(context.age, targeting.get("age_range"))
            boost += _interest_boost(context.interests, targeting.get("interests", []))
            boost += _gender_boost(context.gender, targeting.get("genders"))

        final_score = min(1.0, base_score + boost)
        scored.append((campaign_data, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Use model_construct to skip Pydantic validation for performance
    return [
        Campaign.model_construct(
            campaign_id=c["campaign_id"],
            relevance_score=round(score, 4),
            advertiser=c.get("advertiser"),
            title=c.get("title"),
            categories=c.get("categories"),
        )
        for c, score in scored[:top_k]
    ]


def _location_boost(user_loc: str | None, campaign_locs: list[str]) -> float:
    if not user_loc or not campaign_locs:
        return 0.0
    user_loc_lower = user_loc.lower()
    for loc in campaign_locs:
        if user_loc_lower in loc.lower() or loc.lower() in user_loc_lower:
            return LOCATION_BOOST
    return 0.0


def _age_boost(user_age: int | None, age_range: list[int] | None) -> float:
    if user_age is None or not age_range or len(age_range) != 2:
        return 0.0
    if age_range[0] <= user_age <= age_range[1]:
        return AGE_BOOST
    return 0.0


def _interest_boost(user_interests: list[str] | None, campaign_interests: list[str]) -> float:
    if not user_interests or not campaign_interests:
        return 0.0
    user_set = {i.lower() for i in user_interests}
    campaign_set = {i.lower() for i in campaign_interests}
    overlap = len(user_set & campaign_set)
    return min(overlap * INTEREST_BOOST, 0.15)


def _gender_boost(user_gender: str | None, campaign_genders: str | list[str] | None) -> float:
    if not user_gender or not campaign_genders:
        return 0.0
    if campaign_genders == "all":
        return GENDER_BOOST
    if isinstance(campaign_genders, list):
        if user_gender.lower() in [g.lower() for g in campaign_genders]:
            return GENDER_BOOST
    elif user_gender.lower() == campaign_genders.lower():
        return GENDER_BOOST
    return 0.0
