def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int = 2000,
) -> list[tuple[str, float]]:
    fused_scores: dict[str, float] = {}

    for results in result_lists:
        for rank, (campaign_id, _score) in enumerate(results):
            if campaign_id not in fused_scores:
                fused_scores[campaign_id] = 0.0
            fused_scores[campaign_id] += 1.0 / (k + rank + 1)

    if not fused_scores:
        return []

    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    max_score = sorted_results[0][1] if sorted_results else 1.0
    return [(cid, score / max_score) for cid, score in sorted_results]
