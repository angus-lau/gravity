"""Reciprocal Rank Fusion for combining multiple retrieval result lists."""


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int = 2000,
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score: score(d) = sum(1 / (k + rank_i(d))) for each list i
    Score-distribution-agnostic — no normalization needed between sources.

    Args:
        result_lists: List of result lists, each containing (campaign_id, score) tuples
            sorted by score descending.
        k: RRF constant (default 60, standard value from the original paper).
        top_k: Maximum number of results to return.

    Returns:
        Fused results as (campaign_id, normalized_score) sorted by score descending.
    """
    fused_scores: dict[str, float] = {}

    for results in result_lists:
        for rank, (campaign_id, _score) in enumerate(results):
            if campaign_id not in fused_scores:
                fused_scores[campaign_id] = 0.0
            fused_scores[campaign_id] += 1.0 / (k + rank + 1)  # rank is 0-indexed, +1 for 1-indexed

    if not fused_scores:
        return []

    # Sort by fused score descending
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Normalize scores to [0, 1] for compatibility with ranker
    max_score = sorted_results[0][1] if sorted_results else 1.0
    return [(cid, score / max_score) for cid, score in sorted_results]
