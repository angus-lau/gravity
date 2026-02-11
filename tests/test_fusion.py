import pytest

from app.retrieval.fusion import reciprocal_rank_fusion


class TestReciprocalRankFusion:

    def test_single_list_passthrough(self):
        results = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        fused = reciprocal_rank_fusion([results])
        assert len(fused) == 3
        # Order should be preserved
        assert fused[0][0] == "a"
        assert fused[1][0] == "b"
        assert fused[2][0] == "c"

    def test_two_lists_overlap_ranks_higher(self):
        """Items appearing in both lists should rank higher than single-list items."""
        list1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        list2 = [("b", 0.95), ("d", 0.85), ("a", 0.75)]

        fused = reciprocal_rank_fusion([list1, list2])

        # Both 'a' and 'b' appear in both lists, should rank highest
        top_ids = [cid for cid, _ in fused[:2]]
        assert "a" in top_ids or "b" in top_ids

    def test_scores_normalized(self):
        """Fused scores should be in [0, 1] range."""
        list1 = [("a", 0.9), ("b", 0.8)]
        list2 = [("c", 0.7), ("d", 0.6)]

        fused = reciprocal_rank_fusion([list1, list2])
        for _cid, score in fused:
            assert 0.0 <= score <= 1.0

        # Top result should have score 1.0 (normalized)
        assert fused[0][1] == 1.0

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_top_k_limit(self):
        list1 = [(f"item_{i}", 1.0 - i * 0.01) for i in range(100)]
        fused = reciprocal_rank_fusion([list1], top_k=10)
        assert len(fused) == 10

    def test_three_lists_fusion(self):
        """Three-way fusion should combine all sources."""
        faiss = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        bm25 = [("b", 5.0), ("d", 4.0), ("a", 3.0)]
        clip = [("c", 0.6), ("a", 0.5), ("e", 0.4)]

        fused = reciprocal_rank_fusion([faiss, bm25, clip])

        # 'a' appears in all 3 lists — should rank highest
        assert fused[0][0] == "a"

    def test_one_empty_one_populated(self):
        """One empty list shouldn't break fusion."""
        list1 = [("a", 0.9), ("b", 0.8)]
        fused = reciprocal_rank_fusion([list1, []])
        assert len(fused) == 2
        assert fused[0][0] == "a"

    def test_duplicate_ids_within_single_list(self):
        """Duplicate IDs in a single list accumulate RRF scores for both ranks."""
        list1 = [("a", 0.9), ("a", 0.8)]
        fused = reciprocal_rank_fusion([list1])
        # 'a' appears twice, gets score for rank 0 and rank 1
        assert len(fused) == 1
        assert fused[0][0] == "a"

    def test_all_same_id(self):
        """All entries being the same ID should produce a single result."""
        list1 = [("x", 0.5)]
        list2 = [("x", 0.5)]
        list3 = [("x", 0.5)]
        fused = reciprocal_rank_fusion([list1, list2, list3])
        assert len(fused) == 1
        assert fused[0][0] == "x"
        assert fused[0][1] == 1.0  # normalized to 1.0

    def test_large_k_parameter(self):
        """Custom k parameter changes ranking behavior."""
        list1 = [("a", 0.9), ("b", 0.8)]
        list2 = [("b", 0.9), ("a", 0.8)]
        fused_default = reciprocal_rank_fusion([list1, list2], k=60)
        fused_high_k = reciprocal_rank_fusion([list1, list2], k=1000)
        # Both should have same ordering (a and b tie-break same way)
        # but scores differ due to different k
        assert len(fused_default) == len(fused_high_k) == 2
