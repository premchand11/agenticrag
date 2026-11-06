from uuid import UUID

import pytest
from app.retrieval.rrf import reciprocal_rank_fusion


def test_a_chunk_found_by_both_lists_ranks_first():
    both, vector_only, text_only = UUID(int=1), UUID(int=2), UUID(int=3)
    fused = reciprocal_rank_fusion([[both, vector_only], [both, text_only]], k=60, limit=3)
    assert fused[0].chunk_id == both
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 61)
    assert {item.chunk_id for item in fused[1:]} == {vector_only, text_only}
    assert fused[1].score == pytest.approx(1 / 62)


def test_an_earlier_rank_beats_a_later_one():
    first, second = UUID(int=1), UUID(int=2)
    fused = reciprocal_rank_fusion([[first, second]], k=60, limit=2)
    assert [item.chunk_id for item in fused] == [first, second]


def test_limit_drops_the_tail():
    ranks = [UUID(int=value) for value in range(1, 5)]
    fused = reciprocal_rank_fusion([ranks], k=60, limit=2)
    assert [item.chunk_id for item in fused] == ranks[:2]


def test_a_repeated_id_keeps_its_best_rank():
    chunk_id = UUID(int=1)
    fused = reciprocal_rank_fusion([[chunk_id, chunk_id]], k=60, limit=1)
    assert fused[0].score == pytest.approx(1 / 61)
