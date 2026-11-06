from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class FusedRank:
    chunk_id: UUID
    score: float


def reciprocal_rank_fusion(rankings: list[list[UUID]], *, k: int, limit: int) -> list[FusedRank]:
    """Merge best-first lists. A chunk's score is the sum of 1 / (k + rank) from each list."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    scores: dict[UUID, float] = {}
    for ranking in rankings:
        seen: set[UUID] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], str(chunk_id)))
    return [FusedRank(chunk_id=chunk_id, score=scores[chunk_id]) for chunk_id in ordered[:limit]]
