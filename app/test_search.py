from __future__ import annotations

from app.search import SearchHit, Searcher


def test_search_hybrid_uses_rrf_rank_1_based() -> None:
    searcher = Searcher()
    kw_hits = [
        SearchHit("doc_a", "A", "kw first", 10.0),
        SearchHit("doc_b", "B", "kw second", 9.0),
    ]
    sem_hits = [
        SearchHit("doc_b", "B", "sem first", 0.9),
        SearchHit("doc_c", "C", "sem second", 0.8),
    ]

    searcher._search_keyword = lambda query, top_k: kw_hits[:top_k]  # type: ignore[method-assign]
    searcher._search_semantic = lambda query, top_k: sem_hits[:top_k]  # type: ignore[method-assign]

    fused = searcher._search_hybrid("autoscaling", top_k=3, rrf_k=60)

    assert [hit.doc_id for hit in fused] == ["doc_b", "doc_a", "doc_c"]
    assert fused[0].score > fused[1].score > fused[2].score
