"""Phase 0: the "other library" CRAG falls back to when the main index has nothing good.

Contextual #2 downloaded 149 cricket Wikipedia articles (pinned revisions) and indexed 140: the corpus was capped near
5,000 chunks, so the 9 least-cited player biographies were left out (manifest.json, include = false). They become
the fallback collection here, built exactly like the main one f6a searches (cricket_header_match_note):

    Contextual's chunker (300/50 tokens, never across a heading) -> free label (article › section, match box)
    -> section-mode LLM note (Contextual's notes.py, prompt v2) -> dense bge-small + BM25 + SPLADE vectors

The main collection is never touched: Contextual's ingest() deletes and rewrites the collection its variant names, so
it is not called; its pieces (index text, embedding caches, vectors) are, and the target name is checked. The notes
and embeddings land in Contextual's caches next to the main index's.

Run from corrective_rag_in_depth/ (Qdrant running, OpenAI key sourced for the notes, ~$0.15 once):
    uv run corrective/fallback_index.py
"""

import json
import time
from pathlib import Path

import crag_bridge  # noqa: F401
import chunks
import ingest
import notes
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from retrieve import BM25, DENSE, SPLADE, dense_search, sparse_search

FALLBACK = "cricket_fallback_header_match_note"
MAIN = ingest.collection("header_match_note")
RESULTS = Path(__file__).resolve().parent / "results" / "phase0.json"
SANITY = ["Cheteshwar Pujara first-class debut", "Rishabh Pant car accident", "Kane Williamson captaincy",
          "Kevin Pietersen switch-hit", "Travis Head century"]


def left_out() -> list[str]:
    return [e["doc_id"] for e in json.loads(chunks.MANIFEST.read_text()) if not e["include"]]


def fallback_chunks(docs: list[str]) -> list[chunks.Chunk]:
    """Contextual's load_chunks, run over the left-out articles (it reads included_docs by name at call time)."""
    original = chunks.included_docs
    chunks.included_docs = lambda: docs
    try:
        return chunks.load_chunks.__wrapped__()
    finally:
        chunks.included_docs = original


def build(cs: list[chunks.Chunk]) -> None:
    assert FALLBACK != MAIN
    texts = [ingest.VARIANTS["header_match_note"](c) for c in cs]
    dense_model = TextEmbedding(chunks.EMBED_MODEL)
    vectors = {
        DENSE: ingest.cached(f"dense_{chunks.EMBED_MODEL}", texts,
                             lambda ts: dense_model.passage_embed(ts, batch_size=8)),
        BM25: list(SparseTextEmbedding(ingest.BM25_MODEL, avg_len=ingest.avg_words(texts)).embed(texts)),
        SPLADE: ingest.cached(f"splade_{ingest.SPLADE_MODEL}", texts,
                              lambda ts: SparseTextEmbedding(ingest.SPLADE_MODEL).embed(ts, batch_size=8)),
    }
    client = QdrantClient(url=ingest.QDRANT_URL, timeout=300)
    if client.collection_exists(FALLBACK):
        client.delete_collection(FALLBACK)
    client.create_collection(
        FALLBACK,
        vectors_config={DENSE: models.VectorParams(size=len(vectors[DENSE][0]), distance=models.Distance.COSINE)},
        sparse_vectors_config={BM25: models.SparseVectorParams(modifier=models.Modifier.IDF),
                               SPLADE: models.SparseVectorParams()},
    )
    client.upload_points(
        FALLBACK,
        points=(models.PointStruct(
            id=ingest.point_id(c.chunk_id),
            vector={DENSE: vectors[DENSE][i].tolist(), BM25: ingest.sparse(vectors[BM25][i]),
                    SPLADE: ingest.sparse(vectors[SPLADE][i])},
            payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "path": list(c.path), "text": c.text})
            for i, c in enumerate(cs)),
        batch_size=32,
        wait=True,
    )


def main():
    docs = left_out()
    cs = fallback_chunks(docs)
    print(f"{len(cs)} chunks from {len(docs)} left-out articles: {', '.join(docs)}")
    start = time.perf_counter()
    usage = notes.add_notes("section", cs)
    print(f"notes: {usage['calls']} new, ${notes.cost(usage):.4f}, {time.perf_counter() - start:.0f}s")
    ingest.section_notes.cache_clear()  # it was loaded before the new notes existed
    start = time.perf_counter()
    build(cs)
    client = QdrantClient(url=ingest.QDRANT_URL)
    stored, main_count = client.count(FALLBACK, exact=True).count, client.count(MAIN, exact=True).count
    print(f"stored {stored} points in {FALLBACK} ({time.perf_counter() - start:.0f}s); {MAIN} still {main_count}")

    sanity = {}
    for q in SANITY:
        hits = {"dense": dense_search(q, 3, FALLBACK), "bm25": sparse_search(q, 3, BM25, FALLBACK)}
        sanity[q] = {k: [h.chunk_id for h in v] for k, v in hits.items()}
        print(f"  {q!r}: dense {[h.chunk_id for h in hits['dense']]}")

    out = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    out["fallback"] = {"collection": FALLBACK, "docs": docs, "chunks": len(cs), "stored": stored,
                       "main_count": main_count, "notes_new": usage["calls"], "notes_cost": notes.cost(usage),
                       "sanity": sanity}
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
