"""Phase 3: CRAG (Yan et al., 2024) between Adaptive's search and its answer step.

Both of Adaptive's search routes end in routes.answer_from(question, passages, ...): single with f6a's top 5, multi
with the IRCoT loop's whole pool. install() swaps that one function for crag_answer_from, so CRAG grades what the
answer step would have read (the loop's pool once, after the loop) and the none route is never graded.

    grade every passage (grader.py: 10 / 5 / 0)  ->  CORRECT (a 10) / INCORRECT (all 0) / AMBIGUOUS
    drop      keep passages scored 5 or 10
    fallback  AMBIGUOUS / INCORRECT: f6a on the 9 left-out biographies (cricket_fallback_header_match_note),
              top 5, graded, keep 5s and 10s
    deep      AMBIGUOUS / INCORRECT: the main library's ranks 6-40 for the original question (6-20 from f6a's
              reranked list, 21-40 from its fused list; no new search), graded, keep 10s only (deep ranks are
              mostly look-alikes)
    strips    every kept passage is cut into strips (sentences and lines, packed to ~250 characters), each graded
              under its article › section and match label; strips scored 5 or 10 are kept
Nothing kept -> "I don't know" without an answer call. Otherwise answer prompt v2 (unchanged) reads what is kept.

CONFIG holds the switches of the step being run; LOG[question] what CRAG did, for crag_eval.py.
"""

import re

import crag_bridge  # noqa: F401
import hyde
import ladder
import routes
from ingest import VARIANTS
from llm_cache import chat

import fallback_index
import grader

CONFIG = {"drop": False, "fallback": False, "deep": False, "strips": False}
LOG: dict[str, dict] = {}
TOP_K = 5
DEEP = (5, 40)  # ranks 6-40
STRIP_CHARS = 250
_original = routes.answer_from
_by_id = ladder._chunks_by_id()  # labelled() reads chunks from here; the fallback's are added so it can label them
_by_id.update({c.chunk_id: c for c in fallback_index.fallback_chunks(fallback_index.left_out())})


def _f5a(collection: str | None = None, rerank: bool = True) -> dict:
    import fusion_ladder  # noqa: F401  (registers f5a)
    from rag_fusion import FUSION
    opts = {**FUSION["f5a"][1], "rerank": rerank}
    return opts | ({"collection": collection} if collection else {})


def fallback_search(question: str) -> list:
    from rag_fusion import fusion_search
    return fusion_search(question, TOP_K, **_f5a(fallback_index.FALLBACK))


def deep_search(question: str, seen: set[str]) -> list:
    """Ranks 6-40 of the original question's f6a lists. Both calls replay what the first search already did
    (cached rewrites and rerank), so their tokens and time are not counted twice."""
    from rag_fusion import fusion_search
    before = dict(hyde.USAGE)
    hits = fusion_search(question, 20, **_f5a())[DEEP[0]:] + fusion_search(question, DEEP[1], **_f5a(rerank=False))[20:]
    hyde.USAGE.update(before)
    out = []
    for h in hits:
        if h.chunk_id not in seen:
            seen = seen | {h.chunk_id}
            out.append(h)
    return out


def strips(chunk) -> list[str]:
    units = [u for line in chunk.text.split("\n") for u in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])", line) if u.strip()]
    out = []
    for u in units:
        if out and len(out[-1]) + len(u) < STRIP_CHARS:
            out[-1] += " " + u
        else:
            out.append(u)
    return out


def header(chunk) -> str:
    return "\n".join([" › ".join(chunk.path), *chunk.match])


def refine(question: str, hits: list) -> tuple[list, list[str], list, dict]:
    """Knowledge refinement: keep only the strips graded 5 or 10, under their passage's header."""
    cut = {h.chunk_id: strips(_by_id[h.chunk_id]) for h in hits}
    batch = [(h, s) for h in hits for s in cut[h.chunk_id]]
    scores, replies = grader.grade(question, [f"{header(_by_id[h.chunk_id])}\n{s}" for h, s in batch])
    kept_hits, texts = [], []
    for h in hits:
        keep = [s for (x, s), sc in zip(batch, scores) if x is h and sc >= grader.LOWER]
        if keep:
            kept_hits.append(h)
            texts.append("\n".join([header(_by_id[h.chunk_id]), *keep]))
    return kept_hits, texts, replies, {"strips": len(batch), "strips_kept": sum(s >= grader.LOWER for s in scores)}


def crag_answer_from(question: str, hits: list, replies: list, answer: str = "v2", notes: str = ""):
    texts = [ladder.labelled(h) for h in hits]
    scores, graded = grader.grade(question, texts)
    act = grader.action(scores)
    log = {"action": act, "scores": scores, "pool": len(hits)}
    kept = [h for h, s in zip(hits, scores) if s >= grader.LOWER] if CONFIG["drop"] else list(hits)
    if act != "CORRECT":
        extra = {}
        if CONFIG["fallback"]:
            extra["fallback"] = (fallback_search(question), grader.LOWER)
        if CONFIG["deep"]:
            extra["deep"] = (deep_search(question, {h.chunk_id for h in hits}), grader.UPPER)
        for name, (found, bar) in extra.items():
            s, r = grader.grade(question, [ladder.labelled(h) for h in found])
            graded += r
            kept += [h for h, x in zip(found, s) if x >= bar]
            log[name] = {"scores": s, "kept": [h.chunk_id for h, x in zip(found, s) if x >= bar]}
    if CONFIG["strips"] and kept:
        kept, kept_texts, r, log["strips"] = refine(question, kept)
        graded += r
    else:
        kept_texts = [ladder.labelled(h) for h in kept]
    log["kept"] = [h.chunk_id for h in kept]
    LOG[question] = log
    if not kept:
        return routes._output(routes.NO_ANSWER, [], [], replies + graded)
    context = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(kept_texts, start=1))
    reply = chat(routes.ANSWERS[answer].format(no_answer=routes.NO_ANSWER, context=context, question=question,
                                               notes=notes))
    return routes._output(reply.text, kept, kept_texts, replies + graded + [reply])


def install(config: dict | None) -> None:
    """config None: Adaptive as it is (3.0). Otherwise CRAG with these switches."""
    LOG.clear()
    if config is None:
        routes.answer_from = _original
    else:
        CONFIG.update({k: False for k in CONFIG} | config)
        routes.answer_from = crag_answer_from
