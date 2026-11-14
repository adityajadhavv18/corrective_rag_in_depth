"""CRAG's retrieval evaluator: gpt-4o-mini reads the ORIGINAL question and one passage (the header_match text the
answer step reads) and scores how much the passage helps, 0-10. The paper's evaluator is a fine-tuned T5-large; ours
is an LLM because a classifier needs infra we don't have (Phase 0). decide() and grade() are the only things the rest
of CRAG calls, so a classifier can replace score() later.

decide() turns the passage scores into CRAG's action for the question:
    CORRECT     some passage scores >= upper
    INCORRECT   every passage scores < lower
    AMBIGUOUS   anything else

grade() scores a batch at once (Phase 3): uncached calls run 8 at a time, and the batch counts as long as its
slowest call in the cached time, as it would live. llm_cache.chat is not thread-safe, so warm() fills its cache
from the threads and writes it once.
"""

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import crag_bridge  # noqa: F401
import llm_cache
from eval_harness import settings, traced_openai
from llm_cache import CACHED_MS, Reply, chat

PROMPT = """You check whether a passage from a cricket library helps answer a question.

Question: {question}

Passage:
{passage}

How much does this passage help answer the question? Score it from 0 to 10:
10 = it states the full answer
5 = it gives one part of the answer, or one step towards it
0 = it does not help
Reply with the number only."""

UPPER, LOWER = 7, 3  # Phase 2: the grader only says 0, 5 or 10, so 10 = CORRECT, all 0 = INCORRECT, else AMBIGUOUS
MAX_TOKENS = 3
THREADS = 8


def score(question: str, passage: str) -> tuple[int, Reply]:
    reply = chat(PROMPT.format(question=question, passage=passage), max_tokens=MAX_TOKENS)
    m = re.search(r"\d+", reply.text)
    return (min(int(m.group()), 10) if m else 0), reply


def action(scores: list[int], upper: int = UPPER, lower: int = LOWER) -> str:
    if any(s >= upper for s in scores):
        return "CORRECT"
    return "INCORRECT" if all(s < lower for s in scores) else "AMBIGUOUS"


def decide(question: str, passages: list[str], upper: int = UPPER, lower: int = LOWER) -> tuple[str, list[int], list[Reply]]:
    scores, replies = grade(question, passages)
    return action(scores, upper, lower), scores, replies


def warm(prompts: list[str]) -> None:
    """Put every prompt chat() hasn't seen into its cache (same key as chat), THREADS calls at a time."""
    model, cache = settings.llm_model, llm_cache._load()
    key = lambda p: hashlib.sha1(json.dumps([model, 0.0, MAX_TOKENS, p]).encode()).hexdigest()
    todo = list({key(p): p for p in prompts if key(p) not in cache}.items())

    def call(prompt: str) -> dict:
        start = time.perf_counter()
        resp = traced_openai().chat.completions.create(model=model, temperature=0.0, max_tokens=MAX_TOKENS,
                                                       messages=[{"role": "user", "content": prompt}])
        return Reply(resp.choices[0].message.content.strip(), resp.usage.prompt_tokens,
                     resp.usage.completion_tokens, (time.perf_counter() - start) * 1000).__dict__

    with ThreadPoolExecutor(THREADS) as pool:
        for (k, _), reply in zip(todo, pool.map(lambda kp: call(kp[1]), todo)):
            cache[k] = reply
    if todo:
        llm_cache.CACHE.write_text(json.dumps(cache, ensure_ascii=False))


def grade(question: str, passages: list[str]) -> tuple[list[int], list[Reply]]:
    warm([PROMPT.format(question=question, passage=p) for p in passages])
    scored = [score(question, p) for p in passages]  # all cache hits now
    replies = [r for _, r in scored]
    if replies:  # parallel: the batch takes as long as its slowest call, not the sum
        CACHED_MS["ms"] -= sum(r.ms for r in replies) - max(r.ms for r in replies)
    return [s for s, _ in scored], replies
