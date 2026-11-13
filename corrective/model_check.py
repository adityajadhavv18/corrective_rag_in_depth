"""Phase 0: can the decision models run on this 8 GB Mac, and how fast? No accuracy claims yet (Phase 3).

    laya   convaiinnovations/laya (English root: ModernBERT-large, 421M), the pip package `laya`, in this process
    kev    jaredpalmer/kev-0.8b (Qwen3.5-0.8B-Base + LoRA + pointer head), served by the official repo's kev.serve
           (cloned at tag kev-1.0 in RAGS/kev; not on PyPI: `pip install kev` is an unrelated package), over HTTP
    jev    not used: hosted only, no API key

20 sample decisions: 10 tune questions, each with one passage that holds its golden evidence (relevant) and one
passage from another article (irrelevant), as the answer step sees them (article › section, match box, text). One
choice question per pair, relevant / partial / irrelevant, with neutral keys (Laya's card: its yes/no type can
follow its own labels). The hit count is a smoke check on 20 easy pairs, not a measurement.

Run from corrective_rag_in_depth/ (no OpenAI calls):
    uv run corrective/model_check.py laya
    # in RAGS/kev:  uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b@v1.0 --port 8009
    uv run corrective/model_check.py kev
"""

import json
import random
import resource
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import crag_bridge  # noqa: F401
from chunks import load_chunks
from eval_harness import load_golden
from ladder import fragments, holds
from ingest import VARIANTS

import route_eval

RESULTS = Path(__file__).resolve().parent / "results" / "phase0.json"
KEV_URL = "http://127.0.0.1:8009/v1/systemone"
OPTIONS = {"A": "relevant: the passage states the answer, or a fact the answer needs",
           "B": "partial: the passage is about the same match, player or ground, but does not state the answer",
           "C": "irrelevant: the passage is about something else"}
TRUTH = {True: "A", False: "C"}


def pairs(n: int = 10) -> list[dict]:
    golden = [g for g in load_golden(route_eval.TUNE) if g.question_type != "no_answer" and g.evidence]
    cs, rng, out = load_chunks(), random.Random(0), []
    for g in golden[:: max(1, len(golden) // n)][:n]:
        parts = fragments(g)
        hit = next(c for c in cs if any(holds(p, VARIANTS["header_match"](c)) for p in parts))
        miss = rng.choice([c for c in cs if c.doc_id != hit.doc_id])
        for c, relevant in ((hit, True), (miss, False)):
            out.append({"id": g.id, "question": g.question, "passage": VARIANTS["header_match"](c),
                        "chunk_id": c.chunk_id, "relevant": relevant})
    return out


def question(q: str) -> dict:
    return {"relevance": {"type": "choice", "criteria": OPTIONS,
                          "instructions": f"Question: {q}\nHow useful is the passage for answering this question?"}}


def laya_run(ps: list[dict]) -> dict:
    import laya
    start = time.perf_counter()
    agent = laya.load("convaiinnovations/laya")
    load_s = time.perf_counter() - start
    agent.predict(ps[0]["passage"], question(ps[0]["question"]))  # warm-up
    rows = []
    for p in ps:
        t = time.perf_counter()
        a = agent.predict(p["passage"], question(p["question"]))["answers"]["relevance"]
        rows.append({**p, "pick": a["choice"], "confidence": a.get("confidence"), "ms": (time.perf_counter() - t) * 1000})
    return {"load_s": load_s, "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
            "device": str(getattr(agent, "device", "?")), "cfg": {k: agent.cfg.get(k) for k in ("max_len", "head_max_len")}
            if hasattr(agent, "cfg") else {}, "rows": rows}


def kev_run(ps: list[dict]) -> dict:
    def call(p):
        body = json.dumps({"state": p["passage"], "model": "kev-latest", "questions": question(p["question"])}).encode()
        req = urllib.request.Request(KEV_URL, body, {"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    call(ps[0])  # warm-up
    rows = []
    for p in ps:
        t = time.perf_counter()
        a = call(p)["answers"]["relevance"]
        rows.append({**p, "pick": a["choice"], "confidence": a.get("confidence"), "ms": (time.perf_counter() - t) * 1000})
    return {"rows": rows}


def main():
    name = sys.argv[1]
    ps = pairs()
    res = {"laya": laya_run, "kev": kev_run}[name](ps)
    ms = [r["ms"] for r in res["rows"]]
    hits = sum(r["pick"] == TRUTH[r["relevant"]] for r in res["rows"])
    res |= {"ms_p50": statistics.median(ms), "ms_max": max(ms), "smoke_hits": hits, "pairs": len(ps),
            "picks": {k: sum(r["pick"] == k for r in res["rows"]) for k in OPTIONS}}
    out = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    out.setdefault("models", {})[name] = res
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print({k: v for k, v in res.items() if k != "rows"})
    for r in res["rows"][:6]:
        print(f"  {r['id']:10} {'rel' if r['relevant'] else 'irr'} -> {r['pick']} ({r['confidence']}) {r['ms']:.0f} ms")


if __name__ == "__main__":
    main()
