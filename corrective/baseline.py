"""Phase 0: the bar CRAG plugs into. Adaptive #4's final system on its tune set, run live through the bridge:

    few-shot router -> none (memory) / single (one f6a search) / multi (IRCoT loop on f6a) -> answer prompt v2

and graded with Contextual #2's grader (1 vote) and Adaptive's 3-vote grader. Every LLM call should come from the
caches; run it with a dummy OpenAI key and any uncached call fails instead of spending. The result must match
Adaptive's saved Phase 7 numbers (v2, router: 0.935) question by question, or the reuse is broken.

Results go to corrective/results/phase0.json ("baseline").

Run from corrective_rag_in_depth/ (Qdrant running):
    OPENAI_API_KEY=sk-dummy uv run corrective/baseline.py
"""

import json
import statistics
from pathlib import Path

import crag_bridge
from eval_harness import PipelineOutput, load_golden

import answer_tuning
import llm_cache
import route_eval
import router
import routes
import stacked

RESULTS = Path(__file__).resolve().parent / "results" / "phase0.json"
SAVED = crag_bridge.ADAPTIVE / "results" / "answer_tuning_tune.json"
ROUTE_FNS = {"none": routes.no_search, **stacked.F6A_ROUTES}


def routed(picks: dict):
    """Adaptive's final system: the few-shot router, then its route. The router's tokens and time are added in."""
    def pipeline(question: str) -> PipelineOutput:
        pick, reply, _ = router.route(question, "few")
        out = ROUTE_FNS[pick](question)  # resets CACHED_MS, so the router's cached time is added after
        llm_cache.CACHED_MS["ms"] += reply.ms
        picks[question] = pick
        return PipelineOutput(answer=out.answer, contexts=out.contexts, context_ids=out.context_ids,
                              tokens_in=out.tokens_in + reply.tokens_in, tokens_out=out.tokens_out + reply.tokens_out)
    return pipeline


def main():
    golden = {g.id: g for g in load_golden(route_eval.TUNE)}
    picks = {}
    rows = route_eval.run_route("crag_baseline", golden, routed(picks))
    for i, row in rows.items():
        row["route"] = picks[golden[i].question]
        row["correct3"] = answer_tuning.grade3(golden[i], row["answer"])

    saved = json.loads(SAVED.read_text())["systems"]
    then = {"correct": saved["v2 1 vote"]["router"], "correct3": saved["v2 3 votes"]["router"]}
    mean = lambda xs: statistics.mean(xs)
    report = {}
    for field, old in then.items():
        diff = [i for i in golden if rows[i][field] != old[i]["correct"] or rows[i]["route"] != old[i]["route"]]
        report[field] = {"now": mean([r[field] for r in rows.values()]),
                         "saved": mean([r["correct"] for r in old.values()]), "differs": diff}
    report["cost_per_question"] = mean([r["cost"] for r in rows.values()])
    report["p50_ms"] = statistics.median([r["ms"] for r in rows.values()])
    report["routes"] = {r: sum(x["route"] == r for x in rows.values()) for r in ROUTE_FNS}

    out = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    out["baseline"] = {"summary": report, "questions": rows}
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n| grader | now | Adaptive saved | questions that differ |\n|---|---|---|---|")
    for field, name in (("correct", "1 vote"), ("correct3", "3 votes")):
        r = report[field]
        print(f"| {name} | {r['now']:.3f} | {r['saved']:.3f} | {', '.join(r['differs']) or 'none'} |")
    print(f"\nroutes {report['routes']}, ${report['cost_per_question']:.5f}/question (original calls), "
          f"p50 {report['p50_ms'] / 1000:.1f} s")


if __name__ == "__main__":
    main()
