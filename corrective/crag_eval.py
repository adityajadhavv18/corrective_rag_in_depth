"""Phase 3: CRAG's actions added to Adaptive one at a time, on the CRAG tune set (58 questions). Each step runs the
whole system (few-shot router -> none / single / multi -> CRAG -> answer prompt v2) through the harness and is graded
like Phase 0 (Contextual's 1-vote grader and Adaptive's 3-vote grader).

    3.0  Adaptive as it is (the 0.935 was on 46 of these 58)
    3a   drop: keep passages graded 5 or 10 (INCORRECT keeps nothing -> "I don't know")
    3b   3a + strips
    3c   3a + fallback library
    3d   3a + deeper main search (ranks 6-40, keep 10s only)
    3e   all of them
    3f   fallback without drop: CORRECT / AMBIGUOUS keep every main passage (AMBIGUOUS adds useful fallback ones),
         INCORRECT swaps them for the fallback's

Per step: right answers (1 and 3 votes) overall and by kind, actions, questions fixed / broken against 3.0, cost per
question and p50 time. Times come from the cached replay (run a step again with a dummy key: $0, and it proves
nothing new was called), where each grading batch counts as its slowest call. Results: results/phase3.json.

Run from corrective_rag_in_depth/ (Qdrant running; OpenAI key sourced for the first run of a step):
    uv run corrective/crag_eval.py 3.0 3a 3b 3c 3d 3e
"""

import json
import statistics
import sys
from collections import Counter

import crag_bridge
from eval_harness import load_golden

import answer_tuning
import baseline
import crag
import crag_golden_check as gc
import route_eval

RESULTS = crag_bridge.ROOT / "corrective" / "results" / "phase3.json"
ALL = {"drop": True, "strips": True, "fallback": True, "deep": True}
STEPS = {
    "3.0": ("Adaptive as it is", None),
    "3a": ("drop", {"drop": True}),
    "3b": ("drop + strips", {"drop": True, "strips": True}),
    "3c": ("drop + fallback", {"drop": True, "fallback": True}),
    "3d": ("drop + deep main search", {"drop": True, "deep": True}),
    "3e": ("drop + strips + fallback + deep", ALL),
    "3f": ("fallback, main passages kept unless INCORRECT", {"discard": True, "fallback": True}),
}
KINDS = ("old", "solo", "chain", "gap", "trap", "none")


def run(step: str, golden: dict, split: str = "tune") -> dict:
    crag.install(STEPS[step][1])
    picks = {}
    rows = route_eval.run_route(f"crag_{step}", golden, baseline.routed(picks), golden_set=gc.golden_name(split))
    for i, row in rows.items():
        q = golden[i].question
        row["route"], row["kind"] = picks[q], gc.kind(i)
        row["correct3"] = answer_tuning.grade3(golden[i], row["answer"])
        row["crag"] = crag.LOG.get(q)
    return rows


def summary(rows: dict, base: dict | None) -> dict:
    mean = lambda xs: statistics.mean(xs) if xs else None
    out = {"correct": mean([r["correct"] for r in rows.values()]),
           "correct3": mean([r["correct3"] for r in rows.values()]),
           "by_kind": {k: f"{sum(r['correct'] for r in rows.values() if r['kind'] == k):g}/"
                          f"{sum(r['kind'] == k for r in rows.values())}" for k in KINDS if any(r["kind"] == k for r in rows.values())},
           "actions": dict(Counter(r["crag"]["action"] for r in rows.values() if r["crag"])),
           "routes": dict(Counter(r["route"] for r in rows.values())),
           "refused": sum(r["refused"] for r in rows.values()),
           "cost_per_question": mean([r["cost"] for r in rows.values()]),
           "p50_ms": statistics.median([r["ms"] for r in rows.values()])}
    if base:
        out["fixed"] = [i for i in rows if rows[i]["correct"] > base[i]["correct"]]
        out["broken"] = [i for i in rows if rows[i]["correct"] < base[i]["correct"]]
    return out


def main():
    golden = {g.id: g for g in load_golden(gc.golden_name("tune"))}
    saved = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for step in sys.argv[1:]:
        rows = run(step, golden)
        base = saved.get("3.0", {}).get("questions") if step != "3.0" else None
        saved[step] = {"what": STEPS[step][0], "summary": summary(rows, base), "questions": rows}
        RESULTS.write_text(json.dumps(saved, indent=2, ensure_ascii=False))

    print("\n| step | what | right (1 vote) | right (3 votes) | " + " | ".join(KINDS) + " | actions | $/question | p50 s |")
    print("|---" * (8 + len(KINDS)) + "|")
    for step in (s for s in STEPS if s in saved):
        s = saved[step]["summary"]
        print(f"| {step} | {saved[step]['what']} | {s['correct']:.3f} | {s['correct3']:.3f} | "
              + " | ".join(s["by_kind"].get(k, "-") for k in KINDS)
              + f" | {', '.join(f'{a} {n}' for a, n in sorted(s['actions'].items())) or '-'} | "
              f"{s['cost_per_question']:.5f} | {s['p50_ms'] / 1000:.1f} |")
    for step in (s for s in STEPS if s in saved and s != "3.0"):
        s = saved[step]["summary"]
        print(f"{step}: fixed {', '.join(s.get('fixed', [])) or '-'}; broken {', '.join(s.get('broken', [])) or '-'}")


if __name__ == "__main__":
    main()
