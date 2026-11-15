"""Phase 2b: Laya (zero-shot) as the retrieval grader, on exactly Phase 2's 290 pairs (f6a's top 5 for the 58 tune
questions), scored against the same hand-checked labels as the LLM grader.

Laya answers Phase 0's choice question (relevant / partial / irrelevant, model_check.py); its pick becomes a
10 / 5 / 0 score so the same bars (10 = CORRECT, all 0 = INCORRECT) and grader_eval.summary apply. In 3f the grader
has one job: say INCORRECT when nothing in the main library helps (-> the fallback), and not otherwise (a CORRECT
question sent to the fallback loses its passages). Both are counted.

Results: results/phase2_laya.json and a short write-up, results/phase2_laya.md.

Run from corrective_rag_in_depth/ (no OpenAI calls; Laya on MPS, ~1 GB):
    uv run corrective/laya_eval.py
"""

import json
import statistics
import time
from types import SimpleNamespace

import crag_bridge
from eval_harness import load_golden
from ladder import labelled

import crag_golden_check as gc
import grader
import grader_eval
import model_check

RESULTS = crag_bridge.ROOT / "corrective" / "results"
SCORE = {"A": 10, "B": 5, "C": 0}


def incorrect_calls(s: dict) -> dict:
    acts = s["actions"].values()
    return {"incorrect_caught": f"{sum(a['key'] == 'INCORRECT' == a['grader'] for a in acts)}/"
                                f"{sum(a['key'] == 'INCORRECT' for a in acts)}",
            "false_incorrect": sum(a["key"] != "INCORRECT" and a["grader"] == "INCORRECT" for a in acts)}


def main():
    import laya
    golden = {g.id: g for g in load_golden(gc.golden_name("tune"))}
    phase2 = json.loads((RESULTS / "phase2.json").read_text())
    hand = json.loads((RESULTS / "phase2_handcheck.json").read_text())
    fixed = lambda r: hand.get(f"{r['id']}#{r['rank']}", {}).get("hand", r["label"])

    agent = laya.load("convaiinnovations/laya")
    llm, rows, ms = phase2["passages"], [], []
    for p in llm:
        q = golden[p["id"]].question
        t = time.perf_counter()
        a = agent.predict(labelled(SimpleNamespace(chunk_id=p["chunk_id"])), model_check.question(q))["answers"]["relevance"]
        ms.append((time.perf_counter() - t) * 1000)
        rows.append({k: p[k] for k in ("id", "kind", "rank", "chunk_id", "label")}
                    | {"score": SCORE.get(a["choice"], 0), "pick": a["choice"], "confidence": a.get("confidence")})

    bars = (grader.UPPER, grader.LOWER)
    report = {}
    for name, rs in (("llm", llm), ("laya", rows)):
        s = grader_eval.summary(rs, fixed, *bars)
        report[name] = {k: s[k] for k in ("action_match", "action_by_kind", "passage_match", "catch_rate", "false_alarms",
                                          "passage_confusion")} | incorrect_calls(s) | {"actions": s["actions"]}
    report["laya"] |= {"picks": {k: sum(r["pick"] == k for r in rows) for k in SCORE}, "p50_ms": statistics.median(ms)}
    report["llm"]["p50_ms"] = phase2["p50_ms_per_decision"]
    (RESULTS / "phase2_laya.json").write_text(json.dumps(report | {"passages": rows}, indent=2, ensure_ascii=False))

    lines = ["# Phase 2b: Laya vs the LLM grader (290 pairs, hand-checked labels, bars 10 / 5 / 0)", "",
             "| | passage match | useful kept | useless kept | action match | INCORRECT caught | CORRECT/AMBIGUOUS wrongly sent to fallback | time per decision |",
             "|---|---|---|---|---|---|---|---|"]
    for name, label in (("llm", "LLM grader (gpt-4o-mini)"), ("laya", "Laya, zero-shot (ModernBERT-large, 421M, MPS)")):
        r = report[name]
        lines.append(f"| {label} | {r['passage_match']:.3f} | {r['catch_rate']:.3f} | {r['false_alarms']:.3f} | "
                     f"{r['action_match']:.3f} | {r['incorrect_caught']} | {r['false_incorrect']} | {r['p50_ms']:.0f} ms |")
    lines += ["", f"Laya's picks: {report['laya']['picks']} (A relevant, B partial, C irrelevant).",
              "LLM grader cost: $0.00029 per question (5 decisions); Laya: $0, runs on the Mac."]
    (RESULTS / "phase2_laya.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for name in ("llm", "laya"):
        print(name, "by kind:", report[name]["action_by_kind"])


if __name__ == "__main__":
    main()
