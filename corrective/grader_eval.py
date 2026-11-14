"""Phase 2: the LLM grader alone, against Phase 1's chunk labels. f6a's top 5 for each of the 58 tune questions
(290 passages, the ones the answer step reads); each is scored against the original question, not the rewrites.

The answer key:
    passage    relevant / partial / irrelevant (labels/chunk_labels_tune.jsonl)
    question   CORRECT if a top-5 passage is relevant, AMBIGUOUS if only partial ones, else INCORRECT
Every (upper, lower) pair of bars is tried; the one whose actions match the key on most questions is kept.

The labels only see exact wordings. Passages the grader calls useful (>= lower) but the key calls irrelevant, and
relevant ones it scores < lower, are written to results/phase2_disagree.json to read by hand; my verdicts go in
results/phase2_handcheck.json ({"<id>#<rank>": {"hand": label, "why": ...}}), and the run is scored on the labels as
they are and after the hand check. Bars are picked after the hand check. Results: results/phase2.json.

Run from corrective_rag_in_depth/ (OpenAI key sourced; ~$0.03 once, then cached):
    uv run corrective/grader_eval.py
"""

import json
import statistics
from collections import Counter
from types import SimpleNamespace

import crag_bridge  # noqa: F401
from eval_harness import load_golden, settings
from eval_harness.metrics import cost_usd
from ladder import labelled, norm

import crag_golden_check as gc
import grader

RESULTS = crag_bridge.ROOT / "corrective" / "results"
LABELS = ("relevant", "partial", "irrelevant")
ACTIONS = ("CORRECT", "AMBIGUOUS", "INCORRECT")
TOP = 5


def key_action(labels: list[str]) -> str:
    return "CORRECT" if "relevant" in labels else "AMBIGUOUS" if "partial" in labels else "INCORRECT"


def band(s: int, upper: int, lower: int) -> str:  # what the grader's score says, in the labels' terms
    return "relevant" if s >= upper else "irrelevant" if s < lower else "partial"


def summary(rows: list[dict], label_of, upper: int, lower: int) -> dict:
    qs = {}
    for r in rows:
        qs.setdefault(r["id"], []).append(r)
    acts = {i: (key_action([label_of(r) for r in rs]), grader.action([r["score"] for r in rs], upper, lower), rs[0]["kind"])
            for i, rs in qs.items()}
    conf = Counter((label_of(r), band(r["score"], upper, lower)) for r in rows)
    useful = [r for r in rows if label_of(r) != "irrelevant"]
    useless = [r for r in rows if label_of(r) == "irrelevant"]
    return {
        "bars": [upper, lower],
        "action_match": sum(k == g for k, g, _ in acts.values()) / len(acts),
        "action_by_kind": {kd: f"{sum(k == g for k, g, x in acts.values() if x == kd)}/{sum(x == kd for *_, x in acts.values())}"
                           for kd in sorted({x for *_, x in acts.values()})},
        "action_confusion": {f"{a} -> {b}": n for (a, b), n in Counter((k, g) for k, g, _ in acts.values()).items()},
        "passage_match": sum(conf[(lab, lab)] for lab in LABELS) / len(rows),
        "passage_confusion": {lab: {b: conf[(lab, b)] for b in LABELS} for lab in LABELS},
        "catch_rate": sum(r["score"] >= lower for r in useful) / len(useful),  # useful passages it doesn't throw away
        "false_alarms": sum(r["score"] >= lower for r in useless) / len(useless),  # useless passages it keeps
        "actions": {i: {"key": k, "grader": g} for i, (k, g, _) in acts.items()},
    }


def main():
    golden = {g.id: g for g in load_golden(gc.golden_name("tune"))}
    decoys = json.loads((gc.LABELS / "decoys.json").read_text())
    rows = [r for r in map(json.loads, (gc.LABELS / "chunk_labels_tune.jsonl").read_text().splitlines()) if r["rank"] <= TOP]
    cost, ms = [], []
    for r in rows:
        text = labelled(SimpleNamespace(chunk_id=r["chunk_id"]))
        r["score"], reply = grader.score(golden[r["id"]].question, text)
        r["decoy"] = r["label"] == "irrelevant" and any(norm(d) in norm(text) for d in decoys.get(r["id"], []))
        r["text"] = text
        cost.append(cost_usd(reply.tokens_in, reply.tokens_out, settings.llm_model))
        ms.append(reply.ms)

    hand_file = RESULTS / "phase2_handcheck.json"
    hand = json.loads(hand_file.read_text()) if hand_file.exists() else {}
    as_is = lambda r: r["label"]
    fixed = lambda r: hand.get(f"{r['id']}#{r['rank']}", {}).get("hand", r["label"])
    sweep = sorted(((u, lo) for u in range(1, 11) for lo in range(1, u + 1)),
                   key=lambda b: (-summary(rows, fixed, *b)["action_match"], -summary(rows, fixed, *b)["passage_match"], b))
    best = sweep[0]
    report = {"bars_picked": best, "labels_as_is": summary(rows, as_is, *best),
              "after_hand_check": summary(rows, fixed, *best), "hand_checked": len(hand),
              "paper_style_7_3": summary(rows, fixed, 7, 3),
              "score_by_label": {lab: Counter(r["score"] for r in rows if fixed(r) == lab) for lab in LABELS},
              "decoys": [{"id": r["id"], "rank": r["rank"], "score": r["score"]} for r in rows if r["decoy"]],
              "cost_per_decision": statistics.mean(cost), "cost_per_question": statistics.mean(cost) * TOP,
              "p50_ms_per_decision": statistics.median(ms)}
    disagree = [{k: r[k] for k in ("id", "rank", "chunk_id", "label", "score", "text")} | {"question": golden[r["id"]].question,
                "evidence": golden[r["id"]].evidence} for r in rows
                if (r["label"] == "irrelevant" and r["score"] >= best[1]) or (r["label"] == "relevant" and r["score"] < best[1])]
    (RESULTS / "phase2_disagree.json").write_text(json.dumps(disagree, indent=2, ensure_ascii=False))  # holds article text: not committed
    (RESULTS / "phase2.json").write_text(json.dumps(
        report | {"passages": [{k: v for k, v in r.items() if k != "text"} for r in rows]}, indent=2, ensure_ascii=False))

    print(f"{len(rows)} passages, {len({r['id'] for r in rows})} questions; bars picked (upper, lower) = {best}; "
          f"{len(hand)} hand-checked; {len(disagree)} disagreements written")
    print("top 5 bar pairs:", [(b, round(summary(rows, fixed, *b)['action_match'], 3)) for b in sweep[:5]])
    for name in ("labels_as_is", "after_hand_check", "paper_style_7_3"):
        s = report[name]
        print(f"\n{name} {s['bars']}: action match {s['action_match']:.3f}, passage match {s['passage_match']:.3f}, "
              f"catch {s['catch_rate']:.3f}, false alarms {s['false_alarms']:.3f}")
        print("  by kind:", s["action_by_kind"])
        print("  actions (key -> grader):", s["action_confusion"])
        print("  passages (key -> grader band):", s["passage_confusion"])
    print("\nscores by label:", {k: dict(sorted(v.items())) for k, v in report["score_by_label"].items()})
    print("decoys:", report["decoys"])
    print(f"${report['cost_per_question']:.5f}/question (5 decisions), p50 {report['p50_ms_per_decision']:.0f} ms/decision")


if __name__ == "__main__":
    main()
