"""Phase 4 label fix: none-06 said the documents do not name the player of the 2026 Men's T20 World Cup final; they
do (Bumrah: jasprit_bumrah::s008::02, narendra_modi_stadium::s014::05). It is now solo-07 in the test set. Nothing is
run again: the answers saved in results/phase4_test.json are re-marked against the fixed label, and the corrected
scores go to results/phase4_corrected.json (phase4_test.json stays as run).

Run from corrective_rag_in_depth/ (OpenAI key sourced; 8 grader calls):
    uv run corrective/test_relabel.py
"""

import json
import statistics

import crag_bridge  # noqa: F401
from eval_harness import load_golden
from eval_harness.metrics import is_refusal

import answer_tuning
import crag_eval
import crag_golden_check as gc
import route_eval

RENAMED = {"none-06": "solo-07"}


def main():
    run = json.loads((crag_eval.RESULTS.parent / "phase4_test.json").read_text())
    golden = {g.id: g for g in load_golden(gc.golden_name("test"))}
    out = {"renamed": RENAMED}
    for step, data in run.items():
        rows = {i: dict(r) for i, r in data["questions"].items()}
        for old, new in RENAMED.items():
            g, r = golden[new], rows.pop(old)
            r["correct"] = 0.0 if is_refusal(r["answer"]) else route_eval.grade(g.question, g.ground_truth, r["answer"])
            r["correct3"], r["kind"] = answer_tuning.grade3(g, r["answer"]), gc.kind(new)
            rows[new] = r
        out[step] = {"correct": statistics.mean(r["correct"] for r in rows.values()),
                     "correct3": statistics.mean(r["correct3"] for r in rows.values()),
                     "by_kind": crag_eval.summary(rows, None)["by_kind"],
                     "relabelled": {RENAMED[o]: rows[RENAMED[o]]["correct"] for o in RENAMED}}
        print(step, {k: v for k, v in out[step].items()})
    (crag_eval.RESULTS.parent / "phase4_corrected.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
