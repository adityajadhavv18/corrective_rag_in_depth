"""Phase 4: the CRAG test set (20 questions nobody tuned on: 6 solo, 4 chain, 4 gap, 4 trap, 2 none), run ONCE.

    run 1   Adaptive as it is                     (3.0: router -> none / single / multi -> answer v2)
    run 2   Adaptive + CRAG 3f, the system picked on tune (fallback; main passages kept unless INCORRECT)

Same router, searches, loop and answer prompt; the only difference is the CRAG step, so the gap between the two is
what CRAG adds on unseen questions. Graded like every phase (1-vote and 3-vote answer checkers).

The results are written once to results/phase4_test.json and the script refuses to run again: whatever the numbers
are, they are the ones reported.

Run from corrective_rag_in_depth/ (Qdrant running, OpenAI key sourced):
    uv run corrective/test_once.py
"""

import json
import sys

import crag_bridge  # noqa: F401
from eval_harness import load_golden

import crag_eval
import crag_golden_check as gc

RESULTS = crag_eval.RESULTS.parent / "phase4_test.json"
RUNS = ("3.0", "3f")


def main():
    if RESULTS.exists():
        sys.exit(f"{RESULTS.name} exists: the test set has been run once already. Not running it again.")
    golden = {g.id: g for g in load_golden(gc.golden_name("test"))}
    out = {}
    for step in RUNS:
        rows = crag_eval.run(step, golden, split="test")
        out[step] = {"what": crag_eval.STEPS[step][0],
                     "summary": crag_eval.summary(rows, out["3.0"]["questions"] if step != "3.0" else None),
                     "questions": rows}
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    kinds = [k for k in crag_eval.KINDS if k in out["3.0"]["summary"]["by_kind"]]
    print("\n| run | system | right (1 vote) | right (3 votes) | " + " | ".join(kinds) + " | actions | $/question | p50 s |")
    print("|---" * (7 + len(kinds)) + "|")
    for step in RUNS:
        s = out[step]["summary"]
        print(f"| {step} | {out[step]['what']} | {s['correct']:.3f} | {s['correct3']:.3f} | "
              + " | ".join(s["by_kind"][k] for k in kinds)
              + f" | {', '.join(f'{a} {n}' for a, n in sorted(s['actions'].items())) or '-'} | "
              f"{s['cost_per_question']:.5f} | {s['p50_ms'] / 1000:.1f} |")
    s = out["3f"]["summary"]
    print(f"3f vs 3.0: fixed {', '.join(s['fixed']) or '-'}; broken {', '.join(s['broken']) or '-'}")


if __name__ == "__main__":
    main()
