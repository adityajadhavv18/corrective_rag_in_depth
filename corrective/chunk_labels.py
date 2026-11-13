"""Phase 1: free labels for the retrieval grader. For every tune question, f6a's top 20 on the main index (the list
the answer step's top 5 is cut from), each passage labelled by the harness's matching rule:

    relevant     the passage holds every evidence part
    partial      it holds some parts (one step of a chain, one half of a two-part answer)
    irrelevant   it holds none (every passage of a gap or no-answer question)

Most searches come from the caches (Adaptive's runs and the checker's); the gap questions' are new (~$0.01 once). Rows go to labels/chunk_labels_tune.jsonl.
`--sample` prints a stratified sample to check by hand: the labels only see exact wordings, so a passage that states
the answer in other words is labelled irrelevant.

Run from corrective_rag_in_depth/ (Qdrant running):
    uv run corrective/chunk_labels.py [--sample]
"""

import json
import random
import sys
from collections import Counter

import crag_bridge  # noqa: F401
import fusion_ladder  # noqa: F401  (registers f5a, the search of f6a)
from eval_harness import load_golden
from ladder import RUNGS, fragments, holds, labelled

import crag_golden_check as gc

OUT = gc.LABELS / "chunk_labels_tune.jsonl"
TOP = 20


def label(parts: list[list[str]], text: str) -> tuple[str, int]:
    n = sum(holds(p, text) for p in parts)
    return ("relevant" if parts and n == len(parts) else "partial" if n else "irrelevant"), n


def main():
    golden = load_golden(gc.golden_name("tune"))
    rows = []
    for g in golden:
        parts = fragments(g) if g.question_type != "no_answer" else []
        for rank, h in enumerate(RUNGS["f5a"][1](g.question, TOP), start=1):
            text = labelled(h)
            lab, n = label(parts, text)
            rows.append({"id": g.id, "kind": gc.kind(g.id), "question": g.question, "rank": rank,
                         "chunk_id": h.chunk_id, "label": lab, "parts_held": n, "parts": len(parts), "text": text})
    OUT.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "text"}, ensure_ascii=False) + "\n"
                           for r in rows))  # chunk ids only: article text is not committed (ladder.labelled rebuilds it)

    top5 = [r for r in rows if r["rank"] <= 5]
    print(f"{len(rows)} pairs from {len(golden)} questions -> {OUT.name}")
    for name, rs in (("top 20", rows), ("top 5", top5)):
        c = Counter(r["label"] for r in rs)
        print(f"  {name}: " + ", ".join(f"{k} {c[k]}" for k in ("relevant", "partial", "irrelevant")))
    by_kind = Counter((r["kind"], r["label"]) for r in top5)
    for k in sorted({r["kind"] for r in rows}):
        print(f"  top 5, {k:5}: " + ", ".join(f"{lab} {by_kind[(k, lab)]}" for lab in ("relevant", "partial", "irrelevant")))
    no_good = [g.id for g in golden if g.question_type != "no_answer"
               and not any(r["id"] == g.id and r["label"] == "relevant" for r in top5)]
    print(f"  answerable questions with no relevant passage in the top 5: {len(no_good)} ({', '.join(no_good)})")

    if "--sample" in sys.argv:
        rng = random.Random(0)
        pick = lambda lab, n, pool=top5: rng.sample([r for r in pool if r["label"] == lab and r["kind"] not in ("gap", "none")],
                                                   min(n, len([r for r in pool if r["label"] == lab and r["kind"] not in ("gap", "none")])))
        for r in pick("relevant", 8) + pick("partial", 8) + pick("irrelevant", 14):
            print(f"\n[{r['label']}] {r['id']} #{r['rank']}  Q: {r['question']}\n  {r['text'][:420]}")


if __name__ == "__main__":
    main()
