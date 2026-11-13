"""Phase 1: build and check CRAG's golden sets before anything is run on them.

golden_cricket_crag_tune_v1.jsonl = Adaptive's 46 tune questions (unchanged) + labels/new_tune.jsonl
golden_cricket_crag_test_v1.jsonl = labels/new_test.jsonl (fresh: Adaptive's test set is spent)

Kinds (from the id; written to labels/kind_labels_<split>.jsonl, outside the harness contract):
    old     Adaptive's tune questions                          main index has the answer
    solo    one match fact, one search                         main index has the answer
    chain   two steps: match -> ground or player -> fact       main index has the answer, in two chunks
    gap     the answer is only in one of the 9 left-out biographies (the fallback library)
    trap    main index has the answer, and a look-alike passage holds a tempting wrong one (labels/decoys.json)
    none    no answer anywhere

Checks, for every new question:
  1. the harness loads the file (every source_doc in the main corpus or the fallback library)
  2. every evidence part is in at least one chunk, word for word (the harness's rule), and the last part is in the
     source_doc
  3. gap: no main-index chunk holds any evidence part; the fallback library holds them all
  4. trap: each decoy is in a main-index chunk that holds no evidence part
  5. chain: no single chunk holds every step
With --search (f6a, main index, top 20; the rewrite and rerank calls are cached, ~$0.04 once):
  solo / trap: the evidence is in the top 5 (else the question tests search, not grading)
  trap: a decoy reaches the top 5 (it would reach the answer step) or only the top 20
  chain: the top 5 does not hold every step
With --memory (the none route, ~$0.002 once): gap questions the model answers right from memory are rejected;
  for the others it is only reported.

Run from corrective_rag_in_depth/ (Qdrant running; OpenAI key sourced for --search / --memory):
    uv run corrective/crag_golden_check.py [--search] [--memory]
"""

import json
import sys
from pathlib import Path

import crag_bridge
from chunks import included_docs, load_chunks
from eval_harness import load_golden
from eval_harness.metrics import is_refusal
from ingest import VARIANTS
from ladder import fragments, holds, norm

import fallback_index

ROOT = crag_bridge.ROOT
LABELS = ROOT / "labels"
GOLDEN = crag_bridge.RAGS / "eval_harness" / "golden_sets"
ADAPTIVE_TUNE = GOLDEN / "golden_cricket_adaptive_tune_v1.jsonl"
RESULTS = ROOT / "corrective" / "results" / "phase1_check.json"
KINDS = ("solo", "chain", "gap", "trap", "none")


def golden_name(split: str) -> str:
    return f"golden_cricket_crag_{split}_v1.jsonl"


def kind(gid: str) -> str:
    return next((k for k in KINDS if gid.startswith(k + "-")), "old")


def assemble() -> None:
    """Write the two golden files and the kind labels from Adaptive's tune set and labels/new_*.jsonl."""
    for split in ("tune", "test"):
        new = (LABELS / f"new_{split}.jsonl").read_text().strip().splitlines()
        lines = (ADAPTIVE_TUNE.read_text().strip().splitlines() if split == "tune" else []) + new
        (GOLDEN / golden_name(split)).write_text("\n".join(lines) + "\n")
        ids = [json.loads(line)["id"] for line in lines]
        (LABELS / f"kind_labels_{split}.jsonl").write_text(
            "".join(json.dumps({"id": i, "kind": kind(i)}) + "\n" for i in ids))


def check(split: str, main, fallback, decoys, search=None, memory=None) -> tuple[list[str], dict]:
    docs = set(included_docs()) | set(fallback_index.left_out())
    golden = load_golden(golden_name(split), known_docs=docs)
    text = lambda c: VARIANTS["header_match"](c)  # what the answer step reads
    problems, report = [], {}
    other = {g.id for g in load_golden(golden_name("test" if split == "tune" else "tune"))}
    problems += [f"{i}: id also in the other split" for i in {g.id for g in golden} & other]
    for g in (g for g in golden if kind(g.id) != "old"):
        k, row = kind(g.id), {"kind": kind(g.id)}
        if k == "none":
            if g.question_type != "no_answer":
                problems.append(f"{g.id}: none-* must be no_answer")
        else:
            parts = fragments(g)
            in_main = [[c for c in main if holds(p, text(c))] for p in parts]
            in_fall = [[c for c in fallback if holds(p, text(c))] for p in parts]
            row["chunks"] = [f"{len(m)} main + {len(f)} fallback" for m, f in zip(in_main, in_fall)]
            if not all(m or f for m, f in zip(in_main, in_fall)):
                problems.append(f"{g.id}: an evidence part is in no chunk")
            elif not any(c.doc_id == g.source_doc for c in in_main[-1] + in_fall[-1]):
                problems.append(f"{g.id}: the answer part is not in {g.source_doc}")
            if k == "gap" and (any(in_main) or not all(in_fall)):
                problems.append(f"{g.id}: gap, but the main index holds a part or the fallback misses one")
            if k == "trap":
                for d in decoys.get(g.id, []):
                    where = [c for c in main if norm(d) in norm(text(c)) and not any(holds(p, text(c)) for p in parts)]
                    row.setdefault("decoy_chunks", []).append(len(where))
                    if not where:
                        problems.append(f"{g.id}: decoy {d[:40]!r} is in no chunk apart from the evidence")
            if k == "chain" and any(all(holds(p, text(c)) for p in parts) for c in main):
                problems.append(f"{g.id}: one chunk holds every step")
            if search and k in ("solo", "trap", "chain"):
                texts = [text_of(h) for h in search(g.question, 20)]
                found5 = [any(holds(p, t) for t in texts[:5]) for p in parts]
                row["top5_parts"] = f"{sum(found5)}/{len(parts)}"
                if k in ("solo", "trap") and not all(found5):
                    problems.append(f"{g.id}: f6a's top 5 misses the evidence")
                if k == "chain" and all(found5):
                    problems.append(f"{g.id}: f6a's top 5 holds every step")
                if k == "trap":
                    ranks = [i for i, t in enumerate(texts, start=1)
                             if any(norm(d) in norm(t) for d in decoys.get(g.id, []))
                             and not any(holds(p, t) for p in parts)]
                    row["decoy_ranks"] = ranks
        if memory and k != "none":
            right, answer = memory(g)
            row["memory"] = {"correct": right, "answer": answer}
            if k == "gap" and right == 1.0:
                problems.append(f"{g.id}: the model answers it from memory")
        report[g.id] = row
        print(f"  {g.id:9} {k:5} " + "  ".join(f"{a}={b}" for a, b in row.items() if a not in ("kind", "memory"))
              + (f"  memory={row['memory']['correct']:g}" if "memory" in row else ""))
    return problems, report


def text_of(hit) -> str:
    from ladder import labelled
    return labelled(hit)


def main():
    assemble()
    search = memory = None
    if "--search" in sys.argv:
        import fusion_ladder  # noqa: F401  (registers f5a, the search of f6a)
        from ladder import RUNGS
        search = RUNGS["f5a"][1]
    if "--memory" in sys.argv:
        import route_eval
        import routes

        def memory(g):  # the harness's rule: a refusal on an answerable question is wrong, before any grading
            answer = routes.no_search(g.question).answer
            return 0.0 if is_refusal(answer) else route_eval.grade(g.question, g.ground_truth, answer), answer
    main_chunks = load_chunks()
    fallback = fallback_index.fallback_chunks(fallback_index.left_out())
    decoys = json.loads((LABELS / "decoys.json").read_text())
    out, bad = {}, 0
    for split in ("tune", "test"):
        print(f"\n{split}:")
        problems, out[split] = check(split, main_chunks, fallback, decoys, search, memory)
        for p in problems:
            print(f"  PROBLEM {p}")
        out[split + "_problems"] = problems
        bad += len(problems)
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n{'all checks passed' if not bad else f'{bad} problem(s)'}")


if __name__ == "__main__":
    main()
