"""Reuse Adaptive RAG #4 as-is (and through it RAG Fusion #3 and Contextual RAG #2): router, routes, f6a search,
answer prompt v2, grader, chunks and Qdrant collections.

Adaptive's folder goes on sys.path; its own bridge adds Fusion's fusion/ and Contextual's cricket/. Every module there
is imported by bare name, so our modules must not reuse any of their names:
    Contextual #2: chunks, ingest, retrieve, hyde, llm_rerank, notes, ladder, answer
    Fusion #3:     contextual, rewrite, rag_fusion, diagnose, fusion_ladder, fusion_answer
    Adaptive #4:   bridge, bar, golden_check, llm_cache, routes, route_eval, loop_knobs, router, router_eval,
                   end_to_end, stacked, answer_tuning, test_run

Adaptive's LLM cache (answers, loop notes, router, grades) is copied once into this build's data/cache/llm.json and
llm_cache is pointed at the copy: replays are free and new calls land here, not in Adaptive's folder. The f6a
search's rewrite and rerank caches stay where they live (Fusion's and Contextual's folders) and are extended there,
as Adaptive did.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAGS = ROOT.parent
ADAPTIVE = RAGS / "adaptive_rag_in_depth" / "adaptive"
sys.path.append(str(ADAPTIVE))

import bridge  # noqa: E402, F401  (Adaptive's bridge: adds Fusion's fusion/ and Contextual's cricket/)
import llm_cache  # noqa: E402

CACHE = ROOT / "data" / "cache" / "llm.json"
if not CACHE.exists():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(llm_cache.CACHE, CACHE)
llm_cache.CACHE = CACHE
