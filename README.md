# Corrective RAG #5: cricket on Wikipedia

Ask "on what date did Cheteshwar Pujara announce his retirement?" and Adaptive RAG #4 says "I don't know": the answer is not in its library. Corrective RAG checks what the search brought back before answering. Here every passage scores 0, so it searches a second library, finds "Cheteshwar Pujara announced his retirement from all forms of cricket on 24 August 2025", and answers with the date. A question the main library does answer goes through exactly as before.

**Result on the test set (20 questions nobody tuned on): Adaptive 0.75 → Adaptive + CRAG 0.90.** Questions whose answer is only in the second library went from 0/4 to 3/4, and no question that Adaptive answered was broken. The cost was $0.0006 more per question.

```
                         ┌──────────────────────┐
                         │   User's question    │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ ROUTER (Adaptive #4) │   few-shot gpt-4o-mini, one word
                         └──┬────────┬────────┬─┘
                            │        │        │
                            ▼        ▼        ▼
                     ┌────────┐ ┌──────────┐ ┌─────────────────────────┐
                     │ NONE   │ │ SINGLE   │ │ MULTI (IRCoT loop)      │
                     │ answer │ │ one f6a  │ │ f6a(question) → note    │
                     │ from   │ │ search,  │ │ → f6a(note) → …         │
                     │ memory │ │ top 5    │ │ (at most 3 searches)    │
                     └───┬────┘ └────┬─────┘ └────────────┬────────────┘
                         │           └──────────┬─────────┘
                         │                      ▼  the passages the answer step would read
                         │        ┌─────────────────────────────────────┐
                         │        │ CRAG GRADER (new)                   │
                         │        │ gpt-4o-mini scores each passage     │
                         │        │ against the original question:      │
                         │        │ 10 answers it / 5 part of it / 0 no │
                         │        └──┬──────────────┬───────────────┬───┘
                         │      any 10 │   no 10, some 5 │     all 0 │
                         │           ▼              ▼               ▼
                         │       CORRECT        AMBIGUOUS       INCORRECT
                         │       keep all       keep all +      throw them away,
                         │       passages       useful passages use the useful
                         │                      from the        passages from the
                         │                      FALLBACK        FALLBACK library
                         │                      library         (or "I don't know")
                         │           └──────────────┬───────────────┘
                         ▼                          ▼
                         ┌─────────────────────────────────────┐
                         │ ANSWER (prompt v2, unchanged)       │
                         │ [n] citations, or "I don't know"    │
                         └─────────────────────────────────────┘
```

Think of a family doctor. Most of the time they treat you themselves. When the tests show nothing useful, they refer you to a specialist. Only if the specialist finds nothing either do they say "we don't know yet". The value of CRAG is the referral: knowing when the first library has failed, and where to look next.

- **CRAG** (Yan et al., 2024, "Corrective Retrieval Augmented Generation"): a retrieval evaluator scores each retrieved document. The question is CORRECT (refine the documents into useful strips), AMBIGUOUS (refined strips plus web search) or INCORRECT (discard them, rewrite the query, web search). The paper's evaluator is a fine-tuned T5-large; mine is an LLM prompt, and my "web" is a second library.
- **Adaptive RAG #4** ([adaptive_rag_in_depth](https://github.com/adityajadhavv18/adaptive_rag_in_depth)): the whole system underneath, unchanged.

## Same system as Adaptive #4, plus a second library

Rebuilding a RAG system to test CRAG would teach nothing new, so I reused Adaptive #4 as it is: the 140 cricket Wikipedia articles and 4,650 chunks of [contextual-rag-in-depth](https://github.com/adityajadhavv18/contextual-rag-in-depth), Fusion #3's f6a search ([rag_fusion_in_depth](https://github.com/adityajadhavv18/rag_fusion_in_depth): 4 keyword rewrites + the question, dense / BM25 / SPLADE, RRF, LLM rerank), the few-shot router, the IRCoT loop and answer prompt v2. `corrective/crag_bridge.py` puts Adaptive's code on the import path, and CRAG replaces one function, the hand-over from search to answer (`routes.answer_from`). The none route is never graded, and the loop's pool is graded once, after the loop.

What is new:

- **The fallback library.** When the cricket corpus was built, 9 player biographies were left out (Pietersen, Cook, Head, Labuschagne, Pujara, Johnson, Williamson, Suryakumar Yadav, Pant). They are now indexed exactly like the main library, in their own Qdrant collection (306 chunks, $0.06 of section notes). The main library is untouched.
- **New questions**, in `eval_harness/golden_sets/golden_cricket_crag_{tune,test}_v1.jsonl`:

| Kind | Tune | Test | What it tests | Example |
|---|---|---|---|---|
| old: Adaptive #4's tune set | 46 | – | nothing breaks | "how was Root out in the second innings of the first Test of the 2023 Ashes?" |
| **solo**: one match fact | – | 7 | plain search still works | "what did Sunrisers Hyderabad score in the 2024 IPL final?" |
| **chain**: two steps | (8 in old) | 4 | the loop still works | "how many spectators can the ground that hosted the fourth Test of India's 2025 tour of England hold?" |
| **gap**: answer only in the fallback library | 6 | 4 | the fallback | "with which player was Travis Head in an on-field altercation in the 2026 IPL?" |
| **trap**: a look-alike passage holds a tempting wrong answer | 6 | 4 | the grader | "highest Test score by an Indian batter against England?" (Kohli's 235 was "the highest" once; Nair's 303* is) |
| **none**: no answer anywhere | (2 in old) | 1 | "I don't know" | "who was player of the match in the 2026 IPL final?" |
| **all** | **58** | **20** | | |

`corrective/crag_golden_check.py` checks every new question: the evidence is word for word in a chunk; a gap question's answer is in the fallback and nowhere in the main library; a trap's look-alike sits in a main-library passage; no single chunk holds both steps of a chain; and the model cannot answer a gap question from memory. Adaptive's test set had been used, so the 20 test questions are all new.

Every setting was chosen on tune. The test set was run once, at the end.

## Run it

From `corrective_rag_in_depth/`, next to working `adaptive_rag_in_depth/`, `rag_fusion_in_depth/`, `contextual_rag_in_depth/` and `eval_harness/`. Qdrant runs in Docker (`docker start qdrant`). The OpenAI key is needed for anything not yet cached.

```bash
uv run corrective/baseline.py                  # Phase 0: Adaptive's tune score, reproduced from the caches
uv run corrective/fallback_index.py            # Phase 0: the fallback library
uv run corrective/model_check.py laya          # Phase 0: can a local classifier run on the Mac?
uv run corrective/crag_golden_check.py --search --memory   # Phase 1: check the new questions
uv run corrective/chunk_labels.py              # Phase 1: labels for f6a's top 20 of every tune question
uv run corrective/grader_eval.py               # Phase 2: the LLM grader against the labels
uv run corrective/laya_eval.py                 # Phase 2: Laya on the same passages
uv run corrective/crag_eval.py 3.0 3a 3b 3c 3d 3e 3f   # Phase 3: CRAG's actions, one at a time
uv run corrective/test_once.py                 # Phase 4: test, once (refuses to run again)
uv run corrective/test_relabel.py              # Phase 4: re-mark the saved answers after the label fix
```

## Results

Scores are the share of questions answered right, marked by Contextual #2's answer checker (1 = right, 0.5 = half, 0 = wrong; "I don't know" on an answerable question is 0). A second checker that asks three times and takes the majority agreed within 0.01 everywhere.

### Phase 0: the reuse works, and the classifiers are slow on a Mac

Adaptive's final system, run through the bridge on its own 46 tune questions, scored **0.935**: the same as Adaptive's saved result, question by question, at $0 (every call came from the cache; a dummy API key proved it).

CRAG's evaluator is a small trained model, so I first checked whether a ready-made decision model could run on my 8 GB Mac. Each judged 20 easy pairs (a question with a passage that answers it, and one from another article):

| Model | How it runs | Memory | Time per passage | 20 easy pairs |
|---|---|---|---|---|
| Laya (ModernBERT-large, 421M) | pip package, on the Mac's GPU | ~1 GB | 0.30 s | 10/20, called 17 "relevant" |
| Kev (0.8B) | the repo's own server, MLX | 3.1 GB (peak 4.4) | 0.38 s | 13/20 |
| Jev | hosted only | – | – | not tried (no API key) |

Neither was usable without fine-tuning, which needs labelled data and GPU time I don't have, so the grader is gpt-4o-mini with a prompt. It sits behind one function (`grader.decide`) so a classifier can be swapped in later.

### Phase 1: labels for every passage

For each of the 58 tune questions I labelled f6a's top 20 passages by the harness's rule: **relevant** (holds the whole answer), **partial** (one part, or one step of a chain), **irrelevant** (none). That gives 1,160 passages, of which 290 are in the top 5 the answer step reads:

| Top 5 (290 passages) | Relevant | Partial | Irrelevant |
|---|---|---|---|
| count | 50 | 13 | 227 |

About 4 in 5 passages the answer step reads do not hold the answer, yet Adaptive still scores 0.935: one good passage is enough. The look-alike traps rarely got that far: f6a's LLM rerank kept the look-alike out of the top 5 in 8 of 10 traps. I checked 30 labels by hand and agreed with 28; both disagreements were passages that state the fact in other words, which an exact-wording rule cannot see.

### Phase 2: the grader alone

The grader read each of the 290 top-5 passages with the original question and scored it. It only ever gave three scores, **0, 5 or 10**, so the bars are: any 10 → CORRECT, all 0 → INCORRECT, otherwise AMBIGUOUS. I read every passage where the grader and the labels disagreed (42): 28 of the "useless but scored 10" passages did state the answer in other words ("Champions: India", "Gambhir and Dhoni added 109"), so the labels were fixed by hand before scoring.

| Grader on 290 passages (bars 10 / 5 / 0) | LLM grader | Laya, zero-shot |
|---|---|---|
| passages marked like the labels | 0.786 | 0.234 |
| useful passages it keeps (5 or 10) | 0.923 | 1.000 |
| useless passages it keeps | 0.201 | 1.000 |
| questions given the labels' action | 0.810 | 0.672 |
| **INCORRECT questions caught** | **8/10** | **0/10** |
| time per passage | 0.83 s | 0.32 s |
| cost per question (5 passages) | $0.0003 | $0 |

The LLM grader is good at the one thing that matters most here: noticing that nothing in the top 5 helps (all 6 gap questions went to INCORRECT or AMBIGUOUS, both of which search the fallback). It is weaker at judging single passages. The 11 useless passages it scored 10 were all look-alikes: the right match but not the asked detail, such as another innings or day (4), the wrong ground (2), an old record or a look-alike fact (5), e.g. Dhoni's 224 as "the highest Test score by an Indian captain", a record broken twice since. It reads the topic more than the detail.

Laya said "relevant" to 268 of 290 passages and "irrelevant" to none. Its confidence was no help either: 0.23 on useful passages and 0.26 on useless ones (it separates them barely better than a coin flip, 0.56 where 0.5 is a coin flip). In CRAG it would never send a question to the fallback.

Multi-step questions are where the grader and the labels disagreed most (9 of 11 action misses). Seen alone, a step-one passage ("the fourth Test was at Old Trafford") does not answer "how many spectators can the ground hold?", so it scores 0.

### Phase 3: CRAG's actions on Adaptive, one at a time (tune 58)

| Step | What CRAG does | Right answers | Gap | Fixed / broken vs Adaptive | $ per question | p50 time |
|---|---|---|---|---|---|---|
| 3.0 | nothing (Adaptive as it is) | 0.828 | 0/6 | – | 0.0025 | 9.2 s |
| 3a | drop the passages scored 0 | 0.767 | 0/6 | 0 / 4 | 0.0028 | 10.3 s |
| 3b | 3a + cut kept passages into strips, keep the useful ones | 0.750 | 0/6 | 1 / 6 | 0.0030 | 11.3 s |
| 3c | 3a + fallback library (the paper's recipe, minus strips) | 0.871 | 6/6 | 6 / 4 | 0.0030 | 11.8 s |
| 3d | 3a + the main library's ranks 6–40, keep 10s only | 0.767 | 0/6 | 0 / 4 | 0.0031 | 10.3 s |
| 3e | all of the above | 0.853 | 6/6 | 7 / 6 | 0.0036 | 13.2 s |
| **3f** | **keep every main passage; INCORRECT swaps them for the fallback's, AMBIGUOUS adds the fallback's** | **0.931** | **6/6** | **6 / 0** | **0.0031** | **11.6 s** |

(3.0 is 0.828 rather than 0.935 because the 12 new tune questions are in: the 6 gap questions are impossible for Adaptive.)

- **The fallback is the whole gain.** It answered all 6 gap questions, which Adaptive cannot answer at all.
- **Dropping passages hurts.** Two chain questions lost the step-one passage the grader scored 0, and two easy questions went wrong only because the answer step saw fewer passages ("200 runs" instead of 109, "Archer" instead of Hazlewood). The answer step is sensitive to small changes in what it reads.
- **Strips cut both ways.** They removed a look-alike line and fixed one trap, but broke six questions by cutting context the answer needed; one answer stitched fragments into "254 by Rohit Sharma".
- **Searching deeper in the main library never triggered usefully.** Some questions do have their answer at rank 21 or 27, but for those the top 5 already looked fine to the grader (and the loop's follow-up searches find them anyway). The questions CRAG flags are the ones whose answer is not in the main library at all. On this corpus the problem CRAG fixes is "not in the library", not "ranked too low".
- **3f keeps the fallback and drops the dropping.** The grader is trusted only for its strong call (nothing here helps), never to throw away single passages. That is lighter than the paper's CRAG, and on tune it fixed all 6 gap questions without breaking any. It was the system I took to test.

### Phase 4: test, once (20 questions)

| 20 test questions | Adaptive | Adaptive + CRAG 3f |
|---|---|---|
| **right answers** | **0.750** | **0.900** |
| as run, before the label fix below | 0.700 | 0.850 |
| solo / chain / trap / none | 7/7, 3/4, 4/4, 1/1 | 7/7, 3/4, 4/4, 1/1 |
| **gap** | **0/4** | **3/4** |
| fixed / broken | – | 3 / 0 |
| actions | – | CORRECT 16, AMBIGUOUS 1, INCORRECT 3 |
| $ per question / p50 time | $0.0026 / 10.3 s | $0.0032 / 12.2 s |

One test question was labelled "no answer in the documents" but the documents do name the player of the 2026 T20 World Cup final (Jasprit Bumrah), and both systems said so. I fixed the label and re-marked the two saved answers; nothing was run again, and the gap between the systems is the same either way.

The fallback answered three questions Adaptive could only refuse: Head's altercation with Kohli in the 2026 IPL, Labuschagne as Queensland captain, and Pietersen as Delhi Capitals mentor. The tune result held on unseen questions: **+0.15, nothing broken, for $0.0006 and about 2 s more per question.**

## When I'd reach for it

When the library **can't hold everything** and there is a **second place to look**: an archive, a wider index, the web. CRAG earns its keep by noticing that the first search failed and going somewhere else. Use the grader for **the call it makes well**, "nothing here helps", and don't let it throw away single passages: an LLM grader reads the topic more than the detail, and an answer step reads better with the passages it was tuned on. If everything you need is in one well-searched library, CRAG adds cost and little else.

## Files

| File | Purpose |
|---|---|
| `corrective/crag_bridge.py` | Puts Adaptive #4's code (and through it Fusion #3's and Contextual #2's) on the import path; shares Adaptive's LLM cache |
| `corrective/baseline.py` | Phase 0: Adaptive's tune score through the bridge, from the caches |
| `corrective/fallback_index.py` | Phase 0: the fallback library from the 9 left-out biographies |
| `corrective/model_check.py` | Phase 0: Laya and Kev on the Mac |
| `corrective/crag_golden_check.py` | Phase 1: builds and checks the CRAG golden sets |
| `corrective/chunk_labels.py` | Phase 1: relevant / partial / irrelevant labels for f6a's top 20 (chunk ids only) |
| `corrective/grader.py` | The CRAG grader: prompt, 10 / 5 / 0 bars, parallel grading |
| `corrective/grader_eval.py`, `laya_eval.py` | Phase 2: the LLM grader and Laya against the hand-checked labels |
| `corrective/crag.py` | CRAG between search and answer: grade, action, drop, strips, fallback, deep search, 3f |
| `corrective/crag_eval.py` | Phase 3: steps 3.0–3f on tune |
| `corrective/test_once.py`, `test_relabel.py` | Phase 4: the test run (once) and the label fix |
| `corrective/results/*.json`, `*.md` | Every phase's numbers, per question |
| `labels/` | New questions, kinds, look-alike strings, chunk labels |
