# Phase 2b: Laya vs the LLM grader (290 pairs, hand-checked labels, bars 10 / 5 / 0)

| | passage match | useful kept | useless kept | action match | INCORRECT caught | CORRECT/AMBIGUOUS wrongly sent to fallback | time per decision |
|---|---|---|---|---|---|---|---|
| LLM grader (gpt-4o-mini) | 0.786 | 0.923 | 0.201 | 0.810 | 8/10 | 3 | 827 ms |
| Laya, zero-shot (ModernBERT-large, 421M, MPS) | 0.234 | 1.000 | 1.000 | 0.672 | 0/10 | 0 | 315 ms |

Laya's picks: {'A': 268, 'B': 22, 'C': 0} (A relevant, B partial, C irrelevant).
LLM grader cost: $0.00029 per question (5 decisions); Laya: $0, runs on the Mac.

Could a threshold on Laya's confidence rescue it? No: its "relevant" confidence averages 0.226 on useful passages and
0.258 on useless ones; AUC 0.559 (0.5 = coin flip). Zero-shot Laya cannot tell the two apart, so it never says
INCORRECT and in 3f would never send a question to the fallback: the system would stay at Adaptive's 0.828 on tune.
Laya's model card says its base checkpoints need fine-tuning for a task; that needs labelled data and GPU time we
don't have (Phase 0), so the LLM grader stays. Kev (13/20 on Phase 0's smoke pairs) was not re-run.
