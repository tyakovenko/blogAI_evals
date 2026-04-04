# blogAI_evals — Project Log

---

## State
**Current task:** Study 2 scaffolded — eval scripts written, notebooks pending
**Branch:** N/A (not yet committed)
**Last action:** Wrote eval/ scripts (substance_fidelity.py, voice_rubric.py, factual_consistency.py, run_all.py) and README.md study plan

---

## Constraints
*Hard limits. Read before touching code.*

| Constraint | Source |
|---|---|
| Article is floor constraint only (hallucination check) — not a primary metric | Study design |
| Voice scoring must be automated (regex/spaCy) — no LLM-as-judge | Study design |
| Substance fidelity measured against notes, not article | Study design |
| Same input dataset as Study 1 where possible (enables cross-study comparison) | Study design |

---

## Todo

- [ ] **Scaffold notebooks/01_generate.ipynb** — Colab notebook: runs all 3 conditions × 2 modes, saves outputs to data/outputs/*.jsonl. Needs HF Inference API (Qwen) and Anthropic API (Haiku) calls. Logs latency per generation.
- [ ] **Scaffold notebooks/02_evaluate.ipynb** — Colab or local: runs eval/run_all.py pipeline, produces results/scores.csv. Needs spaCy model download (`en_core_web_sm`) and sentence-transformers.
- [ ] **Scaffold notebooks/03_analysis.ipynb** — 2D scatter plots (substance vs voice per condition), per-mode breakdowns, Wilcoxon signed-rank tests, effect sizes.
- [ ] **Collect inputs** — log 30–50 real BlogAI runs to data/inputs.jsonl. Min 3 domains, ~60% blog / ~40% linkedin.
- [ ] **Initial commit** — commit current scaffold (eval scripts, README, project-log, requirements)

---

## Decisions

| Decision | Rationale | Rejected | Date |
|---|---|---|---|
| Voice rubric is regex/spaCy only | Rules in taya-voice skill are precise enough to automate; LLM-as-judge adds cost and noise | LLM-as-judge scoring | 2026-04-03 |
| Article enters as hallucination floor only | BlogAI goal is to amplify user notes, not summarize article | Article coverage as primary metric | 2026-04-03 |
| Two modes (blog + linkedin) per sample | Mode adaptation is the core tension being studied | Single mode only | 2026-04-03 |

---

## Done

---

## Done Archive
