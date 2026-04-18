# blogAI_evals — Project Log

---

## State
**Current task:** Eval pipeline running (run_all.py — NLI model slow, still in progress at session end)
**Branch:** N/A (not yet committed)
**Last action:** Generated 27 samples × 3 conditions × 2 modes; rubric validation passed (ρ=0.926/0.876); gold_standard.jsonl written; eval running; make_figures.py + make_report.py scaffolded and ready

---

## Constraints
*Hard limits. Read before touching code.*

| Constraint | Source |
|---|---|
| Article is floor constraint only (hallucination check) — not a primary metric | Study design |
| Voice scoring must be automated (regex/spaCy) — no LLM-as-judge | Study design |
| Substance fidelity measured against notes, not article | Study design |
| Same input dataset as Study 1 where possible (enables cross-study comparison) | Study design |
| No manual edits to any output at any stage — Haiku edit prompt is the only intervention | Study design |
| Haiku edit prompt must be identical across all samples within a mode — changing it invalidates Phase 2 results | Study design |
| Token/cost tracking applies to Haiku only — free-tier models (Gemma, Qwen, Mistral) are $0 by definition | Study design |
| Phase 2 runs edit pass on Phase 1 winner only — do not run edit pass on all models | Study design |
| Haiku generation prompts frozen after Phase 0 rubric validation — changing them breaks Phase 2 comparison baseline | Study design |
| Model versions must be pinned to exact HF model IDs from app/config.py before generation begins | Reproducibility |
| Gemma and Mistral fail on HF free tier serverless API — both silently fall back to Haiku in production. BlogAI has effectively been running Qwen + Haiku only. Confirmed 2026-04-14 via live test. | Live test finding |
| HF free tier does not support `google/gemma-2-9b-it` or `mistralai/Mistral-7B-Instruct-v0.3` on the serverless chat endpoint — eval model set must be revised accordingly | Live test finding |
| 8 article URLs blocked by trafilatura (paywalls or bot detection): sample_006, 009, 010, 011, 020, 021, 024, 027. Failed silently — script printed FAILED and skipped the sample, all 3 conditions missing. Article text for samples 009, 010, 011, 020, 021 manually copy-pasted by user to `data/articles/{id}.txt`. Samples 006, 024, 027 remain dropped (paywalled). generate.py checks for local txt before fetching. Note in findings: article source for txt samples is user-copied not scraped; sample_020 contains ad copy mixed in with article text — outputs from this sample may have noise. | Live test finding |
| Haiku edit prompt does not include original notes — edit pass tested on draft only, cannot restore already-lost substance | Study design |
| Phase 1 winner selection: substance × 0.6 + voice × 0.4; substance is tiebreaker within 0.02 | Study design |
| Rubric validation gate must pass (Spearman ρ ≥ 0.5) before full study proceeds | Study design |

---

## Todo

### Skills & Protocol
- [ ] **Fix project-sync skill description** — skill currently checks `.git/config` (current directory assumption), but should check project being worked on instead. Affected blogAI_evals session where project-sync skipped (not a git repo).

### Platform Style Research (NEW — in progress)
- [x] **Design platform research component** — plan for LinkedIn vs blog style differentiation
- [x] **Create research/ directory** — scraper.py, analyze.py, linkedin_formatter.py
- [x] **Curate blog URLs** — 50 URLs across research institutions, tech blogs, neuroscience, philosophy
- [x] **Collect LinkedIn posts** — 50 posts manually via linkedin_formatter.py interactive mode
- [x] **Run analyze.py** — extract features, produce analysis_results.json (LinkedIn-only; blog baseline dropped)
- [ ] **Validate/tune score_linkedin()** — rubric misalignment documented; correction deferred
- [ ] **Update blogAI prompt** — apply findings to FORMAT_CONFIGS["LinkedIn"] in app/config.py

### Study 2 (Original — paused)
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

## Notion Writes (2026-04-16 session, end)
- null-baseline-calibration — updated — gold standard: editing model outputs, not writing from scratch; avoid circular comparison
- scraping-local-fallback — created — local txt fallback for paywalled/bot-blocked URLs in scraping pipelines
- eval-condition-discovery — created — derive conditions dynamically from output files, not hardcoded list

## Notion Writes (2026-04-16 session)
- prompt-rubric-alignment — created — prompts must not contradict rubric checks
- null-baseline-calibration — created — null baseline establishes eval floor before study runs
- pretooluse-interpreter-rewrite — created — PreToolUse hook rewrites python → python3 globally

## Notion Writes

- study-validation-vs-discovery — created — engineering vs research evaluation design distinction
- metric-operationalization-substance — created — cosine similarity ≠ argument preservation
- metric-operationalization-voice — created — surface markers ≠ voice; two-tier rubric
- experimental-design-within-subjects — created — same inputs required across mode conditions
- pipeline-intermediate-scoring — created — score each stage to isolate failure location
- automatable-rubric-design — created — rule-based evaluation beats LLM-as-judge when patterns are learnable
- manual-collection-automated-format — created — manual data collection with automated formatting for ToS-constrained APIs
- feature-driven-platform-validation — created — extract linguistic features before designing platform rubrics
- llm-api-model-availability — updated — added silent fallback mechanism, startup validation pattern, eval dry-run recommendation; domain expanded to huggingface+api
- resumable-api-generation — created — JSONL {id: record} load-skip-append pattern for interruptible generation pipelines
- edit-prompt-as-style-nudge — created — edit pass should be a short parameterized nudge, not a formatting spec
- constraint-first-study-design — created — identify hard constraints upfront and design around them

## Project Sync (2026-04-16 end)
- Detail page exists (no update)
- Database: Last worked on (2026-04-16), Last action, Status (Active) updated
- Status: ✓ Complete

## Project Sync (2026-04-16)
- Detail page exists (no update)
- Database: Last worked on (2026-04-16), Last action, Status (Active) updated
- Status: ✓ Complete

## Project Sync (2026-04-14)
- Detail page exists (no update)
- Database: Last worked on (2026-04-14), Last action, Status (Active) updated
- Status: ✓ Complete

---

## Done

- 2026-04-04 — **LinkedIn platform style research component designed and scaffolded**
  - Identified gap: current `score_linkedin()` routes to `score_blog()` (placeholder)
  - Designed data-driven approach: scrape blogs + manually collect LinkedIn posts → analyze structural differences → validate rubric
  - Constraint: LinkedIn ToS prohibits automated scraping; manual collection only (solved via interactive formatter)
  - Created `research/scraper.py` (trafilatura-based blog fetcher)
  - Created `research/analyze.py` (spaCy-based feature extraction: hook velocity, paragraph density, line breaks, conversational markers, formal transitions, CTAs)
  - Created `research/linkedin_formatter.py` (interactive JSONL builder for LinkedIn posts)
  - Created `notebooks/04_platform_style_analysis.ipynb` (visualization + summary stats)
  - Added 7-check `score_linkedin()` rubric to `eval/voice_rubric.py` (distinct from blog rubric)
  - Updated router in `voice_rubric.py` to call `score_linkedin()` instead of placeholder
  - Curated 50 diverse blog URLs (research institutions, tech analysis, neuroscience, philosophy, personal essays)
  - Collected 50 LinkedIn posts manually via interactive formatter
  - Updated requirements.txt with trafilatura + requests
  - Next: run scraper + analyzer to produce data-grounded findings

---

## Done Archive

## Session Note (2026-04-16)
HF serverless API went down mid-generation. Haiku standalone outputs completed for all available samples. Qwen and qwen_haiku outputs incomplete — script is resumable, will retry Qwen calls when HF is back up. Affected: samples 001/003/004 (Qwen timeout retries) + 009/010/011/020/021 (new txt samples, Qwen not yet run).
