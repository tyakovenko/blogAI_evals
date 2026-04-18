# blogAI_evals — Study 2: Substance & Voice Fidelity Under Mode Transformation

## Research Question

When an LLM transforms user notes into mode-appropriate content (blog post, LinkedIn post), does it preserve the user's intellectual substance and voice — and which model configuration best maintains that fidelity under mode transformation?

## What This Study Is Not

This is not a summarization study. The article is not the primary source of content — the user's notes are. The article enters the evaluation through one question only: did the model fabricate something that contradicts the source? That is a floor constraint (hallucination check), not a scored dimension.

## What This Study Is

An evaluation of LLMs as **thought amplifiers** — tools that reformat existing ideas into new modes without distorting or replacing them. The core tension: mode adaptation (making output blog- or LinkedIn-appropriate) vs. substance preservation (keeping the user's actual argument intact).

This design is based on a critical review of the original plan (`v2.md`). Every design decision here incorporates the fixes documented there.

---

## Model Versions

Exact model IDs must be pinned before generation notebooks are written. Changing model versions after any run invalidates comparisons for that condition.

| Label | HF / API Model ID | Notes |
|---|---|---|
| `gemma` | `google/gemma-2-9b-it` | HF free tier — **NOT available on serverless API** |
| `qwen` | `Qwen/Qwen2.5-7B-Instruct` | HF free tier — confirmed working |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` | HF free tier — **NOT a chat model on serverless API** |
| `haiku` | `claude-haiku-4-5-20251001` | Anthropic — confirmed working |

**Finding (confirmed 2026-04-14):** Gemma and Mistral both fail on the HF free tier serverless API and fall back silently to Haiku in BlogAI production. The app has effectively been running Qwen + Haiku only. The eval model set must be revised to reflect reality — testing Gemma and Mistral as distinct conditions is not possible without HF Pro or a different endpoint.

**Revised effective model set:** `qwen` and `haiku` as standalone conditions. Phase 2 edit pass compares `qwen_haiku` vs. `haiku`. Gemma and Mistral are excluded until HF access is resolved.

---

## Study Structure

Two phases, 30 inputs each.

### Phase 1 — Standalone Model Comparison

**Goal:** Identify which model best preserves substance and voice without any editing intervention.

| Condition | Description |
|---|---|
| `gemma` | Gemma, mode-appropriate prompt, no edit |
| `qwen` | Qwen 2.5 7B, same prompt, no edit |
| `mistral` | Mistral, same prompt, no edit |
| `haiku` | Claude Haiku, same prompt, no edit |

4 models × 2 modes × 30 inputs = **240 outputs**

**Phase 1 winner selection:** The winner is the model with the highest combined score, computed as:

```
combined = (substance_aggregate × 0.6) + (voice_combined × 0.4)
```

Substance weighted higher because it is the primary research question — voice adaptation without substance preservation is not a useful product outcome. If two models tie within 0.02 combined score, substance score is the tiebreaker.

The winner is recorded in `results/phase1_winner.txt` (single line: model label, e.g. `gemma`). The Phase 2 generation notebook reads this file to configure itself — no manual wiring required.

### Phase 2 — Edit Pass Evaluation

**Goal:** Determine whether a Haiku edit pass improves the Phase 1 winner — and whether the combined pipeline is competitive with Haiku generating from scratch.

| Condition | Description |
|---|---|
| `best_model_haiku` | Phase 1 winner → Haiku edit pass |
| `haiku` | Haiku standalone (Phase 1 data — no new generation needed) |

1 pipeline × 2 modes × 30 inputs = **60 new outputs** (plus 30 intermediate pre-edit outputs per mode)

**The key comparison:** `best_model_haiku` vs. `haiku` on substance, voice, and cost. This answers:
- Does editing improve on the best standalone model?
- Is the pipeline competitive with Haiku generating from scratch?
- At what cost difference?

**Intermediate output capture:** The Phase 1 winner's raw output is saved before Haiku edits it and scored independently, enabling delta analysis — what did Haiku add, and what did it damage?

**Edit prompt scope:** Haiku receives the base model's draft only — not the original notes. This is intentional: the edit pass is tested on what it can do with the draft alone, without access to the source material. The implication is that if the base model already dropped substance, Haiku cannot restore it. Phase 2 substance scores therefore reflect the ceiling imposed by the base model, not editing quality alone. This must be kept in mind when interpreting results — low Phase 2 substance is a base model problem, not an editing problem.

**No manual edits.** Hard constraint. No human editing of any output at any stage. If an output is poor after Phase 2, the fix is the prompt, not the post.

**Haiku prompt freeze:** All Haiku generation prompts (`haiku_blog.txt`, `haiku_linkedin.txt`) are frozen after Phase 0 rubric validation and must not change for the remainder of the study. Changing them invalidates the Phase 1 Haiku outputs used as the Phase 2 comparison baseline.

---

## Inputs

**Target:** 30 samples. Same inputs.jsonl schema as Study 1 (enables cross-study comparison).

```json
{
  "id": "sample_001",
  "article_url": "https://...",
  "article_text": "...",
  "notes": "...",
  "domain": "tech",
  "note_complexity": null
}
```

`note_complexity` starts as `null` and is populated in-place by `eval/note_complexity.py` before Phase 1 generation. `run_all.py` reads it directly from `inputs.jsonl`. Every sample is run in both modes — no per-sample mode designation. Min 3 domains.

---

## Calibration Set

Before full-scale evaluation, establish metric anchors.

**Gold standard (ceiling):** 5 samples where Taya writes the output herself from the same notes.

**Null baseline (floor):** 5 article-summary-only outputs (notes ignored entirely).

All model conditions are plotted relative to this band. A metric that doesn't separate the null baseline from any model condition is not discriminating and must be revised before results are interpreted.

Stored in `data/calibration/`. Scored in `notebooks/05_calibration.ipynb` after the rubric validation gate passes.

---

## Metrics

### Primary Metric 1 — Substance Fidelity (Argument-Level)

**Question:** Do the user's ideas from the notes actually appear in the output, or did the model replace them with article summary?

**Why not cosine similarity:** Topic-level cosine similarity cannot detect the core failure mode. A model that replaces "X because Y — which means Z" with "X tends to produce worse outcomes" scores high on topic overlap while having lost the substance entirely.

**Implementation:**

**Step 1 — Argument decomposition.** Each note is split into sentences. Each sentence is typed using keyword pattern matching:

| Component type | Detection rule |
|---|---|
| `logic` | Contains causal/connective keywords: `"because"`, `"which means"`, `"this is why"`, `"as a result"`, `"therefore"`, `"that's why"` |
| `implication` | Contains consequence markers: `"the implication"`, `"this means"`, `"which is why"`, `"so the"`, `"meaning that"` |
| `evidence` | Contains example markers: `"for example"`, `"for instance"`, `"such as"`, `"like when"`, `"consider"`, `"take"` (at sentence start) |
| `claim` | Default — sentence that matches none of the above |

Sentences under 10 characters are discarded. Compound sentences with multiple keyword types are split at the keyword boundary and each part typed separately.

**Step 2 — Embedding and matching.** Embed each typed component with `sentence-transformers` (`all-mpnet-base-v2`). For each component, compute max cosine similarity against all output sentences.

**Step 3 — Scoring.** Report per-component-type mean scores (`claim`, `evidence`, `logic`, `implication`) AND an aggregate mean across all components.

**Step 4 — Flattening flag.** If `claim` score ≥ 0.6 AND (`logic` score + `implication` score) / 2 < 0.4, the sample is flagged as "flattened" — model kept the topic, stripped the reasoning. This is distinct from omission (low aggregate) and reported separately.

**Step 5 — Review flag.** Samples with aggregate < 0.5 flagged for manual review.

**Limitation:** Keyword-based decomposition is imperfect. Notes written without explicit connective language (e.g., terse bullet fragments) will over-classify as `claim`. This is acceptable — it degrades to the cosine similarity baseline for simple notes and produces meaningful component breakdowns for argumentative notes.

### Claim-Type Analysis

Each note component (from argument decomposition above) is additionally tagged by claim type:

| Claim type | Detection rule |
|---|---|
| `question` | Sentence ends with `?` |
| `connection` | Matches `logic` component keywords (see above) |
| `opinion` | Contains hedging or first-person language: `"I think"`, `"I believe"`, `"seems"`, `"arguably"`, `"might be"`, `"I'd say"` |
| `fact` | Default — doesn't match above |

Analyze which types survive transformation per condition. Hypothesis: `connection` and `opinion` are lost first; `fact` is preserved. Report as per-type mean substance score per condition.

### Primary Metric 2 — Voice Fidelity (Two-Tier)

**Question:** Does the output exhibit the linguistic fingerprint documented in the taya-voice skill?

Scored in two tiers. The Tier 1 vs. Tier 2 delta is itself a finding: if Tier 1 is high and Tier 2 is low, the model is mimicking surface voice without capturing structural voice.

**Tier 1 — Surface Markers** (easy to mimic):
- Em-dash restatement present at least once
- `"More specifically"` present at least once
- Contractions present (`it's`, `I'd`, `that's`)
- No banned words: `"clearly"`, `"obviously"`, `"paradigm"`, `"thrilled"`
- No sequential transitions: `"First,"`, `"Second,"`, `"Finally,"`
- No bullet points in prose body
- No markdown headers

**Tier 2 — Structural Patterns** (hard to mimic):
- Example-before-generalization ordering within paragraphs (spaCy: concrete noun phrases precede abstract claims in paragraph-opening sentences)
- Subordinate clause ratio above threshold (spaCy dependency parse: count `advcl`, `relcl`, `acl` relative to total sentences)
- Additive-not-sequential transitions present: `"moreover"`, `"what's more"`, `"and yet"`, `"even so"` — uses `ADDITIVE_TRANSITIONS` list in `voice_rubric.py`
- Concession-redirect present: `"[X]. But [Y]"`, `"[X]. That said, [Y]"`, `"[X]. The problem is [Y]"`
- Paragraphs end on implication: last sentence contains an explicit consequence marker (`"this means"`, `"the implication is"`, `"which is why"`) — replaces the over-permissive `\bwhich\b` check

Report Tier 1 score, Tier 2 score, and delta (T1 − T2) separately. Mode-specific rubrics apply — `score_blog()` and `score_linkedin()` are distinct implementations.

**Validation gate:** Before running at scale, score 10 samples through both tiers AND have Taya score manually. Compute Spearman rank correlation. If ρ < 0.5, iterate on Tier 2 checks. Full study does not proceed until this gate passes. The 10 validation samples come from the calibration set (5 gold standard + 5 null baseline) — these are available before full input collection.

### Note Complexity Stratification

Computed by `eval/note_complexity.py` and written back to `inputs.jsonl` before Phase 1:

```
complexity = (number of claims) × (mean claim length in words) × (1 + has_connecting_logic)
```

`has_connecting_logic` = 1 if any note sentence matches `logic` component keywords, else 0.

Stratify results by tercile: simple / moderate / complex. Report substance and voice metrics per stratum per condition.

### Floor Constraint — Factual Consistency

**Fixed implementation** — replaces original 1500-character truncation:

For each output sentence, retrieve top-3 most relevant article passages via BM25 (`rank_bm25`), then run NLI against those passages using `roberta-large-mnli`. Scores the sentence against its best-matching passage, removing truncation bias for long articles.

Flag outputs with > 20% contradiction rate for manual review. Not included in primary scoring.

---

## Output Schema

All evaluation outputs conform to these schemas. Both phases use the same column set — `phase` field distinguishes them.

**`results/phase1_scores.csv` and `results/phase2_scores.csv`:**

| Column | Type | Description |
|---|---|---|
| `id` | str | Sample ID |
| `phase` | int | 1 or 2 |
| `condition` | str | `gemma`, `qwen`, `mistral`, `haiku`, `best_model_haiku`, `best_model_pre_edit` |
| `mode` | str | `blog` or `linkedin` |
| `domain` | str | From inputs |
| `note_complexity` | float | Computed complexity score |
| `complexity_tier` | str | `simple`, `moderate`, `complex` |
| `substance_aggregate` | float | Mean across all components |
| `substance_claim` | float | Per-type mean score |
| `substance_evidence` | float | Per-type mean score |
| `substance_logic` | float | Per-type mean score |
| `substance_implication` | float | Per-type mean score |
| `substance_flagged` | int | 1 if aggregate < 0.5 |
| `flattening_flagged` | int | 1 if claim ≥ 0.6 and (logic+implication)/2 < 0.4 |
| `claimtype_opinion_score` | float | Mean substance score for opinion-typed components |
| `claimtype_fact_score` | float | Mean substance score for fact-typed components |
| `claimtype_connection_score` | float | Mean substance score for connection-typed components |
| `voice_tier1` | float | Tier 1 score |
| `voice_tier2` | float | Tier 2 score |
| `voice_combined` | float | (tier1 + tier2) / 2 |
| `voice_delta` | float | tier1 − tier2 |
| `contradiction_rate` | float | Fraction of output sentences flagged as contradictions |
| `contradiction_flagged` | int | 1 if rate > 0.20 |
| `edit_scope` | float | Phase 2 only — sentence-level change rate vs. pre-edit output; null in Phase 1 |
| `haiku_input_tokens` | int | Haiku calls only; null for free-tier models |
| `haiku_output_tokens` | int | Haiku calls only; null for free-tier models |
| `haiku_cost_usd` | float | Haiku calls only; null for free-tier models |
| `latency_ms` | int | Generation latency |

---

## Cost Tracking (Haiku Only)

Free-tier models (Gemma, Qwen, Mistral) are $0 by definition — tracking their tokens does not answer the cost research question.

Track Haiku calls only. Log on every Haiku API call, whether standalone generation (Phase 1) or edit pass (Phase 2). Fields map directly to the output schema above.

**Edit scope** is computed during Phase 2 evaluation by `run_all.py`, comparing `best_model_pre_edit_{mode}.jsonl` against `best_model_haiku_{mode}.jsonl` sentence-by-sentence:

```
edit_scope = (sentences added + sentences removed) / total sentences in pre-edit output
```

Near 0 = light polish. Near 1 = near-complete rewrite. Heavy rewrites are a risk signal for substance loss — if edit scope is high and substance drops from pre-edit to post-edit, Haiku overwrote the argument in the process of fixing voice.

**The cost comparison:**
- `haiku` standalone: full generation cost (Phase 1 tokens)
- `best_model_haiku`: edit pass cost only — input is base model output + edit prompt, shorter than a full generation prompt

If `best_model_haiku` matches `haiku` on quality, the pipeline is cheaper. That's a concrete product decision.

---

## Prompt Versioning

Prompts are version-controlled in `prompts/`. Prompts must be stable across a run — changing any prompt file invalidates prior results for that condition.

```
prompts/
├── gemma_blog.txt
├── gemma_linkedin.txt
├── qwen_blog.txt
├── qwen_linkedin.txt
├── mistral_blog.txt
├── mistral_linkedin.txt
├── haiku_blog.txt
├── haiku_linkedin.txt
├── haiku_edit_blog.txt
└── haiku_edit_linkedin.txt
```

Each file contains: system prompt, user prompt template, and sampling parameters (temperature, max tokens).

**Edit prompt structure:** Same instructions for every post in the mode — only `{output}` changes. Blog example:

```
System:
You are editing a blog post draft. Your job is to adjust voice and
length without changing the substance or the author's arguments.

User:
Edit the following blog post draft. Requirements:
- Target length: 500–700 words
- Use em-dashes for restatement at least once
- Include "More specifically" as a zoom-in move at least once
- No bullet points, no markdown headers
- No sequential transitions (First, Second, Finally)
- Paragraphs should end on an implication or consequence
- Do not add information that wasn't in the draft
- Do not remove the author's reasoning or connecting logic

Draft:
{output}
```

LinkedIn edit prompt follows the same structure with mode-appropriate length (200–300 words) and voice targets (hook in first line, CTA at close, shorter paragraphs).

If a post requires changes beyond what the prompt specifies, the fix is to update the prompt — not to handle that post differently. Changing the prompt invalidates all prior Phase 2 results.

**Haiku generation prompt freeze:** `haiku_blog.txt` and `haiku_linkedin.txt` are frozen after Phase 0 rubric validation and cannot change. These prompts produce the Phase 1 Haiku outputs that serve as the Phase 2 comparison baseline — any change breaks that comparison.

---

## Statistical Tests

- **Mode comparison:** Wilcoxon signed-rank (paired, within-subjects) — same input, blog vs. LinkedIn. Valid because every input runs in both modes.
- **Condition comparison:** Wilcoxon signed-rank per metric per condition pair
- **Phase 2 edit comparison:** Wilcoxon signed-rank, `best_model_haiku` vs. `haiku` on same inputs (paired)
- **Effect size:** rank-biserial correlation
- α = 0.05
- **Power note:** N=30 requires large effect sizes (d > 0.7) for significance at α=0.05. Report power estimates alongside results. Flag underpowered tests explicitly — non-significance at this N is not a null finding.

---

## The Core Tradeoff Visualization

For each condition, plot each sample as a point in 2D space:
- X axis: `substance_aggregate`
- Y axis: `voice_combined`

Overlay the calibration band: gold standard ceiling and null baseline floor as reference lines.

Quadrant interpretation:
- Top-right: model preserves both
- Top-left: voice present, substance lost
- Bottom-right: substance present, voice absent
- Scattered: inconsistent

Secondary plots:
- `voice_tier1` vs. `voice_tier2` per condition — surface mimicry vs. structural voice
- Substance by component type per condition — which argument components survive
- Substance by `complexity_tier` — where models fail as input gets harder
- Claim-type survival rates per condition — opinion/fact/connection dropout patterns
- Phase 2: `best_model_haiku` vs. `haiku` on same axes, with `haiku_cost_usd` as bubble size

---

## File Structure

```
blogAI_evals/
├── README.md
├── v2.md                                 # critical review record — do not delete
├── requirements.txt
├── prompts/
│   ├── gemma_blog.txt
│   ├── gemma_linkedin.txt
│   ├── qwen_blog.txt
│   ├── qwen_linkedin.txt
│   ├── mistral_blog.txt
│   ├── mistral_linkedin.txt
│   ├── haiku_blog.txt                    # frozen after Phase 0
│   ├── haiku_linkedin.txt                # frozen after Phase 0
│   ├── haiku_edit_blog.txt
│   └── haiku_edit_linkedin.txt
├── data/
│   ├── inputs.jsonl                      # note_complexity populated before Phase 1
│   ├── calibration/
│   │   ├── gold_standard.jsonl           # Taya-written references (n=5)
│   │   └── null_baseline.jsonl           # article-summary-only outputs (n=5)
│   └── outputs/
│       ├── phase1/
│       │   ├── gemma_blog.jsonl
│       │   ├── gemma_linkedin.jsonl
│       │   ├── qwen_blog.jsonl
│       │   ├── qwen_linkedin.jsonl
│       │   ├── mistral_blog.jsonl
│       │   ├── mistral_linkedin.jsonl
│       │   ├── haiku_blog.jsonl
│       │   └── haiku_linkedin.jsonl
│       └── phase2/
│           ├── best_model_pre_edit_blog.jsonl
│           ├── best_model_pre_edit_linkedin.jsonl
│           ├── best_model_haiku_blog.jsonl
│           └── best_model_haiku_linkedin.jsonl
├── eval/
│   ├── substance_fidelity.py             # argument decomposition + claim-type tagging
│   ├── voice_rubric.py                   # two-tier scoring (score_blog, score_linkedin)
│   ├── factual_consistency.py            # BM25 retrieval + NLI
│   ├── note_complexity.py                # complexity score → writes back to inputs.jsonl
│   ├── cost_tracker.py                   # Haiku token count + pricing
│   └── run_all.py                        # orchestrates all scorers → phase1/phase2 scores.csv
├── research/
│   ├── scraper.py
│   ├── analyze.py
│   └── linkedin_formatter.py
├── notebooks/
│   ├── 01_generate_phase1.ipynb          # 4 models × 2 modes × 30 inputs
│   ├── 02_generate_phase2.ipynb          # reads phase1_winner.txt, runs edit pass
│   ├── 03_evaluate.ipynb                 # runs run_all.py for phase1 and phase2
│   ├── 04_analysis.ipynb                 # all plots and stats
│   ├── 05_calibration.ipynb              # gold standard + null baseline scoring
│   └── 06_platform_style_analysis.ipynb
└── results/
    ├── phase1_winner.txt                 # single line: winning model label
    ├── phase1_scores.csv
    ├── phase2_scores.csv
    └── calibration.csv
```

---

## Execution Sequence

### Phase 0 — Setup and Validation

1. **Confirm model versions** — open `app/config.py` in blogAI repo, extract exact HF model IDs for Gemma, Qwen, Mistral. Record in Model Versions table above.
2. **Run platform style research** — `research/scraper.py` + `research/analyze.py`, review `06_platform_style_analysis.ipynb`, validate and tune `score_linkedin()`.
3. **Fix `voice_rubric.py`** — wire `ADDITIVE_TRANSITIONS` into a check function; fix `_paragraphs_end_on_consequence` (remove `\bwhich\b`, add explicit consequence markers).
4. **Rubric validation gate** — produce 5 gold standard + 5 null baseline outputs (10 samples total). Score through full voice rubric. Have Taya score the same 10 manually. Compute Spearman ρ. If ρ < 0.5, iterate on Tier 2 before proceeding. Do not advance until this passes.
5. **Run `05_calibration.ipynb`** — score calibration set through full pipeline, establish metric ceilings and floors, write `results/calibration.csv`.
6. **Freeze Haiku generation prompts** — `haiku_blog.txt` and `haiku_linkedin.txt` are locked from this point.

### Phase 1 — Standalone Model Comparison

7. **Collect 30 inputs** → `data/inputs.jsonl`. Min 3 domains.
8. **Compute note complexity** — run `eval/note_complexity.py`, which populates `note_complexity` field in `inputs.jsonl` in-place.
9. **Run `01_generate_phase1.ipynb`** — 4 models × 2 modes × 30 inputs = 240 outputs. Log Haiku token counts per sample.
10. **Run `03_evaluate.ipynb` (Phase 1)** — runs `run_all.py` against `data/outputs/phase1/`, writes `results/phase1_scores.csv`.
11. **Run `04_analysis.ipynb` (Phase 1)** — identify Phase 1 winner using combined score formula (substance × 0.6 + voice × 0.4), write winner label to `results/phase1_winner.txt`.

### Phase 2 — Edit Pass

12. **Run `02_generate_phase2.ipynb`** — reads `results/phase1_winner.txt`, runs Haiku edit pass on winner outputs for both modes, saves pre-edit intermediates.
13. **Run `03_evaluate.ipynb` (Phase 2)** — runs `run_all.py` against `data/outputs/phase2/`, computes edit scope for all Phase 2 samples, writes `results/phase2_scores.csv`.
14. **Run `04_analysis.ipynb` (Phase 2)** — `best_model_haiku` vs. `haiku` comparison with cost overlay, edit scope vs. substance loss scatter.
15. **Manual review** — inspect all flagged samples (substance < 0.5 or contradiction > 20% or edit scope > 0.8).

---

## What This Design Answers

| Question | Answered by |
|---|---|
| Which standalone model best preserves substance + voice? | Phase 1 |
| Does mode (blog vs. LinkedIn) shift the tradeoff? | Phase 1 — within-subjects, paired |
| Does a Haiku edit pass improve the best model? | Phase 2 |
| Is the pipeline competitive with Haiku standalone? | Phase 2 — direct comparison using Phase 1 Haiku data |
| At what cost difference? | Phase 2 — Haiku cost tracking (edit tokens vs. generation tokens) |
| Which argument components are lost? | Both phases — claim/evidence/logic/implication breakdown |
| Is voice real or surface mimicry? | Both phases — Tier 1 vs. Tier 2 delta |
| Where in the pipeline does loss occur? | Phase 2 — intermediate pre-edit output scoring |
| Does note complexity predict failure? | Both phases — complexity stratification |
| Which claim types survive transformation? | Both phases — opinion/fact/connection dropout analysis |
| Does heavy editing cause substance loss? | Phase 2 — edit scope vs. substance delta |

---

## Relationship to Study 1

Study 1 (in blogAI repo) tests whether the edit pass fixes a known engineering failure (style constraints). Study 2 tests whether any configuration solves the harder problem: preserving user thought under transformation. The datasets overlap by design — cross-study comparison is intentional.
