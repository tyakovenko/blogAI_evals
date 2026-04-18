# blogAI_evals — Study 2: Substance & Voice Fidelity Under Mode Transformation

> **Scope note:** This plan covers the two models currently functional on HF free tier — Qwen and Haiku. Gemma and Mistral fail silently on the serverless API and fall back to Haiku in production (confirmed 2026-04-14). When those models are connected, the full 4-model comparison plan is in `README_full_plan.md` — the eval logic here applies directly, just add conditions.

---

## Research Question

When an LLM transforms user notes into mode-appropriate content (blog post, LinkedIn post), does it preserve the user's intellectual substance and voice — and is the Qwen→Haiku pipeline competitive with Haiku generating from scratch?

## What This Study Is

An evaluation of LLMs as **thought amplifiers** — tools that reformat existing ideas into new modes without distorting or replacing them. The core tension: mode adaptation vs. substance preservation.

The primary comparison: **`qwen_haiku` vs. `haiku`** — pipeline vs. standalone on quality and cost.

---

## Conditions

3 conditions × 2 modes × 30 inputs = **180 outputs** (plus 60 intermediate pre-edit outputs)

| Condition | Description | Purpose |
|---|---|---|
| `qwen` | Qwen 2.5 7B standalone, no edit | Baseline — what the pipeline starts from |
| `haiku` | Claude Haiku standalone | Comparison target |
| `qwen_haiku` | Qwen output → Haiku edit pass | Primary condition |

**Intermediate capture:** For `qwen_haiku`, Qwen's raw output is saved before the Haiku edit and scored independently. This enables delta analysis — what did the edit add, and what did it damage?

**The three comparisons this enables:**
- `qwen_haiku` vs. `haiku` — is the pipeline competitive with Haiku standalone?
- `qwen` vs. `qwen_haiku` — what does the Haiku edit actually contribute?
- `qwen` vs. `haiku` — raw capability gap between the two models

---

## Inputs

30 samples. Schema:

```json
{
  "id": "sample_001",
  "article_url": "https://...",
  "notes": "...",
  "domain": "tech",
  "note_complexity": null
}
```

`article_text` is fetched at generation time and stored in the output record — not in inputs. `note_complexity` is computed before generation and written back in-place. Min 3 domains.

---

## Calibration Set

5 gold standard outputs and 5 null baseline outputs. Establishes metric ceilings and floors before any model evaluation. Stored in `data/calibration/`.

**Gold standard:** Haiku generates standalone blog + LinkedIn posts for samples 001–005 using the frozen prompts. Taya edits those outputs directly — corrections to voice, substance, and structure become the gold standard. This reflects the actual workflow (Taya edits, not writes from scratch) and avoids circular comparison: the gold standard base is Haiku standalone, which is one of the test conditions but not the primary one (`qwen_haiku` is).

**Null baseline:** Article summary only — notes are ignored entirely. Establishes the floor: a model that reads the article but ignores the user's voice.

A metric that doesn't separate the null baseline from any model condition is not discriminating and must be revised before results are interpreted.

---

## Metrics

### Primary Metric 1 — Substance Fidelity (Argument-Level)

Decompose each note into typed components, embed with `sentence-transformers` (`all-mpnet-base-v2`), score each against the output.

**Component detection (keyword pattern matching):**

| Type | Detection rule |
|---|---|
| `logic` | Contains: `"because"`, `"which means"`, `"this is why"`, `"as a result"`, `"therefore"` |
| `implication` | Contains: `"the implication"`, `"this means"`, `"which is why"`, `"so the"`, `"meaning that"` |
| `evidence` | Contains: `"for example"`, `"for instance"`, `"such as"`, `"like when"` |
| `claim` | Default — matches none of the above |

Report per-type scores and aggregate. **Flattening flag:** `claim` ≥ 0.6 AND (`logic` + `implication`) / 2 < 0.4 — model kept the topic, stripped the reasoning.

**Claim-type tagging** (separate from component type):

| Claim type | Detection |
|---|---|
| `question` | Ends with `?` |
| `connection` | Matches logic keywords |
| `opinion` | Contains: `"I think"`, `"I believe"`, `"seems"`, `"arguably"` |
| `fact` | Default |

### Primary Metric 2 — Voice Fidelity (Two-Tier)

**Tier 1 — Surface markers** (easy to mimic): em-dash present, `"More specifically"` present, contractions present, no banned words (`"clearly"`, `"obviously"`, `"paradigm"`, `"thrilled"`), no sequential transitions, no bullet points, no markdown headers.

**Tier 2 — Structural patterns** (hard to mimic): example-before-generalization ordering (spaCy), subordinate clause ratio above threshold (spaCy dependency parse), additive transitions present (`ADDITIVE_TRANSITIONS` list), concession-redirect present, paragraphs end on implication (explicit consequence markers only — not `\bwhich\b`).

Report Tier 1, Tier 2, and delta separately. The T1−T2 delta is a finding: high T1, low T2 = surface mimicry without structural voice.

Mode-specific rubrics: `score_blog()` and `score_linkedin()` are distinct.

**Validation gate:** Score 10 calibration samples through rubric + manually. Spearman ρ ≥ 0.5 required before full study runs.

### Note Complexity Stratification

`complexity = (number of claims) × (mean claim length) × (1 + has_connecting_logic)`

Stratify by tercile: simple / moderate / complex. Written back to `inputs.jsonl` before generation.

### Floor Constraint — Factual Consistency

BM25 retrieval (`rank_bm25`) per output sentence → top-3 article passages → NLI via `roberta-large-mnli`. Replaces truncation approach. Flag outputs with > 20% contradiction rate. Not in primary scoring.

---

## Cost Tracking (Haiku Only)

Qwen is free tier — no cost tracking needed.

Log on every Haiku call, whether standalone generation (`haiku` condition) or edit pass (`qwen_haiku` condition):

```json
{
  "id": "sample_001",
  "condition": "haiku",
  "mode": "blog",
  "input_tokens": 1306,
  "output_tokens": 510,
  "estimated_cost_usd": 0.000964,
  "latency_ms": 7461,
  "edit_scope": null
}
```

`edit_scope` is null for standalone Haiku and populated for `qwen_haiku` edit pass calls:

```
edit_scope = (sentences added + sentences removed) / total sentences in pre-edit output
```

Near 0 = light polish. Near 1 = near-complete rewrite. High edit scope + substance drop = Haiku overwrote the argument while fixing voice.

**The cost comparison:**
- `haiku` standalone: full generation tokens per sample
- `qwen_haiku`: edit pass tokens only — input is Qwen's output + edit prompt, shorter than full generation

At 30 samples × 2 modes, expected total Haiku cost is under $0.10 for the full study.

---

## Prompts

Blog and LinkedIn prompts live in `prompts/`. Haiku and Qwen share prompt structure — separate files per model per mode. Edit prompts are eval-specific (not in blogAI).

```
prompts/
├── haiku_blog_system.txt
├── haiku_blog_user.txt
├── haiku_linkedin_system.txt
├── haiku_linkedin_user.txt
├── qwen_blog_system.txt
├── qwen_blog_user.txt
├── qwen_linkedin_system.txt
├── qwen_linkedin_user.txt
├── haiku_edit_blog_system.txt
└── haiku_edit_blog_user.txt    # {output} = Qwen's draft
└── haiku_edit_linkedin_system.txt
└── haiku_edit_linkedin_user.txt
```

**Edit prompt constraint:** Same instructions for every post within a mode — only `{output}` changes. If a post needs changes beyond what the prompt specifies, fix the prompt, not the post. Changing the edit prompt invalidates prior `qwen_haiku` results.

**Haiku generation prompt freeze:** `haiku_blog_*.txt` and `haiku_linkedin_*.txt` are frozen after rubric validation. These produce the standalone Haiku outputs used as the comparison baseline — any change breaks that comparison.

---

## Output Schema

Both `scores.csv` columns and per-condition JSONL output files:

**Output JSONL per condition/mode** (`data/outputs/qwen_blog.jsonl`, etc.):

```json
{
  "id": "sample_001",
  "article_text": "...",
  "output": "...",
  "input_tokens": null,
  "output_tokens": null,
  "estimated_cost_usd": null,
  "latency_ms": 1200
}
```

Haiku files have token fields populated; Qwen files have nulls.

**`results/scores.csv` columns:**

| Column | Description |
|---|---|
| `id` | Sample ID |
| `condition` | `qwen`, `haiku`, `qwen_haiku`, `qwen_pre_edit` |
| `mode` | `blog` or `linkedin` |
| `domain` | From inputs |
| `note_complexity` | Computed score |
| `complexity_tier` | `simple`, `moderate`, `complex` |
| `substance_aggregate` | Mean across all components |
| `substance_claim` | Per-type mean score |
| `substance_evidence` | Per-type mean score |
| `substance_logic` | Per-type mean score |
| `substance_implication` | Per-type mean score |
| `substance_flagged` | 1 if aggregate < 0.5 |
| `flattening_flagged` | 1 if claim ≥ 0.6 and (logic+implication)/2 < 0.4 |
| `claimtype_opinion_score` | Mean substance score for opinion components |
| `claimtype_fact_score` | Mean substance score for fact components |
| `claimtype_connection_score` | Mean substance score for connection components |
| `voice_tier1` | Tier 1 score |
| `voice_tier2` | Tier 2 score |
| `voice_combined` | (tier1 + tier2) / 2 |
| `voice_delta` | tier1 − tier2 |
| `contradiction_rate` | Fraction of output sentences flagged |
| `contradiction_flagged` | 1 if rate > 0.20 |
| `edit_scope` | `qwen_haiku` only — sentence change rate vs. pre-edit |
| `haiku_input_tokens` | Haiku conditions only |
| `haiku_output_tokens` | Haiku conditions only |
| `haiku_cost_usd` | Haiku conditions only |
| `latency_ms` | Generation latency |

---

## Statistical Tests

- **Primary comparison:** Wilcoxon signed-rank (paired) — `qwen_haiku` vs. `haiku`, same inputs
- **Mode comparison:** Wilcoxon signed-rank (paired, within-subjects) — blog vs. LinkedIn per condition
- **Edit contribution:** Wilcoxon signed-rank — `qwen` vs. `qwen_haiku`, same inputs
- **Effect size:** rank-biserial correlation
- α = 0.05
- **Power note:** N=30 requires d > 0.7 for significance. Flag underpowered tests — non-significance is not a null finding.

---

## Visualizations

**Primary:** 2D scatter per condition — X: substance aggregate, Y: voice combined. Calibration band overlaid (gold standard ceiling, null baseline floor).

**Secondary:**
- Tier 1 vs. Tier 2 per condition — surface mimicry gap
- Substance by component type — which components survive per condition
- `qwen` → `qwen_pre_edit` → `qwen_haiku` delta — what the edit actually changes
- Edit scope vs. substance delta scatter — does heavy editing cause substance loss?
- Cost: `haiku` generation tokens vs. `qwen_haiku` edit tokens per sample

---

## File Structure

```
blogAI_evals/
├── README.md                           # this plan (Qwen + Haiku)
├── README_full_plan.md                 # backup — 4-model plan for when Gemma/Mistral are connected
├── v2.md                               # original critical review — do not delete
├── requirements.txt
├── .env                                # ANTHROPIC_API_KEY, HF_TOKEN
├── .env.example
├── prompts/
│   ├── haiku_blog_system.txt
│   ├── haiku_blog_user.txt
│   ├── haiku_linkedin_system.txt
│   ├── haiku_linkedin_user.txt
│   ├── qwen_blog_system.txt
│   ├── qwen_blog_user.txt
│   ├── qwen_linkedin_system.txt
│   ├── qwen_linkedin_user.txt
│   ├── haiku_edit_blog_system.txt
│   ├── haiku_edit_blog_user.txt
│   ├── haiku_edit_linkedin_system.txt
│   └── haiku_edit_linkedin_user.txt
├── data/
│   ├── inputs.jsonl
│   ├── calibration/
│   │   ├── gold_standard.jsonl
│   │   └── null_baseline.jsonl
│   └── outputs/
│       ├── qwen_blog.jsonl
│       ├── qwen_linkedin.jsonl
│       ├── haiku_blog.jsonl
│       ├── haiku_linkedin.jsonl
│       ├── qwen_pre_edit_blog.jsonl
│       ├── qwen_pre_edit_linkedin.jsonl
│       ├── qwen_haiku_blog.jsonl
│       └── qwen_haiku_linkedin.jsonl
├── scripts/
│   ├── generate.py                     # runs all 3 conditions × 2 modes
│   └── test_call.py                    # smoke test — single sample, all conditions
├── eval/
│   ├── substance_fidelity.py
│   ├── voice_rubric.py
│   ├── factual_consistency.py
│   ├── note_complexity.py
│   ├── cost_tracker.py
│   └── run_all.py
├── research/
│   ├── scraper.py
│   ├── analyze.py
│   └── linkedin_formatter.py
├── notebooks/
│   ├── 01_analysis.ipynb               # 2D plots, per-mode breakdowns, stats
│   └── 02_calibration.ipynb            # gold standard + null baseline scoring
└── results/
    ├── scores.csv
    └── calibration.csv
```

---

## Execution Sequence

### Setup
1. Fix `voice_rubric.py` — wire `ADDITIVE_TRANSITIONS`, fix `_paragraphs_end_on_consequence`
2. Write Qwen prompt files (`prompts/qwen_*.txt`) — test against Qwen outputs
3. Write edit prompt files (`prompts/haiku_edit_*.txt`)
4. Produce calibration set (5 gold standard + 5 null baseline)
5. **Rubric validation gate** — 10 calibration samples scored by rubric + manually, Spearman ρ ≥ 0.5
6. Run `02_calibration.ipynb` → `results/calibration.csv`
7. **Freeze Haiku generation prompts** after validation

### Data Collection
8. Collect 30 inputs → `data/inputs.jsonl` (min 3 domains)
9. Run `eval/note_complexity.py` → populates `note_complexity` in `inputs.jsonl`

### Generation
10. Run `scripts/generate.py` — 3 conditions × 2 modes × 30 inputs = 180 outputs + 60 pre-edit intermediates

### Evaluation
11. Run `eval/run_all.py` → `results/scores.csv`
12. Manual review — flagged samples (substance < 0.5 or contradiction > 20% or edit scope > 0.8)

### Analysis
13. Run `notebooks/01_analysis.ipynb` — all plots and stats

---

## What This Study Answers

| Question | How |
|---|---|
| Is `qwen_haiku` competitive with `haiku` standalone? | Primary paired comparison |
| Does the edit pass improve on Qwen alone? | `qwen` vs. `qwen_haiku` paired |
| At what cost difference? | Haiku edit tokens vs. full generation tokens |
| Does mode (blog vs. LinkedIn) shift the tradeoff? | Within-subjects, paired across modes |
| Which argument components are lost per condition? | Component-level substance breakdown |
| Is voice real or surface mimicry? | Tier 1 vs. Tier 2 delta |
| Where in the pipeline does loss occur? | `qwen_pre_edit` intermediate scoring |
| Does heavy editing cause substance loss? | Edit scope vs. substance delta |
| Does note complexity predict failure? | Complexity stratification |

---

## Relationship to Study 1

Study 1 tests whether the edit pass fixes a known engineering failure (style constraints). Study 2 tests whether any configuration preserves user thought under transformation. Datasets overlap by design.
