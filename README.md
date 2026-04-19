# blogAI_evals — Substance & Voice Fidelity Under Mode Transformation

**Research question:** When an LLM transforms user notes into mode-appropriate content (blog post, LinkedIn post), does it preserve the user's intellectual substance and voice — and is the Qwen→Haiku pipeline competitive with Haiku generating from scratch?

**Full results:** [`results/report.md`](results/report.md)

---

## What This Study Is

An evaluation of LLMs as **thought amplifiers** — tools that reformat existing ideas into new modes without distorting or replacing them. The core tension: mode adaptation vs. substance preservation.

The primary comparison: **`qwen_haiku` vs. `haiku`** — pipeline vs. standalone on quality and cost.

---

## Conditions

3 primary conditions × 2 modes × 27 inputs (3 dropped — paywalled) = **162 outputs** + 54 intermediate pre-edit outputs (`qwen_pre_edit`)

| Condition | Description | Purpose |
|---|---|---|
| `qwen` | Qwen 2.5 7B standalone, no edit | Baseline — what the pipeline starts from |
| `haiku` | Claude Haiku 4.5 standalone | Comparison target |
| `qwen_haiku` | Qwen draft → Haiku voice-correction edit | Primary condition |
| `qwen_pre_edit` | Qwen output before edit pass | Intermediate — isolates Qwen's contribution |

**Models confirmed working on HF free-tier serverless:** Qwen 2.5 7B Instruct, Claude Haiku 4.5. Gemma and Mistral both fail silently on the free-tier serverless endpoint and were excluded.

---

## Inputs

27 samples (30 collected; 3 dropped — paywalled or bot-blocked). Schema:

```json
{
  "id": "sample_001",
  "article_url": "https://...",
  "notes": "...",
  "domain": "tech",
  "note_complexity": 38.0,
  "complexity_tier": "simple"
}
```

`article_text` is fetched at generation time and stored in the output record. `note_complexity` is computed before generation and written back in-place.

---

## Calibration Set

5 gold standard outputs (author-edited Haiku drafts) and 5 null baseline outputs (article summaries, notes ignored). Establishes metric ceilings and floors before any model evaluation. Stored in `data/calibration/`.

Both metrics passed the Spearman ρ ≥ 0.5 validation gate before generation proceeded: substance ρ = 0.926, voice ρ = 0.876.

---

## Metrics

### Substance Fidelity (Primary)

Each note sentence is typed by keyword pattern matching (logic, implication, evidence, claim), embedded with `all-mpnet-base-v2`, and matched against output sentences via cosine similarity. Per-type means and an aggregate score are reported.

**Flattening flag:** `claim` ≥ 0.6 AND (`logic` + `implication`) / 2 < 0.4 — model kept the topic, stripped the reasoning.

### Voice Fidelity (Primary)

Two-tier rubric operationalizing a known author voice fingerprint:

- **Tier 1 — Surface markers** (regex): em-dash, contractions, banned words, no bullets, no markdown headers
- **Tier 2 — Structural patterns** (spaCy): subordinate clause ratio, additive transitions, concession-redirect, paragraph-ending consequence markers

T1 − T2 delta is a finding: high T1, low T2 = surface mimicry without structural voice.

Mode-specific rubrics: `score_blog()` and `score_linkedin()` are distinct.

### Note Complexity Stratification

`complexity = (number of claims) × (mean claim length) × (1 + has_connecting_logic)`

Stratified into simple / moderate / complex terciles.

### Factual Consistency (Floor Check Only)

BM25 retrieval per output sentence → top-3 article passages → NLI via DeBERTa-v3-large-mnli. Flags outputs with > 20% contradiction rate. Not in primary scoring.

---

## Cost Tracking (Haiku Only)

Qwen is free tier. Logged per Haiku call:

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

`edit_scope` = (sentences added + removed) / total pre-edit sentences. Near 0 = light polish. Near 1 = near-complete rewrite.

Total Haiku spend across the full study (27 samples × 2 modes): **$0.084**.

---

## Prompts

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
├── haiku_edit_blog_user.txt
├── haiku_edit_linkedin_system.txt
└── haiku_edit_linkedin_user.txt
```

Edit prompts receive only `{output}` (Qwen's draft) — not the original notes. Haiku generation prompts are frozen after rubric validation.

---

## Output Schema

Per-condition JSONL files in `data/outputs/`:

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

`results/scores.csv` columns: `id`, `condition`, `mode`, `domain`, `note_complexity`, `complexity_tier`, `substance_aggregate`, `substance_claim`, `substance_evidence`, `substance_logic`, `substance_implication`, `substance_flagged`, `flattening_flagged`, `claimtype_opinion_score`, `claimtype_fact_score`, `claimtype_connection_score`, `voice_tier1`, `voice_tier2`, `voice_combined`, `voice_delta`, `contradiction_rate`, `contradiction_flagged`, `edit_scope`, `haiku_input_tokens`, `haiku_output_tokens`, `haiku_cost_usd`, `latency_ms`.

---

## Statistical Tests

- **Primary:** Wilcoxon signed-rank (paired) — `qwen_haiku` vs. `haiku`
- **Edit contribution:** Wilcoxon signed-rank — `qwen` vs. `qwen_haiku`
- **Mode comparison:** Wilcoxon signed-rank (within-subjects) — blog vs. LinkedIn per condition
- **Effect size:** rank-biserial correlation | α = 0.05

N = 27. Non-significance should be interpreted as underpowered, not as evidence of no effect.

---

## File Structure

```
blogAI_evals/
├── README.md
├── requirements.txt
├── .env.example
├── project-log.md
├── study_history.md
├── prompts/                          # 12 prompt files (system + user per condition × mode)
├── data/
│   ├── inputs.jsonl                  # gitignored
│   ├── calibration/                  # gitignored
│   └── outputs/                      # gitignored
├── scripts/
│   ├── generate.py                   # runs all conditions × modes (resumable)
│   ├── test_call.py                  # smoke test — single sample
│   ├── gen_gold_standard_base.py
│   ├── gen_null_baseline.py
│   ├── score_calibration.py
│   ├── make_report.py
│   ├── make_figures.py
│   └── constants.py
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
│   ├── linkedin_formatter.py
│   ├── analysis_results.json
│   └── linkedin_samples.txt          # 51 posts for LinkedIn rubric validation
├── notebooks/
│   ├── 01_analysis.ipynb
│   └── 02_calibration.ipynb
└── results/
    ├── report.md
    ├── scores.csv
    ├── calibration.csv
    └── figures/
```

---

## Running Locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add HF_TOKEN and ANTHROPIC_API_KEY
python scripts/test_call.py          # smoke test
python scripts/generate.py           # full generation run (resumable)
python eval/run_all.py               # produces results/scores.csv
jupyter notebook notebooks/01_analysis.ipynb
```
