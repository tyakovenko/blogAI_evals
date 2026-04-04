# blogAI_evals — Study 2: Substance & Voice Fidelity Under Mode Transformation

## Research Question

When an LLM transforms user notes into mode-appropriate content (blog post, LinkedIn post), does it preserve the user's intellectual substance and voice — and which model configuration best maintains that fidelity under mode transformation?

## What This Study Is Not

This is not a summarization study. The article is not the primary source of content — the user's notes are. The article enters the evaluation through one question only: did the model fabricate something that contradicts the source? That is a floor constraint (hallucination check), not a scored dimension.

## What This Study Is

An evaluation of LLMs as **thought amplifiers** — tools that reformat existing ideas into new modes without distorting or replacing them. The core tension: mode adaptation (making output blog- or LinkedIn-appropriate) vs. substance preservation (keeping the user's actual argument intact).

---

## Conditions

Same 3-way comparison as Study 1, same inputs:

| Label | Description |
|---|---|
| `qwen_raw` | Qwen 2.5 7B, mode-appropriate style prompt, no edit |
| `haiku_only` | Claude Haiku, same prompt, standalone |
| `qwen_haiku` | Qwen raw → Claude Haiku edit pass |

Two output modes tested per sample:
- `blog` — analytical blog post register
- `linkedin` — social/professional register

This gives 6 output files total (3 conditions × 2 modes).

---

## Inputs

**Target:** 30–50 samples. Same inputs.jsonl schema as Study 1.

```json
{
  "id": "sample_001",
  "article_url": "https://...",
  "article_text": "...",
  "notes": "...",
  "domain": "tech",
  "mode": "blog"
}
```

Each sample has a designated mode. Aim for ~60% blog, ~40% linkedin to reflect realistic usage.

---

## Metrics

### Primary Metric 1 — Substance Fidelity
**Question:** Do the user's ideas from the notes actually appear in the output, or did the model replace them with article summary?

**Implementation:**
- Embed `notes` and `output` using `sentence-transformers` (`all-mpnet-base-v2` — stronger than MiniLM for semantic precision)
- Segment notes into individual claims (split on newlines / sentence boundaries)
- For each claim, compute cosine similarity against all output sentences
- Score = mean of per-claim max similarities
- Flag samples where score < 0.5 as "substance replaced" — inspect manually

**What low scores mean:** The model summarized the article instead of amplifying the notes. This is the primary failure mode.

### Primary Metric 2 — Voice Fidelity
**Question:** Does the output exhibit the linguistic fingerprint documented in the taya-voice skill?

Voice is operationalized as a scored rubric, not a subjective judgment. Scored separately per mode (blog register vs. analytical register have different expected features).

**Blog mode rubric (0–1, each item binary):**
- No bullet points in body prose
- No markdown headers
- Contractions present (`it's`, `I'd`, `that's`)
- Direct address (`you`) present
- Ends with open question or thread (not a conclusion statement)
- Opinions stated before unpacking (not thesis-last)
- No `"clearly"`, `"obviously"`, `"paradigm"`, `"thrilled"` present

**Analytical/blog post rubric (0–1, each item binary):**
- No bullet points in analytical prose
- Em-dash restatement present at least once
- `"More specifically"` present at least once
- Thesis does not appear in sentence 1
- Hedging words present in body (`"I think"`, `"I believe"`, `"might"`, `"seems"`)
- No sequential transitions (`"First"`, `"Second"`, `"Finally"`)
- Every paragraph has an implication (last sentence is consequence, not noun phrase)

Score = items passing / total items per mode rubric.

**Implementation:** Regex + spaCy for sentence-level detection. No LLM-as-judge — rules are precise enough to automate.

### Floor Constraint — Factual Consistency
**Question:** Does the output contradict the article?

**Implementation:** Same as Study 1 — NLI via `roberta-large-mnli`. Output sentences scored for entailment against article text. Flag outputs with > 20% contradiction rate for manual review. Not included in primary scoring.

---

## The Core Tradeoff Visualization

For each condition, plot each sample as a point in 2D space:
- X axis: Substance Fidelity score
- Y axis: Voice Fidelity score

The shape of the point cloud per condition is the finding:
- Clustered top-right: model preserves both
- Scattered: inconsistent
- Top-left: voice present, substance lost (model rewrote the argument)
- Bottom-right: substance present, voice absent (model preserved ideas, ignored style)

The tradeoff question: does mode adaptation (blog vs. linkedin) shift the distribution? Do models that score well on voice sacrifice substance?

---

## File Structure

```
blogAI_evals/
├── README.md
├── requirements.txt
├── data/
│   ├── inputs.jsonl
│   └── outputs/
│       ├── qwen_raw_blog.jsonl
│       ├── qwen_raw_linkedin.jsonl
│       ├── haiku_only_blog.jsonl
│       ├── haiku_only_linkedin.jsonl
│       ├── qwen_haiku_blog.jsonl
│       └── qwen_haiku_linkedin.jsonl
├── eval/
│   ├── substance_fidelity.py     # sentence-transformers claim matching
│   ├── voice_rubric.py           # regex + spaCy voice fingerprint scorer
│   ├── factual_consistency.py    # NLI hallucination floor check
│   └── run_all.py                # → results/scores.csv
├── notebooks/
│   ├── 01_generate.ipynb         # Colab — 3 conditions × 2 modes
│   ├── 02_evaluate.ipynb         # runs full eval pipeline
│   └── 03_analysis.ipynb         # 2D tradeoff plots, per-mode breakdowns
└── results/
    └── scores.csv                # condition × mode × metric × sample_id
```

---

## Execution Sequence

1. **Collect inputs** — 30–50 real BlogAI runs, logged to `data/inputs.jsonl`. Same dataset as Study 1 where possible (enables cross-study comparison).
2. **Run `01_generate.ipynb`** on Colab — generates all 6 output files
3. **Run `02_evaluate.ipynb`** — substance fidelity, voice rubric, hallucination check → `results/scores.csv`
4. **Run `03_analysis.ipynb`** — 2D scatter plots per condition, per-mode breakdown, Wilcoxon tests on primary metrics
5. **Manual review** — inspect flagged samples (substance score < 0.5 or hallucination rate > 20%)

---

## Statistical Tests

- **Wilcoxon signed-rank** (paired, non-parametric) per metric per condition pair
- **Effect size:** rank-biserial correlation
- **Mode comparison:** Mann-Whitney U between blog and linkedin distributions per condition
- α = 0.05

---

## Expected Findings

The mode comparison (blog vs. linkedin) is likely to be the most interesting axis:
- LinkedIn mode may pressure models toward generic professional tone, killing voice
- Blog mode may allow more latitude, preserving more of both substance and voice
- Qwen's style failures (documented in findings.md) predict low voice scores in both modes
- The Haiku edit pass hypothesis: fixes voice, unknown effect on substance — this is the open question

If substance scores diverge between `haiku_only` and `qwen_haiku` at similar voice scores, the finding is: the pipeline adds voice without recovering substance. That would suggest the Qwen generation step is already losing the notes before Haiku sees the output.

---

## Relationship to Study 1

Study 1 (in blogAI repo) tests whether the edit pass fixes a known engineering failure (style constraints). Study 2 tests whether any configuration solves the harder problem: preserving user thought under transformation. The datasets overlap by design — cross-study comparison is intentional.
