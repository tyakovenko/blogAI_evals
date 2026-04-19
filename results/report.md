# Substance and Voice Fidelity Under Mode Transformation
## An Internal Evaluation of LLM Thought Amplification Pipelines

*Generated 2026-04-18 — internal use only*

---

## Abstract

This study evaluates whether LLMs can transform user notes into mode-appropriate content (blog posts, LinkedIn posts) without distorting or replacing the user's intellectual substance and voice. Three conditions were compared across 27 input samples and two output modes: Qwen 2.5 7B standalone, Claude Haiku 4.5 standalone, and a Qwen→Haiku pipeline in which Haiku applies a voice-correction edit pass to Qwen's draft. Substance fidelity was measured using sentence-transformer cosine similarity against the original notes at the argument-component level; voice fidelity was assessed via a two-tier regex/spaCy rubric operationalizing a known linguistic fingerprint. A calibration set of five samples established metric ceilings (gold standard: user-edited Haiku outputs) and floors (null baseline: article summaries ignoring notes). Both rubrics passed a Spearman ρ ≥ 0.5 validation gate before generation proceeded. Results and their implications for the Qwen→Haiku pipeline design are reported below.

## 1. Introduction

BlogAI is a content generation tool designed to amplify a user's existing thinking into mode-appropriate posts — not to summarize source articles or generate generic content. The user provides notes alongside an article link; the model's job is to build the output around those notes while using the article only as backdrop.

This study emerged from a reframing of an earlier evaluation design. The original question was engineering-oriented: *does the Haiku edit pass fix a known style constraint violation?* That framing assumes the fix works and asks only whether the pipeline produces the right surface markers. The more important question — and the one this study addresses — is: *what does success on this task actually look like?* Does any configuration of the pipeline preserve the user's reasoning, not just their stylistic surface? This shift from validation to discovery motivated the two-metric design (substance fidelity + voice fidelity) and the calibration approach that grounds both metrics against real human output.

Two production constraints shaped the study design. First, Gemma and Mistral both fail silently on the HuggingFace free-tier serverless endpoint, falling back to Haiku without any visible error — meaning BlogAI has effectively been a Qwen + Haiku system throughout its production lifetime. The evaluation model set was revised accordingly. Second, the Haiku edit prompt operates on Qwen's draft only; it does not receive the original notes. Any substance already lost in Qwen's generation cannot be recovered by the edit pass.

## 2. Research Questions

1. Is the Qwen→Haiku pipeline competitive with Haiku standalone on substance and voice fidelity?
2. What does the Haiku edit pass contribute — does it improve substance, preserve it, or trade it away for voice?
3. Does mode (blog vs. LinkedIn) shift the fidelity tradeoff?
4. Which argument component types (claim, evidence, logic, implication) are most at risk of loss?
5. At what cost difference does the pipeline operate relative to full Haiku generation?

## 3. Methodology

### 3.1 Conditions

| Condition | Description | Role |
|---|---|---|
| Qwen standalone | Qwen 2.5 7B Instruct (HF free tier) | Baseline — pipeline starting point |
| Haiku standalone | Claude Haiku 4.5 | Comparison target |
| Qwen pre-edit | Qwen output before Haiku edit pass | Intermediate — isolates Qwen's contribution |
| Qwen→Haiku | Qwen draft → Haiku voice-correction edit | Primary condition |

The intermediate pre-edit capture allows delta analysis at every stage: what Qwen produced, what the edit changed, and what the final pipeline delivered — compared against Haiku generating from the same inputs with no intermediary.

**Model selection rationale.** Qwen 2.5 7B Instruct is the de facto production model in BlogAI — Gemma and Mistral both fail silently on the HuggingFace free-tier serverless endpoint, reverting to Qwen without any visible error. Evaluating Qwen is evaluating what actually runs for users. It was selected over larger open-source alternatives because it is the highest-performing model available under the free-tier serverless constraint; deploying a larger model would require paid HuggingFace inference, which falls outside the production setup under study.

Claude Haiku 4.5 was selected as both the standalone comparison and the edit-pass model because it is the configured downstream model in the production pipeline. Among Claude models, Haiku was chosen over Sonnet and Opus for cost reasons: at approximately $0.001 per sample (observed), it is viable for a per-post pipeline in production. Haiku's instruction-following on stylistic tasks is sufficient to operationalize a detailed voice specification; Sonnet and Opus would improve output quality marginally but at 5–10× the cost, which is not justified for a thin edit pass on short-form content. The Qwen→Haiku pipeline configuration was not a design choice for this study — it is the production system, evaluated as-is.

### 3.2 Prompts

Prompts are model-specific. Haiku prompts include full voice specification (em-dash usage, additive transitions, concession-redirect structure, paragraph-ending consequence markers, banned words). Qwen prompts focus on substance fidelity only — voice is handled downstream by the edit pass, and Qwen's instruction-following on stylistic detail is weaker. Edit prompts are short parameterized nudges, not formatting specs; the only variable is `{output}` (Qwen's draft). Edit prompts do not receive the original notes.

Haiku generation prompts were frozen after the rubric validation gate passed. Changing them after that point would invalidate the comparison baseline.

### 3.3 Metrics

**Substance fidelity** measures whether the user's argument components from the notes appear in the output. Each note sentence is typed by keyword pattern matching (logic, implication, evidence, claim), embedded with `all-mpnet-base-v2`, and matched against output sentences via cosine similarity. Per-type means and an aggregate score are reported. A flattening flag is raised when claim score is high but logic+implication mean is low — the model kept the topic but stripped the reasoning.

`all-mpnet-base-v2` was selected over alternatives (BERTScore, ROUGE, exact-match) because the task requires semantic preservation of argument *meaning*, not verbatim reproduction. BERTScore operates at the token level and is sensitive to surface paraphrase, which would penalize legitimate reformulations that preserve the underlying reasoning. ROUGE penalizes paraphrase even more aggressively and is not suited to argumentative text. Cosine similarity over sentence embeddings allows models to rephrase freely while still scoring well if the argument is preserved — which matches the actual requirement. `all-mpnet-base-v2` was chosen over `all-MiniLM-L6-v2` (lighter but less accurate on STS benchmarks) and API-based embedding models (which would add cost and latency to the evaluation pipeline) based on STS benchmark performance at a free, local-inference constraint.

**Voice fidelity** operationalizes a known linguistic fingerprint using regex and spaCy dependency parsing. Blog scoring uses two tiers: Tier 1 (surface markers — em-dash, contractions, banned words, no bullet points) and Tier 2 (structural patterns — subordinate clause ratio, additive transitions, concession-redirect, paragraph endings on consequence). The Tier 1 − Tier 2 delta is a finding in itself: high delta indicates surface mimicry without structural voice. LinkedIn uses a separate 7-check rubric (hook velocity, short paragraphs, line break density, conversational markers, no formal transitions, direct address, CTA ending).

A custom regex/spaCy rubric was required because the voice fingerprint being measured is author-specific — no off-the-shelf metric (ROUGE, BERTScore, genre style classifiers) operationalizes a known individual's linguistic patterns. The two-tier design is deliberate: Tier 1 surface markers are reliably detectable via regex; Tier 2 structural patterns require dependency parsing via spaCy and are what define voice at depth rather than at the surface. Separating the tiers directly surfaces mimicry — a model that passes Tier 1 without Tier 2 is replicating surface form, not structural voice. An LLM-as-judge approach was considered and rejected: it is harder to validate, harder to reproduce, and adds evaluation cost without providing the diagnostic granularity of a rubric with explicit named checks. The ρ ≥ 0.5 validation gate against human-edited gold outputs was required precisely because the rubric is custom — it must be shown to correlate with human quality judgments before its rankings can be trusted.

**Factual consistency** is a floor check only — not a primary metric. BM25 retrieval over article passages followed by NLI (DeBERTa-v3) flags outputs with >20% sentence contradiction rate.

BM25 retrieval + DeBERTa-v3 NLI was selected over simpler approaches (full-document NLI, keyword overlap) because per-sentence grounding in the most relevant passage avoids false contradiction flags from comparing output sentences against unrelated article sections. DeBERTa-v3-large-mnli is a strong, locally-runnable NLI model with high MNLI benchmark performance; avoiding an API call here keeps the evaluation pipeline self-contained. The >20% sentence contradiction threshold was set as a floor constraint — not a ranking signal — to catch egregious hallucination without requiring high precision at the boundary. Factual consistency is secondary because BlogAI outputs are explicitly notes-first, not article summaries; the article is backdrop, not the primary source of truth.

### 3.4 Calibration

Before generation, five calibration samples were scored through both rubrics to validate that the metrics discriminate meaningfully. Gold standard: the author edited Haiku standalone outputs for five samples — corrections to voice, substance, and structure became the ceiling. This reflects the actual workflow (editing, not writing from scratch) and avoids circular comparison with the primary test condition. Null baseline: article summaries generated with notes withheld entirely, establishing the floor.

Both metrics passed the Spearman ρ ≥ 0.5 gate: substance ρ = 0.926 (p = 0.0000), voice ρ = 0.876 (p = 0.0000). Direction was correct for all 10 paired samples on both metrics — gold consistently scored above null with no exceptions.

## 4. Data

Inputs: 27 samples (originally 30; see edge cases below). Each sample consists of an article URL and the author's handwritten notes on that article. Notes vary in length, domain, and complexity. Note complexity was quantified as `(n_claims) × (mean_claim_length) × (1 + has_connecting_logic)` and stratified into simple/moderate/complex terciles before generation.

**Note complexity distribution:**

- Simple: 11 samples
- Moderate: 10 samples
- Complex: 9 samples

### 4.1 Edge Cases and Data Quality Notes

**Dropped samples (3):** `sample_006` (Medium — paywalled), `sample_024` (Axios — bot detection, no manual access), `sample_027` (NYT — paywalled). All three conditions are missing for these samples. They are excluded from all analyses.

**Manually copied article text (5):** `sample_009` (VentureBeat), `sample_010` (Medium), `sample_011` (SBS), `sample_020` (StartupNation), `sample_021` (ScienceDirect). trafilatura was blocked by bot detection or access restrictions on these URLs. Article text was manually copied and stored in `data/articles/{id}.txt`. These samples are included in all analyses; the article source is noted.

**`sample_020` ad noise:** The manually copied text for this sample includes advertising copy interspersed with article content. The factual consistency scores for `sample_020` may be elevated as a result — the NLI model may flag ad copy as contradictions. Substance scores are unaffected (substance is measured against notes, not the article).

**`sample_017` (truncated notes):** This sample has unusually short notes compared to the article length. This is a real input edge case, not a data quality issue — it tests substance fidelity when the user provides minimal signal.

**HuggingFace downtime:** The HF serverless API experienced downtime during generation. Three samples (001, 003, 004) required Qwen retries. The script is resumable and all outputs were ultimately generated successfully for all 27 samples.

## 5. Results

### 5.1 Substance Fidelity

**Blog posts:**

| Condition | Mean | SD | N |
|---|---|---|---|
| Qwen (standalone) | 0.690 | 0.077 | 27 |
| Haiku (standalone) | 0.659 | 0.066 | 27 |
| Qwen pre-edit | 0.690 | 0.077 | 27 |
| Qwen→Haiku | 0.679 | 0.071 | 27 |

Qwen→Haiku vs Haiku (paired Wilcoxon, n=27): p = 0.095, non-significant, r = 0.370. Qwen→Haiku mean (0.679) was higher than Haiku (0.659).

Qwen vs Qwen→Haiku edit contribution (n=27): p = 0.099, non-significant, r = 0.714. Haiku edit degraded substance (0.690 → 0.679).

**Linkedin posts:**

| Condition | Mean | SD | N |
|---|---|---|---|
| Qwen (standalone) | 0.679 | 0.090 | 27 |
| Haiku (standalone) | 0.624 | 0.079 | 27 |
| Qwen pre-edit | 0.679 | 0.090 | 27 |
| Qwen→Haiku | 0.663 | 0.082 | 27 |

Qwen→Haiku vs Haiku (paired Wilcoxon, n=27): p = 0.030, r = 0.476. Qwen→Haiku mean (0.663) was higher than Haiku (0.624).

Qwen vs Qwen→Haiku edit contribution (n=27): p = 0.067, non-significant, r = 0.545. Haiku edit degraded substance (0.679 → 0.663).

### 5.2 Argument Flattening

Flattening flag: claim score ≥ 0.6 and (logic + implication) / 2 < 0.4 — the model preserved the topic but dropped the reasoning structure.

| Condition | Blog | LinkedIn |
|---|---|---|
| Qwen (standalone) | 7% (2/27) | 4% (1/27) |
| Haiku (standalone) | 7% (2/27) | 4% (1/27) |
| Qwen pre-edit | 7% (2/27) | 4% (1/27) |
| Qwen→Haiku | 7% (2/27) | 4% (1/27) |

### 5.3 Substance by Component Type — Blog

Logic and implication components test whether the model preserves causal reasoning. A pattern of high claim scores with low logic/implication scores is the flattening signature.

| Condition | Claim | Evidence | Logic | Implication |
|---|---|---|---|---|
| Qwen (standalone) | 0.672 | 0.669 | 0.764 | — |
| Haiku (standalone) | 0.652 | 0.695 | 0.668 | — |
| Qwen pre-edit | 0.672 | 0.669 | 0.764 | — |
| Qwen→Haiku | 0.663 | 0.673 | 0.743 | — |

### 5.4 Voice Fidelity

**Blog — Tier 1 (surface) vs Tier 2 (structural):**

The delta between tiers identifies surface mimicry: a model that passes surface checks (no bullets, has contractions) without passing structural ones (subordinate clause ratio, concession-redirect, paragraph-ending consequence markers) is imitating form, not voice.

| Condition | Tier 1 | Tier 2 | Δ (T1−T2) |
|---|---|---|---|
| Qwen (standalone) | 0.667 | 0.548 | 0.119 |
| Haiku (standalone) | 0.741 | 0.800 | -0.059 |
| Qwen pre-edit | 0.667 | 0.548 | 0.119 |
| Qwen→Haiku | 0.852 | 0.956 | -0.104 |

**LinkedIn:**

| Condition | Voice score |
|---|---|
| Qwen (standalone) | 0.571 |
| Haiku (standalone) | 0.841 |
| Qwen pre-edit | 0.571 |
| Qwen→Haiku | 0.884 |

### 5.5 Factual Consistency (Floor Check)

Outputs flagged at >20% sentence contradiction rate against source article. This is a floor constraint — not a primary finding. Note: `sample_020` factual scores may be inflated due to ad copy in the article text.

| Condition | Mode | Flagged | Mean contradiction rate |
|---|---|---|---|
| Qwen (standalone) | Blog | 9/27 | 0.153 |
| Qwen (standalone) | Linkedin | 7/27 | 0.122 |
| Haiku (standalone) | Blog | 23/27 | 0.272 |
| Haiku (standalone) | Linkedin | 21/27 | 0.319 |
| Qwen pre-edit | Blog | 9/27 | 0.153 |
| Qwen pre-edit | Linkedin | 7/27 | 0.122 |
| Qwen→Haiku | Blog | 7/27 | 0.144 |
| Qwen→Haiku | Linkedin | 8/27 | 0.174 |

### 5.6 Cost Analysis

Qwen is free tier. Cost tracking applies to Haiku calls only — whether standalone generation (full prompt) or edit pass (shorter: Qwen draft + edit instruction).

| Condition | Mode | Mean cost (USD) | Total (USD) |
|---|---|---|---|
| Haiku (standalone) | Blog | $0.001034 | $0.027925 |
| Haiku (standalone) | Linkedin | $0.000636 | $0.017179 |
| Qwen→Haiku | Blog | $0.001032 | $0.027860 |
| Qwen→Haiku | Linkedin | $0.000396 | $0.010703 |

Total Haiku spend across all 27 samples and both modes: **$0.083667**

### 5.7 Edit Scope — Qwen→Haiku

Edit scope = (sentences added + removed) / total pre-edit sentences. High scope + substance drop = Haiku rewrote the argument to fix voice.

| Mode | Mean | Median | % heavy edits (scope > 0.5) |
|---|---|---|---|
| Blog | 1.188 | 1.192 | 100% |
| Linkedin | 1.772 | 2.000 | 100% |

### 5.8 Substance by Note Complexity

Does more complex reasoning in the notes predict worse preservation?

| Condition | Mode | Simple | Moderate | Complex |
|---|---|---|---|---|
| Haiku (standalone) | Blog | 0.648 | 0.676 | 0.650 |
| Haiku (standalone) | Linkedin | 0.620 | 0.636 | 0.614 |
| Qwen→Haiku | Blog | 0.651 | 0.694 | 0.693 |
| Qwen→Haiku | Linkedin | 0.630 | 0.690 | 0.666 |

## 6. Discussion

The primary question — whether the Qwen→Haiku pipeline is competitive with Haiku standalone — should be read against the cost structure. The pipeline's Haiku call operates on a shorter input (Qwen's draft + a brief edit prompt) rather than a full generation prompt, making it meaningfully cheaper per sample. Whether that cost advantage is worth any substance tradeoff depends on where substance loss occurs: in Qwen's generation, in the edit pass, or in both.

The intermediate pre-edit condition isolates this. If Qwen already preserves substance well and the edit pass is roughly neutral, the pipeline is clearly preferable on cost. If the edit pass actively damages substance — high edit scope with substance drop — the pipeline design needs revision, likely by adding original notes back into the edit prompt.

The voice tier gap (Tier 1 − Tier 2) is a second diagnostic. A large positive delta means a condition is hitting surface markers (em-dash, contractions) without the structural patterns that define the voice at depth (subordinate clause ratio, concession-redirect moves, paragraph-ending consequence markers). This is the mimicry signal — it looks right on the surface but reads differently to someone who knows the author's writing well.

The flattening flag captures a distinct failure mode: the model kept the topic and the claim, but stripped the causal reasoning that made the claim interesting. High flattening rates on logic and implication components indicate the model is summarizing the conclusion without preserving the argument that led there.

## 7. Limitations

**Sample size:** N = 27. At this size, Wilcoxon signed-rank requires approximately d > 0.65 for significance at α = 0.05. Non-significant results should be interpreted as underpowered, not as evidence of no effect.

**Article source heterogeneity:** Five samples use manually copied article text (bot detection blocked automated fetching); three samples were dropped entirely (paywalled). The manually copied samples are subject to copy-paste accuracy and, in the case of `sample_020`, include advertising copy that may inflate factual contradiction scores.

**Voice rubric precision:** The rubric passed the ρ ≥ 0.5 validation gate but is not a high-precision instrument. Surface markers in Tier 1 are reliably detectable; Tier 2 structural checks (particularly subordinate clause ratio via spaCy) are approximations. The rubric was validated against the author's own edited writing, which constrains its generalizability.

**Model reproducibility:** Qwen outputs on the HuggingFace free-tier serverless API are not fully reproducible — temperature was set to 0.85 and no seed was specified. Re-running generation would produce different Qwen outputs.

**Edit prompt constraint:** The Haiku edit prompt does not receive the original notes. Any substance already dropped by Qwen cannot be recovered by the edit pass. This is a known design constraint, not an oversight — changing it would require re-generating all qwen_haiku outputs and invalidate cross-condition comparison.

**LinkedIn rubric misalignment (empirical):** Feature analysis on 51 real LinkedIn posts reveals three rubric checks that almost never fire on real posts: `hook_in_first_sentence` (operationalized as first-person reaction verb — present in 1.9% of posts; real hook signal is short first sentence, median 11 words), `conversational_markers` (specific phrases — "honestly", "here's the thing", "to be fair", "lowkey" — present in 1.9%), and `ends_with_question_or_cta` (question or explicit CTA verb — 1.9% each). `has_direct_address` ("you") is present in only 53.8% of posts, so it also over-penalizes valid LinkedIn content. LinkedIn scores in this study were computed with the original uncorrected rubric — they underestimate platform fidelity for all conditions. Rubric correction is deferred to future work.

## 8. Future Work

**LinkedIn rubric refinement:** Feature analysis was run against 51 manually collected LinkedIn posts (LinkedIn-only; no blog comparison corpus). Blog posts were excluded as a baseline — structural features vary too much across creators to produce a meaningful aggregate, and a blog baseline is not needed to validate rubric calibration against LinkedIn posts specifically. The analysis validates whether rubric checks fire at plausible rates on real LinkedIn content; checks that are systematically over- or under-sensitive can be adjusted from this data alone.

**Notes in the edit prompt:** The most direct fix for substance loss in the edit pass is injecting the original notes alongside Qwen's draft. This requires a new condition (`qwen_haiku_notes`) and a separate prompt, but would cleanly separate the question of *can the edit pass recover lost substance* from *does the edit pass preserve substance that survived Qwen*.

**Extended model set:** Gemma and Mistral are unavailable on HF free-tier serverless but remain relevant for comparison. The evaluation infrastructure is designed for easy extension — adding a new condition requires only new prompt files and a model entry in `generate.py`. See `README_full_plan.md` for the 4-model design.

**Larger sample set:** N=27 is adequate for exploratory findings but underpowered for the effect sizes this comparison may produce. A second collection round targeting domains not well-represented in the current set (particularly the simpler tercile, which constrains the complexity stratification analysis) would increase power meaningfully.

**Voice rubric generalization:** The current rubric operationalizes one author's voice. Extending to a multi-author setting would require either per-author rubric calibration or a shift to relative voice preservation (does the output sound more like the author than a generic alternative?) rather than absolute rubric compliance.

**Production monitoring:** The eval pipeline could be adapted for lightweight production monitoring — sampling real BlogAI outputs periodically and flagging substance drop or flattening rates that exceed study thresholds. This would detect model drift or prompt regression without requiring a full re-run.

## Appendix: Calibration Results

| Sample | Mode | Label | Substance | Voice |
|---|---|---|---|---|
| sample_001 | blog | gold_standard | 0.596 | 0.657 |
| sample_001 | linkedin | gold_standard | 0.540 | 0.857 |
| sample_002 | blog | gold_standard | 0.628 | 0.829 |
| sample_002 | linkedin | gold_standard | 0.653 | 0.857 |
| sample_003 | blog | gold_standard | 0.623 | 0.729 |
| sample_003 | linkedin | gold_standard | 0.567 | 0.857 |
| sample_004 | blog | gold_standard | 0.597 | 0.757 |
| sample_004 | linkedin | gold_standard | 0.696 | 0.857 |
| sample_005 | blog | gold_standard | 0.584 | 0.857 |
| sample_005 | linkedin | gold_standard | 0.630 | 0.857 |
| sample_001 | blog | null | 0.000 | 0.486 |
| sample_001 | linkedin | null | 0.000 | 0.571 |
| sample_002 | blog | null | 0.000 | 0.486 |
| sample_002 | linkedin | null | 0.000 | 0.429 |
| sample_003 | blog | null | 0.000 | 0.586 |
| sample_003 | linkedin | null | 0.000 | 0.571 |
| sample_004 | blog | null | 0.000 | 0.486 |
| sample_004 | linkedin | null | 0.000 | 0.429 |
| sample_005 | blog | null | 0.000 | 0.414 |
| sample_005 | linkedin | null | 0.000 | 0.286 |

Spearman ρ (substance): 0.926 | Spearman ρ (voice): 0.876
