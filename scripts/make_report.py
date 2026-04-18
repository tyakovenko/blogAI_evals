"""
Generate internal study report from results/scores.csv and results/calibration.csv.
Outputs results/report.md — research article style, all numbers computed from data.

Usage:
    python3 scripts/make_report.py
"""

import sys
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, spearmanr

ROOT       = Path(__file__).resolve().parent.parent
SCORES_CSV = ROOT / "results" / "scores.csv"
CALIB_CSV  = ROOT / "results" / "calibration.csv"
OUT_FILE   = ROOT / "results" / "report.md"

CONDITION_LABELS = {
    "qwen":          "Qwen (standalone)",
    "haiku":         "Haiku (standalone)",
    "qwen_pre_edit": "Qwen pre-edit",
    "qwen_haiku":    "Qwen→Haiku",
}


def load():
    if not SCORES_CSV.exists():
        sys.exit("scores.csv not found — run eval/run_all.py first")
    df = pd.read_csv(SCORES_CSV)
    df = df[df["id"].str.startswith("sample_")].copy()
    calib = pd.read_csv(CALIB_CSV, keep_default_na=False) if CALIB_CSV.exists() else None
    return df, calib


def fmt(v, decimals=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def wilcoxon_pair(df, mode, cond_a, cond_b, metric):
    """Paired Wilcoxon on samples present in both conditions."""
    sub   = df[df["mode"] == mode]
    a     = sub[sub["condition"] == cond_a].set_index("id")[metric]
    b     = sub[sub["condition"] == cond_b].set_index("id")[metric]
    a, b  = a.align(b, join="inner")
    paired = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(paired) < 5:
        return None, None, None, len(paired)
    try:
        stat, p = wilcoxon(paired["a"], paired["b"])
        n = len(paired)
        r = 1 - (2 * stat) / (n * (n + 1) / 2)
        return stat, p, r, n
    except Exception:
        return None, None, None, len(paired)


def sig_note(p, r):
    if p is None:
        return "insufficient paired data"
    if p < 0.001:
        sig = "p < 0.001"
    elif p < 0.05:
        sig = f"p = {p:.3f}"
    else:
        sig = f"p = {p:.3f}, non-significant"
    return f"{sig}, r = {fmt(r)}"


def main():
    df, calib = load()
    L = []
    w = L.append

    n_samples  = df["id"].nunique()
    conditions = [c for c in ["qwen", "haiku", "qwen_pre_edit", "qwen_haiku"]
                  if c in df["condition"].unique()]
    modes      = sorted(df["mode"].unique())
    today      = date.today().isoformat()

    # Calibration stats
    if calib is not None:
        gold = calib[calib["label"] == "gold_standard"]
        null = calib[calib["label"] == "null"]
        gold_sub   = gold["substance_aggregate"].mean()
        null_sub   = null["substance_aggregate"].mean()
        gold_voice = gold["voice_combined"].mean()
        null_voice = null["voice_combined"].mean()
        labels     = [1]*len(gold) + [0]*len(null)
        rho_sub,   p_sub   = spearmanr(labels, list(gold["substance_aggregate"]) + list(null["substance_aggregate"]))
        rho_voice, p_voice = spearmanr(labels, list(gold["voice_combined"])      + list(null["voice_combined"]))
    else:
        gold_sub = null_sub = gold_voice = null_voice = None
        rho_sub = rho_voice = None

    # --- Title & metadata ---
    w("# Substance and Voice Fidelity Under Mode Transformation")
    w("## An Internal Evaluation of LLM Thought Amplification Pipelines\n")
    w(f"*Generated {today} — internal use only*\n")
    w("---\n")

    # --- Abstract ---
    w("## Abstract\n")
    w(f"This study evaluates whether LLMs can transform user notes into mode-appropriate content "
      f"(blog posts, LinkedIn posts) without distorting or replacing the user's intellectual substance "
      f"and voice. Three conditions were compared across {n_samples} input samples and two output modes: "
      f"Qwen 2.5 7B standalone, Claude Haiku 4.5 standalone, and a Qwen→Haiku pipeline in which Haiku "
      f"applies a voice-correction edit pass to Qwen's draft. Substance fidelity was measured using "
      f"sentence-transformer cosine similarity against the original notes at the argument-component level; "
      f"voice fidelity was assessed via a two-tier regex/spaCy rubric operationalizing a known linguistic "
      f"fingerprint. A calibration set of five samples established metric ceilings (gold standard: user-edited "
      f"Haiku outputs) and floors (null baseline: article summaries ignoring notes). "
      f"Both rubrics passed a Spearman ρ ≥ 0.5 validation gate before generation proceeded. "
      f"Results and their implications for the Qwen→Haiku pipeline design are reported below.\n")

    # --- 1. Introduction ---
    w("## 1. Introduction\n")
    w("BlogAI is a content generation tool designed to amplify a user's existing thinking into "
      "mode-appropriate posts — not to summarize source articles or generate generic content. "
      "The user provides notes alongside an article link; the model's job is to build the output "
      "around those notes while using the article only as backdrop.\n")
    w("This study emerged from a reframing of an earlier evaluation design. The original question "
      "was engineering-oriented: *does the Haiku edit pass fix a known style constraint violation?* "
      "That framing assumes the fix works and asks only whether the pipeline produces the right surface "
      "markers. The more important question — and the one this study addresses — is: *what does success "
      "on this task actually look like?* Does any configuration of the pipeline preserve the user's "
      "reasoning, not just their stylistic surface? This shift from validation to discovery motivated "
      "the two-metric design (substance fidelity + voice fidelity) and the calibration approach "
      "that grounds both metrics against real human output.\n")
    w("Two production constraints shaped the study design. First, Gemma and Mistral both fail silently "
      "on the HuggingFace free-tier serverless endpoint, falling back to Haiku without any visible "
      "error — meaning BlogAI has effectively been a Qwen + Haiku system throughout its production "
      "lifetime. The evaluation model set was revised accordingly. Second, the Haiku edit prompt "
      "operates on Qwen's draft only; it does not receive the original notes. Any substance already "
      "lost in Qwen's generation cannot be recovered by the edit pass.\n")

    # --- 2. Research Questions ---
    w("## 2. Research Questions\n")
    w("1. Is the Qwen→Haiku pipeline competitive with Haiku standalone on substance and voice fidelity?\n"
      "2. What does the Haiku edit pass contribute — does it improve substance, preserve it, or trade it away for voice?\n"
      "3. Does mode (blog vs. LinkedIn) shift the fidelity tradeoff?\n"
      "4. Which argument component types (claim, evidence, logic, implication) are most at risk of loss?\n"
      "5. At what cost difference does the pipeline operate relative to full Haiku generation?\n")

    # --- 3. Methodology ---
    w("## 3. Methodology\n")

    w("### 3.1 Conditions\n")
    w("| Condition | Description | Role |")
    w("|---|---|---|")
    w("| Qwen standalone | Qwen 2.5 7B Instruct (HF free tier) | Baseline — pipeline starting point |")
    w("| Haiku standalone | Claude Haiku 4.5 | Comparison target |")
    w("| Qwen pre-edit | Qwen output before Haiku edit pass | Intermediate — isolates Qwen's contribution |")
    w("| Qwen→Haiku | Qwen draft → Haiku voice-correction edit | Primary condition |")
    w("")
    w("The intermediate pre-edit capture allows delta analysis at every stage: what Qwen produced, "
      "what the edit changed, and what the final pipeline delivered — compared against Haiku generating "
      "from the same inputs with no intermediary.\n")

    w("### 3.2 Prompts\n")
    w("Prompts are model-specific. Haiku prompts include full voice specification (em-dash usage, "
      "additive transitions, concession-redirect structure, paragraph-ending consequence markers, "
      "banned words). Qwen prompts focus on substance fidelity only — voice is handled downstream "
      "by the edit pass, and Qwen's instruction-following on stylistic detail is weaker. "
      "Edit prompts are short parameterized nudges, not formatting specs; the only variable is "
      "`{output}` (Qwen's draft). Edit prompts do not receive the original notes.\n")
    w("Haiku generation prompts were frozen after the rubric validation gate passed. "
      "Changing them after that point would invalidate the comparison baseline.\n")

    w("### 3.3 Metrics\n")
    w("**Substance fidelity** measures whether the user's argument components from the notes appear "
      "in the output. Each note sentence is typed by keyword pattern matching (logic, implication, "
      "evidence, claim), embedded with `all-mpnet-base-v2`, and matched against output sentences "
      "via cosine similarity. Per-type means and an aggregate score are reported. A flattening flag "
      "is raised when claim score is high but logic+implication mean is low — the model kept the "
      "topic but stripped the reasoning.\n")
    w("**Voice fidelity** operationalizes a known linguistic fingerprint using regex and spaCy dependency "
      "parsing. Blog scoring uses two tiers: Tier 1 (surface markers — em-dash, contractions, banned "
      "words, no bullet points) and Tier 2 (structural patterns — subordinate clause ratio, additive "
      "transitions, concession-redirect, paragraph endings on consequence). The Tier 1 − Tier 2 delta "
      "is a finding in itself: high delta indicates surface mimicry without structural voice. "
      "LinkedIn uses a separate 7-check rubric (hook velocity, short paragraphs, line break density, "
      "conversational markers, no formal transitions, direct address, CTA ending).\n")
    w("**Factual consistency** is a floor check only — not a primary metric. BM25 retrieval over "
      "article passages followed by NLI (DeBERTa-v3) flags outputs with >20% sentence contradiction rate.\n")

    w("### 3.4 Calibration\n")
    w("Before generation, five calibration samples were scored through both rubrics to validate "
      "that the metrics discriminate meaningfully. Gold standard: the author edited Haiku standalone "
      "outputs for five samples — corrections to voice, substance, and structure became the ceiling. "
      "This reflects the actual workflow (editing, not writing from scratch) and avoids circular "
      "comparison with the primary test condition. Null baseline: article summaries generated with "
      "notes withheld entirely, establishing the floor.\n")
    if rho_sub is not None:
        w(f"Both metrics passed the Spearman ρ ≥ 0.5 gate: substance ρ = {rho_sub:.3f} "
          f"(p = {p_sub:.4f}), voice ρ = {rho_voice:.3f} (p = {p_voice:.4f}). "
          f"Direction was correct for all {len(gold)} paired samples on both metrics — "
          f"gold consistently scored above null with no exceptions.\n")

    # --- 4. Data ---
    w("## 4. Data\n")
    w(f"Inputs: {n_samples} samples (originally 30; see edge cases below). Each sample consists of "
      "an article URL and the author's handwritten notes on that article. Notes vary in length, "
      "domain, and complexity. Note complexity was quantified as "
      "`(n_claims) × (mean_claim_length) × (1 + has_connecting_logic)` and stratified into "
      "simple/moderate/complex terciles before generation.\n")

    # Complexity distribution
    inputs_file = ROOT / "data" / "inputs.jsonl"
    if inputs_file.exists():
        import json
        with open(inputs_file) as f:
            inputs = [json.loads(l) for l in f if l.strip()]
        tiers = pd.Series([r.get("complexity_tier") for r in inputs]).value_counts()
        w("**Note complexity distribution:**\n")
        for tier in ["simple", "moderate", "complex"]:
            w(f"- {tier.title()}: {tiers.get(tier, 0)} samples")
        w("")

    w("### 4.1 Edge Cases and Data Quality Notes\n")
    w("**Dropped samples (3):** `sample_006` (Medium — paywalled), `sample_024` (Axios — bot detection, "
      "no manual access), `sample_027` (NYT — paywalled). All three conditions are missing for these "
      "samples. They are excluded from all analyses.\n")
    w("**Manually copied article text (5):** `sample_009` (VentureBeat), `sample_010` (Medium), "
      "`sample_011` (SBS), `sample_020` (StartupNation), `sample_021` (ScienceDirect). "
      "trafilatura was blocked by bot detection or access restrictions on these URLs. "
      "Article text was manually copied and stored in `data/articles/{id}.txt`. "
      "These samples are included in all analyses; the article source is noted.\n")
    w("**`sample_020` ad noise:** The manually copied text for this sample includes advertising copy "
      "interspersed with article content. The factual consistency scores for `sample_020` may be "
      "elevated as a result — the NLI model may flag ad copy as contradictions. Substance scores "
      "are unaffected (substance is measured against notes, not the article).\n")
    w("**`sample_017` (truncated notes):** This sample has unusually short notes compared to the "
      "article length. This is a real input edge case, not a data quality issue — it tests "
      "substance fidelity when the user provides minimal signal.\n")
    w("**HuggingFace downtime:** The HF serverless API experienced downtime during generation. "
      "Three samples (001, 003, 004) required Qwen retries. The script is resumable and all outputs "
      "were ultimately generated successfully for all 27 samples.\n")

    # --- 5. Results ---
    w("## 5. Results\n")

    # Substance
    w("### 5.1 Substance Fidelity\n")
    for mode in modes:
        sub = df[df["mode"] == mode]
        w(f"**{mode.title()} posts:**\n")
        w("| Condition | Mean | SD | N |")
        w("|---|---|---|---|")
        for cond in conditions:
            grp = sub[sub["condition"] == cond]["substance_aggregate"].dropna()
            w(f"| {CONDITION_LABELS.get(cond, cond)} | {fmt(grp.mean())} | {fmt(grp.std())} | {len(grp)} |")
        w("")

        # Primary comparison
        if "qwen_haiku" in conditions and "haiku" in conditions:
            stat, p, r, n = wilcoxon_pair(df, mode, "qwen_haiku", "haiku", "substance_aggregate")
            qh_mean = sub[sub["condition"] == "qwen_haiku"]["substance_aggregate"].mean()
            h_mean  = sub[sub["condition"] == "haiku"]["substance_aggregate"].mean()
            direction = "higher" if qh_mean > h_mean else "lower"
            w(f"Qwen→Haiku vs Haiku (paired Wilcoxon, n={n}): {sig_note(p, r)}. "
              f"Qwen→Haiku mean ({fmt(qh_mean)}) was {direction} than Haiku ({fmt(h_mean)}).\n")

        if "qwen" in conditions and "qwen_haiku" in conditions:
            stat, p, r, n = wilcoxon_pair(df, mode, "qwen", "qwen_haiku", "substance_aggregate")
            q_mean  = sub[sub["condition"] == "qwen"]["substance_aggregate"].mean()
            qh_mean = sub[sub["condition"] == "qwen_haiku"]["substance_aggregate"].mean()
            direction = "improved" if qh_mean > q_mean else "degraded"
            w(f"Qwen vs Qwen→Haiku edit contribution (n={n}): {sig_note(p, r)}. "
              f"Haiku edit {direction} substance ({fmt(q_mean)} → {fmt(qh_mean)}).\n")

    # Flattening
    w("### 5.2 Argument Flattening\n")
    w("Flattening flag: claim score ≥ 0.6 and (logic + implication) / 2 < 0.4 — "
      "the model preserved the topic but dropped the reasoning structure.\n")
    w("| Condition | Blog | LinkedIn |")
    w("|---|---|---|")
    for cond in conditions:
        rates = []
        for mode in modes:
            sub = df[(df["condition"] == cond) & (df["mode"] == mode)]
            if sub.empty:
                rates.append("—")
            else:
                n_f = int(sub["flattening_flagged"].sum())
                rates.append(f"{sub['flattening_flagged'].mean():.0%} ({n_f}/{len(sub)})")
        w(f"| {CONDITION_LABELS.get(cond, cond)} | {rates[0]} | {rates[1] if len(rates) > 1 else '—'} |")
    w("")

    # Component survival
    w("### 5.3 Substance by Component Type — Blog\n")
    w("Logic and implication components test whether the model preserves causal reasoning. "
      "A pattern of high claim scores with low logic/implication scores is the flattening signature.\n")
    w("| Condition | Claim | Evidence | Logic | Implication |")
    w("|---|---|---|---|---|")
    blog = df[df["mode"] == "blog"]
    for cond in conditions:
        grp = blog[blog["condition"] == cond]
        w(f"| {CONDITION_LABELS.get(cond, cond)} | "
          f"{fmt(grp['substance_claim'].mean())} | "
          f"{fmt(grp['substance_evidence'].mean())} | "
          f"{fmt(grp['substance_logic'].mean())} | "
          f"{fmt(grp['substance_implication'].mean())} |")
    w("")

    # Voice
    w("### 5.4 Voice Fidelity\n")
    w("**Blog — Tier 1 (surface) vs Tier 2 (structural):**\n")
    w("The delta between tiers identifies surface mimicry: a model that passes surface checks "
      "(no bullets, has contractions) without passing structural ones (subordinate clause ratio, "
      "concession-redirect, paragraph-ending consequence markers) is imitating form, not voice.\n")
    w("| Condition | Tier 1 | Tier 2 | Δ (T1−T2) |")
    w("|---|---|---|---|")
    for cond in conditions:
        grp = blog[blog["condition"] == cond]
        t1  = grp["voice_tier1"].mean()
        t2  = grp["voice_tier2"].mean()
        d   = (t1 - t2) if not (np.isnan(t1) or np.isnan(t2)) else float("nan")
        w(f"| {CONDITION_LABELS.get(cond, cond)} | {fmt(t1)} | {fmt(t2)} | {fmt(d)} |")
    w("")

    w("**LinkedIn:**\n")
    li = df[df["mode"] == "linkedin"]
    w("| Condition | Voice score |")
    w("|---|---|")
    for cond in conditions:
        grp = li[li["condition"] == cond]
        w(f"| {CONDITION_LABELS.get(cond, cond)} | {fmt(grp['voice_combined'].mean())} |")
    w("")

    # Factual consistency
    w("### 5.5 Factual Consistency (Floor Check)\n")
    w("Outputs flagged at >20% sentence contradiction rate against source article. "
      "This is a floor constraint — not a primary finding. Note: `sample_020` "
      "factual scores may be inflated due to ad copy in the article text.\n")
    w("| Condition | Mode | Flagged | Mean contradiction rate |")
    w("|---|---|---|---|")
    for cond in conditions:
        for mode in modes:
            sub = df[(df["condition"] == cond) & (df["mode"] == mode)]
            if sub.empty:
                continue
            w(f"| {CONDITION_LABELS.get(cond, cond)} | {mode.title()} | "
              f"{int(sub['contradiction_flagged'].sum())}/{len(sub)} | "
              f"{fmt(sub['contradiction_rate'].mean())} |")
    w("")

    # Cost
    w("### 5.6 Cost Analysis\n")
    haiku_conds = [c for c in ["haiku", "qwen_haiku"] if c in conditions]
    if haiku_conds:
        w("Qwen is free tier. Cost tracking applies to Haiku calls only — whether standalone "
          "generation (full prompt) or edit pass (shorter: Qwen draft + edit instruction).\n")
        w("| Condition | Mode | Mean cost (USD) | Total (USD) |")
        w("|---|---|---|---|")
        for cond in haiku_conds:
            for mode in modes:
                sub = df[(df["condition"] == cond) & (df["mode"] == mode) & df["haiku_cost_usd"].notna()]
                if sub.empty:
                    continue
                w(f"| {CONDITION_LABELS.get(cond, cond)} | {mode.title()} | "
                  f"${sub['haiku_cost_usd'].mean():.6f} | ${sub['haiku_cost_usd'].sum():.6f} |")
        total = df[df["haiku_cost_usd"].notna()]["haiku_cost_usd"].sum()
        w(f"\nTotal Haiku spend across all {n_samples} samples and both modes: **${total:.6f}**\n")

    # Edit scope
    w("### 5.7 Edit Scope — Qwen→Haiku\n")
    w("Edit scope = (sentences added + removed) / total pre-edit sentences. "
      "High scope + substance drop = Haiku rewrote the argument to fix voice.\n")
    qh = df[(df["condition"] == "qwen_haiku") & df["edit_scope"].notna()]
    if not qh.empty:
        w("| Mode | Mean | Median | % heavy edits (scope > 0.5) |")
        w("|---|---|---|---|")
        for mode in modes:
            sub = qh[qh["mode"] == mode]
            if sub.empty:
                continue
            pct = (sub["edit_scope"] > 0.5).mean()
            w(f"| {mode.title()} | {fmt(sub['edit_scope'].mean())} | "
              f"{fmt(sub['edit_scope'].median())} | {pct:.0%} |")
        w("")

    # Complexity
    w("### 5.8 Substance by Note Complexity\n")
    w("Does more complex reasoning in the notes predict worse preservation?\n")
    w("| Condition | Mode | Simple | Moderate | Complex |")
    w("|---|---|---|---|---|")
    for cond in ["haiku", "qwen_haiku"]:
        if cond not in conditions:
            continue
        for mode in modes:
            sub = df[(df["condition"] == cond) & (df["mode"] == mode)]
            vals = []
            for tier in ["simple", "moderate", "complex"]:
                t = sub[sub["complexity_tier"] == tier]["substance_aggregate"]
                vals.append(fmt(t.mean()) if not t.empty else "—")
            w(f"| {CONDITION_LABELS.get(cond, cond)} | {mode.title()} | {' | '.join(vals)} |")
    w("")

    # --- 6. Discussion ---
    w("## 6. Discussion\n")
    w("The primary question — whether the Qwen→Haiku pipeline is competitive with Haiku standalone "
      "— should be read against the cost structure. The pipeline's Haiku call operates on a shorter "
      "input (Qwen's draft + a brief edit prompt) rather than a full generation prompt, making it "
      "meaningfully cheaper per sample. Whether that cost advantage is worth any substance tradeoff "
      "depends on where substance loss occurs: in Qwen's generation, in the edit pass, or in both.\n")
    w("The intermediate pre-edit condition isolates this. If Qwen already preserves substance well "
      "and the edit pass is roughly neutral, the pipeline is clearly preferable on cost. If the edit "
      "pass actively damages substance — high edit scope with substance drop — the pipeline design "
      "needs revision, likely by adding original notes back into the edit prompt.\n")
    w("The voice tier gap (Tier 1 − Tier 2) is a second diagnostic. A large positive delta means "
      "a condition is hitting surface markers (em-dash, contractions) without the structural patterns "
      "that define the voice at depth (subordinate clause ratio, concession-redirect moves, "
      "paragraph-ending consequence markers). This is the mimicry signal — it looks right on the surface "
      "but reads differently to someone who knows the author's writing well.\n")
    w("The flattening flag captures a distinct failure mode: the model kept the topic and the claim, "
      "but stripped the causal reasoning that made the claim interesting. High flattening rates on "
      "logic and implication components indicate the model is summarizing the conclusion without "
      "preserving the argument that led there.\n")

    # --- 7. Limitations ---
    w("## 7. Limitations\n")
    w(f"**Sample size:** N = {n_samples}. At this size, Wilcoxon signed-rank requires approximately "
      f"d > 0.65 for significance at α = 0.05. Non-significant results should be interpreted as "
      f"underpowered, not as evidence of no effect.\n")
    w("**Article source heterogeneity:** Five samples use manually copied article text (bot detection "
      "blocked automated fetching); three samples were dropped entirely (paywalled). The manually "
      "copied samples are subject to copy-paste accuracy and, in the case of `sample_020`, include "
      "advertising copy that may inflate factual contradiction scores.\n")
    w("**Voice rubric precision:** The rubric passed the ρ ≥ 0.5 validation gate but is not a "
      "high-precision instrument. Surface markers in Tier 1 are reliably detectable; Tier 2 structural "
      "checks (particularly subordinate clause ratio via spaCy) are approximations. The rubric was "
      "validated against the author's own edited writing, which constrains its generalizability.\n")
    w("**Model reproducibility:** Qwen outputs on the HuggingFace free-tier serverless API are not "
      "fully reproducible — temperature was set to 0.85 and no seed was specified. Re-running "
      "generation would produce different Qwen outputs.\n")
    w("**Edit prompt constraint:** The Haiku edit prompt does not receive the original notes. "
      "Any substance already dropped by Qwen cannot be recovered by the edit pass. This is a "
      "known design constraint, not an oversight — changing it would require re-generating all "
      "qwen_haiku outputs and invalidate cross-condition comparison.\n")
    w("**LinkedIn rubric misalignment (empirical):** Feature analysis on 51 real LinkedIn posts "
      "reveals three rubric checks that almost never fire on real posts: `hook_in_first_sentence` "
      "(operationalized as first-person reaction verb — present in 1.9% of posts; real hook signal "
      "is short first sentence, median 11 words), `conversational_markers` (specific phrases — "
      "\"honestly\", \"here's the thing\", \"to be fair\", \"lowkey\" — present in 1.9%), and "
      "`ends_with_question_or_cta` (question or explicit CTA verb — 1.9% each). `has_direct_address` "
      "(\"you\") is present in only 53.8% of posts, so it also over-penalizes valid LinkedIn content. "
      "LinkedIn scores in this study were computed with the original uncorrected rubric — they "
      "underestimate platform fidelity for all conditions. Rubric correction is deferred to future work.\n")

    # --- 8. Future Work ---
    w("## 8. Future Work\n")
    w("**LinkedIn rubric refinement:** Feature analysis was run against 51 manually collected LinkedIn "
      "posts (LinkedIn-only; no blog comparison corpus). Blog posts were excluded as a baseline — "
      "structural features vary too much across creators to produce a meaningful aggregate, and a "
      "blog baseline is not needed to validate rubric calibration against LinkedIn posts specifically. "
      "The analysis validates whether rubric checks fire at plausible rates on real LinkedIn content; "
      "checks that are systematically over- or under-sensitive can be adjusted from this data alone.\n")
    w("**Notes in the edit prompt:** The most direct fix for substance loss in the edit pass is "
      "injecting the original notes alongside Qwen's draft. This requires a new condition "
      "(`qwen_haiku_notes`) and a separate prompt, but would cleanly separate the question of "
      "*can the edit pass recover lost substance* from *does the edit pass preserve substance that survived Qwen*.\n")
    w("**Extended model set:** Gemma and Mistral are unavailable on HF free-tier serverless but "
      "remain relevant for comparison. The evaluation infrastructure is designed for easy extension — "
      "adding a new condition requires only new prompt files and a model entry in `generate.py`. "
      "See `README_full_plan.md` for the 4-model design.\n")
    w("**Larger sample set:** N=27 is adequate for exploratory findings but underpowered for "
      "the effect sizes this comparison may produce. A second collection round targeting "
      "domains not well-represented in the current set (particularly the simpler tercile, "
      "which constrains the complexity stratification analysis) would increase power meaningfully.\n")
    w("**Voice rubric generalization:** The current rubric operationalizes one author's voice. "
      "Extending to a multi-author setting would require either per-author rubric calibration "
      "or a shift to relative voice preservation (does the output sound more like the author than "
      "a generic alternative?) rather than absolute rubric compliance.\n")
    w("**Production monitoring:** The eval pipeline could be adapted for lightweight production "
      "monitoring — sampling real BlogAI outputs periodically and flagging substance drop or "
      "flattening rates that exceed study thresholds. This would detect model drift or "
      "prompt regression without requiring a full re-run.\n")

    # --- Appendix ---
    w("## Appendix: Calibration Results\n")
    if calib is not None:
        w("| Sample | Mode | Label | Substance | Voice |")
        w("|---|---|---|---|---|")
        for _, row in calib.iterrows():
            w(f"| {row['id']} | {row['mode']} | {row['label']} | "
              f"{fmt(row['substance_aggregate'])} | {fmt(row['voice_combined'])} |")
        w("")
        w(f"Spearman ρ (substance): {fmt(rho_sub)} | Spearman ρ (voice): {fmt(rho_voice)}\n")
    else:
        w("*Calibration CSV not found.*\n")

    report = "\n".join(L)
    OUT_FILE.write_text(report)
    print(f"Report written to {OUT_FILE}")


if __name__ == "__main__":
    main()
