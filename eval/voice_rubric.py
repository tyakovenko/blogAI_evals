"""
Voice Fidelity Rubric Scorer
Operationalizes the taya-voice linguistic fingerprint as a binary rubric.
Scored separately per mode — blog and analytical registers have different expected features.

Score = items passing / total items for the mode's rubric.
No LLM-as-judge — all rules are regex or spaCy detectable.
"""

import re
import spacy

_nlp = None

# Words/phrases she never uses in analytical prose
BANNED_WORDS = [
    r"\bclearly\b", r"\bobviously\b", r"\bparadigm\b", r"\bdiscourse\b",
    r"\bnexus\b", r"\bthrilled\b", r"\bpassionate\b", r"\bgenuinely\b",
]

# Sequential transition markers (she avoids these)
SEQUENTIAL_TRANSITIONS = [
    r"\bfirstly?\b", r"\bsecondly?\b", r"\bthirdly?\b", r"\bfinally\b",
    r"\bin conclusion\b", r"\bto summarize\b", r"\bin summary\b",
]

# Her signature additive (not sequential) connectors — Tier 2 structural check
ADDITIVE_TRANSITIONS = [
    r"\bmoreover\b", r"\bwhat'?s more\b", r"\band yet\b",
    r"\beven so\b", r"\bsimilarly\b",
]


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _has_pattern(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def _has_em_dash_restatement(text: str) -> bool:
    # Em-dash followed by a clause that restates or specifies the previous
    return bool(re.search(r"—\s*\w+", text))


def _has_bullet_points(text: str) -> bool:
    return bool(re.search(r"^\s*[-*•]\s+", text, re.MULTILINE))


def _has_markdown_headers(text: str) -> bool:
    return bool(re.search(r"^#{1,4}\s+", text, re.MULTILINE))


def _has_contractions(text: str) -> bool:
    return bool(re.search(
        r"\b(it's|i'd|that's|you're|i'm|we're|they're|isn't|aren't|don't|can't|won't)\b",
        text, re.IGNORECASE
    ))


def _has_direct_address(text: str) -> bool:
    return bool(re.search(r"\byou\b", text, re.IGNORECASE))


def _thesis_not_in_sentence_one(text: str) -> bool:
    # First sentence should be orienting context, not the central claim
    # Heuristic: first sentence should not contain strong claim markers
    nlp = _get_nlp()
    doc = nlp(text[:500])
    sentences = list(doc.sents)
    if not sentences:
        return True
    first = sentences[0].text.lower()
    # Strong claim markers in sentence 1 = thesis-first (bad)
    claim_markers = [r"\bthis (shows|demonstrates|proves|means)\b",
                     r"\bthe (key|main|central|core) (point|argument|claim)\b",
                     r"\bultimately\b", r"\bin (this|my) (essay|post|piece)\b"]
    return not _has_pattern(first, claim_markers)


def _has_more_specifically(text: str) -> bool:
    return bool(re.search(r"\bmore specifically\b", text, re.IGNORECASE))


def _no_banned_words(text: str) -> bool:
    return not _has_pattern(text, BANNED_WORDS)


def _no_sequential_transitions(text: str) -> bool:
    return not _has_pattern(text, SEQUENTIAL_TRANSITIONS)


def _has_additive_transitions(text: str) -> bool:
    """Check if at least one additive (non-sequential) transition is present."""
    return _has_pattern(text, ADDITIVE_TRANSITIONS)


def _has_concession_redirect(text: str) -> bool:
    """Check for concession-redirect move: [X]. But/That said/The problem is [Y]."""
    patterns = [
        r"[.!?]\s+But\s+\w",
        r"[.!?]\s+That said,",
        r"[.!?]\s+The problem is",
        r"[.!?]\s+And yet",
    ]
    return any(re.search(p, text) for p in patterns)


def _subordinate_clause_ratio(text: str) -> bool:
    """Check if subordinate clause ratio is above threshold using spaCy dependency parse."""
    nlp = _get_nlp()
    doc = nlp(text[:1000])  # limit for performance
    sentences = list(doc.sents)
    if not sentences:
        return False
    subordinate_deps = {"advcl", "relcl", "acl"}
    subordinate_count = sum(
        1 for token in doc if token.dep_ in subordinate_deps
    )
    # Threshold: at least 0.3 subordinate clauses per sentence
    return (subordinate_count / len(sentences)) >= 0.3


def _paragraphs_end_on_consequence(text: str) -> bool:
    """
    Last sentence of each paragraph contains an explicit consequence marker.
    Uses specific markers only — avoids \bwhich\b which fires on ordinary relative clauses.
    Score as ratio of paragraphs passing (>= 0.5 to pass).
    """
    consequence_markers = [
        r"\bthis means\b", r"\bwhich means\b", r"\bthe implication\b",
        r"\bwhich is why\b", r"\bthat'?s the point\b", r"\bthis is why\b",
        r"\bmeaning that\b", r"—"
    ]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return False
    passing = 0
    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para)
        if not sentences:
            continue
        last = sentences[-1].lower()
        if _has_pattern(last, consequence_markers):
            passing += 1
    return (passing / len(paragraphs)) >= 0.5


def _hook_in_first_sentence(text: str) -> bool:
    """
    Check if first sentence contains a reaction/opinion/claim (not context setup).
    First sentence <30 words AND contains first-person marker (I, we) or direct claim verb.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return False
    first = sentences[0].lower()
    word_count = len(first.split())
    if word_count > 30:
        return False
    # Check for reaction markers: I/we + verb (found, learned, realized, noticed, tried)
    reaction_verbs = r"\b(i|we)\s+(found|learned|realized|noticed|tried|discovered|saw|felt|think|believe|know)\b"
    return bool(re.search(reaction_verbs, first))


def _short_paragraphs(text: str) -> bool:
    """
    Check if paragraphs are short (no paragraph exceeds 3 sentences).
    Paragraph = block split on \n\n or single line if < 40 chars (LinkedIn formatting).
    """
    # Split on double newline or treat short lines as separate units
    paragraphs = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        # Further split on single newline if they form short segments (< 40 chars)
        lines = para.split("\n")
        if all(len(line) < 40 for line in lines if line.strip()):
            paragraphs.extend([l for l in lines if l.strip()])
        else:
            paragraphs.append(para)

    if not paragraphs:
        return False

    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        if len(sentences) > 3:
            return False
    return True


def _high_line_break_density(text: str) -> bool:
    """
    Check if line break density is high: >= 1 blank line per 80 words (LinkedIn visual rhythm).
    """
    word_count = len(text.split())
    blank_lines = len([l for l in text.split("\n") if not l.strip()])
    if word_count == 0:
        return False
    blank_per_80_words = (blank_lines / word_count) * 80
    return blank_per_80_words >= 1.0


def _conversational_markers_present(text: str) -> bool:
    """
    Check if at least one conversational marker is present.
    Markers: "honestly", "here's the thing", "to be fair", "lowkey", "this is why".
    """
    markers = [
        r"\bhonestly\b",
        r"\b(here'?s? the thing|here goes)\b",
        r"\bto be fair\b",
        r"\blowkey\b",
        r"\bthis is why\b",
    ]
    return _has_pattern(text, markers)


def _no_formal_transitions(text: str) -> bool:
    """
    Check that formal transitions are absent.
    Formal: "moreover", "furthermore", "in addition", "in conclusion", "thus", "therefore".
    """
    formal = [
        r"\bmoreover\b", r"\bfurthermore\b", r"\bin addition\b",
        r"\bin conclusion\b", r"\bthus\b", r"\btherefore\b",
    ]
    return not _has_pattern(text, formal)


def _ends_with_question_or_cta(text: str) -> bool:
    """
    Check if ends with question mark OR LinkedIn CTA verb.
    CTAs: "share", "comment", "let me know", "I'd love to hear", "what do you think".
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    last = lines[-1]
    if last.endswith("?"):
        return True
    cta_markers = [
        r"\bshare\b", r"\bcomment\b", r"\blet me know\b",
        r"\bi'?d love to hear\b", r"\bwhat do you think\b",
    ]
    return _has_pattern(last, cta_markers)


# --- Mode rubrics ---

def score_linkedin(text: str) -> dict:
    """
    Score output against LinkedIn platform-specific register rubric.
    LinkedIn prioritizes: quick hook, short paragraphs, visual breaks, conversational tone,
    direct audience address, and CTA/question endings.
    Returns score [0,1] and per-item breakdown.
    """
    checks = {
        "hook_in_first_sentence": _hook_in_first_sentence(text),
        "short_paragraphs": _short_paragraphs(text),
        "high_line_break_density": _high_line_break_density(text),
        "conversational_markers": _conversational_markers_present(text),
        "no_formal_transitions": _no_formal_transitions(text),
        "has_direct_address": _has_direct_address(text),
        "ends_with_question_or_cta": _ends_with_question_or_cta(text),
    }
    score = sum(checks.values()) / len(checks)
    return {"score": round(score, 4), "checks": checks, "mode": "linkedin"}


def score_analytical(text: str) -> dict:
    """
    Score blog output in two tiers.

    Tier 1 — surface markers (easy to mimic).
    Tier 2 — structural patterns (hard to mimic).

    Returns tier1, tier2, combined, delta, and per-item breakdown.
    """
    tier1_checks = {
        "no_bullet_points":        not _has_bullet_points(text),
        "no_markdown_headers":     not _has_markdown_headers(text),
        "has_em_dash_restatement": _has_em_dash_restatement(text),
        "has_more_specifically":   _has_more_specifically(text),
        "has_contractions":        _has_contractions(text),
        "no_banned_words":         _no_banned_words(text),
        "no_sequential_transitions": _no_sequential_transitions(text),
    }
    tier2_checks = {
        "subordinate_clause_ratio":      _subordinate_clause_ratio(text),
        "has_additive_transitions":      _has_additive_transitions(text),
        "has_concession_redirect":       _has_concession_redirect(text),
        "paragraphs_end_on_consequence": _paragraphs_end_on_consequence(text),
        "thesis_not_in_sentence_one":    _thesis_not_in_sentence_one(text),
    }
    tier1 = round(sum(tier1_checks.values()) / len(tier1_checks), 4)
    tier2 = round(sum(tier2_checks.values()) / len(tier2_checks), 4)
    return {
        "tier1":    tier1,
        "tier2":    tier2,
        "combined": round((tier1 + tier2) / 2, 4),
        "delta":    round(tier1 - tier2, 4),
        "tier1_checks": tier1_checks,
        "tier2_checks": tier2_checks,
        "mode":     "blog",
    }


def score(text: str, mode: str) -> dict:
    """Route to correct rubric by mode."""
    if mode == "linkedin":
        return score_linkedin(text)
    elif mode == "blog":
        return score_analytical(text)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'blog' or 'linkedin'.")
