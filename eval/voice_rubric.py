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
    r"\bnexus\b", r"\bthrilled\b", r"\bpassionate\b",
]

# Sequential transition markers (she avoids these)
SEQUENTIAL_TRANSITIONS = [
    r"\bfirstly?\b", r"\bsecondly?\b", r"\bthirdly?\b", r"\bfinally\b",
    r"\bin conclusion\b", r"\bto summarize\b", r"\bin summary\b",
]

# Her signature connectors
ADDITIVE_TRANSITIONS = [
    r"\bmoreover\b", r"\bhowever\b", r"\bsimilarly\b", r"\bin addition\b",
    r"\bfurthermore\b",
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


def _ends_with_open_question(text: str) -> bool:
    # Last non-empty line ends with a question mark
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return False
    return lines[-1].endswith("?")


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


def _has_hedging_in_body(text: str) -> bool:
    hedges = [r"\bi think\b", r"\bi believe\b", r"\bit seems\b",
              r"\bmight\b", r"\bperhaps\b", r"\brelatively\b",
              r"\bsomewhat\b", r"\bin a sense\b"]
    return _has_pattern(text, hedges)


def _has_more_specifically(text: str) -> bool:
    return bool(re.search(r"\bmore specifically\b", text, re.IGNORECASE))


def _no_banned_words(text: str) -> bool:
    return not _has_pattern(text, BANNED_WORDS)


def _no_sequential_transitions(text: str) -> bool:
    return not _has_pattern(text, SEQUENTIAL_TRANSITIONS)


def _paragraphs_end_on_consequence(text: str) -> bool:
    """
    Heuristic: last sentence of each paragraph should end on a consequence
    (contains 'thus', 'therefore', 'which means', 'this suggests', 'this means',
    em-dash, or 'which').
    Score as ratio of paragraphs passing.
    """
    consequence_markers = [
        r"\bthus\b", r"\btherefore\b", r"\bwhich means\b",
        r"\bthis suggests\b", r"\bthis means\b", r"\bwhich\b", r"—"
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


# --- Mode rubrics ---

def score_blog(text: str) -> dict:
    """
    Score output against blog/social register rubric.
    Returns score [0,1] and per-item breakdown.
    """
    checks = {
        "no_bullet_points": not _has_bullet_points(text),
        "no_markdown_headers": not _has_markdown_headers(text),
        "has_contractions": _has_contractions(text),
        "has_direct_address": _has_direct_address(text),
        "ends_with_open_question": _ends_with_open_question(text),
        "no_banned_words": _no_banned_words(text),
        "no_sequential_transitions": _no_sequential_transitions(text),
    }
    score = sum(checks.values()) / len(checks)
    return {"score": round(score, 4), "checks": checks, "mode": "blog"}


def score_analytical(text: str) -> dict:
    """
    Score output against analytical blog post register rubric.
    Returns score [0,1] and per-item breakdown.
    """
    checks = {
        "no_bullet_points": not _has_bullet_points(text),
        "no_markdown_headers": not _has_markdown_headers(text),
        "has_em_dash_restatement": _has_em_dash_restatement(text),
        "has_more_specifically": _has_more_specifically(text),
        "thesis_not_in_sentence_one": _thesis_not_in_sentence_one(text),
        "has_hedging_in_body": _has_hedging_in_body(text),
        "no_sequential_transitions": _no_sequential_transitions(text),
        "no_banned_words": _no_banned_words(text),
        "paragraphs_end_on_consequence": _paragraphs_end_on_consequence(text),
    }
    score = sum(checks.values()) / len(checks)
    return {"score": round(score, 4), "checks": checks, "mode": "analytical"}


def score(text: str, mode: str) -> dict:
    """Route to correct rubric by mode."""
    if mode == "linkedin":
        return score_blog(text)
    elif mode in ("blog", "analytical"):
        return score_analytical(text)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'blog', 'analytical', or 'linkedin'.")
