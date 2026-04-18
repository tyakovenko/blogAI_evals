"""
Substance Fidelity Scorer — Argument-Level
Measures whether the user's argument components from notes appear in the output.

Components are typed by keyword pattern matching:
  logic      — causal/connective language
  implication — consequence markers
  evidence   — example markers
  claim      — default (no keyword match)

Each component is embedded and matched against output sentences.
Reports per-type scores, aggregate, and two flags:
  substance_flagged  — aggregate < SUBSTANCE_THRESHOLD
  flattening_flagged — claim score high but logic+implication low
                       (model kept topic, stripped reasoning)
"""

import re
from sentence_transformers import SentenceTransformer, util

MODEL_NAME           = "all-mpnet-base-v2"
SUBSTANCE_THRESHOLD  = 0.5   # below this → substance likely replaced
FLATTENING_CLAIM_MIN = 0.6   # claim score must exceed this
FLATTENING_LOGIC_MAX = 0.4   # and (logic+impl)/2 must be below this

_model = None

# --- Keyword patterns for component typing ---

LOGIC_PATTERNS = re.compile(
    r"\b(because|which means|this is why|as a result|therefore|that's why|thus)\b",
    re.IGNORECASE,
)
IMPLICATION_PATTERNS = re.compile(
    r"\b(the implication|this means|which is why|so the|meaning that|what this means)\b",
    re.IGNORECASE,
)
EVIDENCE_PATTERNS = re.compile(
    r"\b(for example|for instance|such as|like when|consider|take [a-z])",
    re.IGNORECASE,
)

# Claim type tags (separate from component type)
OPINION_PATTERNS = re.compile(
    r"\b(i think|i believe|it seems|arguably|might be|i'd say|seems like)\b",
    re.IGNORECASE,
)
CONNECTION_PATTERNS = LOGIC_PATTERNS   # reuse logic patterns


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _split_sentences(text: str) -> list[str]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sentences = []
    for line in lines:
        parts = re.split(r"(?<=[.!?])\s+", line)
        sentences.extend([p.strip() for p in parts if len(p.strip()) > 10])
    return sentences


def _type_component(sentence: str) -> str:
    """Assign a component type based on keyword presence. Order matters — most specific first."""
    if IMPLICATION_PATTERNS.search(sentence):
        return "implication"
    if LOGIC_PATTERNS.search(sentence):
        return "logic"
    if EVIDENCE_PATTERNS.search(sentence):
        return "evidence"
    return "claim"


def _tag_claim_type(sentence: str) -> str:
    """Tag the semantic claim type (opinion/fact/connection/question)."""
    if sentence.strip().endswith("?"):
        return "question"
    if CONNECTION_PATTERNS.search(sentence):
        return "connection"
    if OPINION_PATTERNS.search(sentence):
        return "opinion"
    return "fact"


def score(notes: str, output: str) -> dict:
    """
    Score substance fidelity between user notes and model output.

    Returns:
        {
            aggregate: float,
            by_component: {claim, evidence, logic, implication: float},
            by_claimtype: {opinion, fact, connection, question: float | None},
            per_sentence: list of {text, component_type, claim_type, max_sim},
            n_components: int,
            substance_flagged: bool,
            flattening_flagged: bool,
        }
    """
    model = _get_model()

    note_sentences  = _split_sentences(notes)
    output_sentences = _split_sentences(output)

    if not note_sentences or not output_sentences:
        return _empty_result()

    # Type each note sentence
    typed = [
        {
            "text":           s,
            "component_type": _type_component(s),
            "claim_type":     _tag_claim_type(s),
        }
        for s in note_sentences
    ]

    # Embed and match
    note_embeddings   = model.encode([t["text"] for t in typed], convert_to_tensor=True)
    output_embeddings = model.encode(output_sentences, convert_to_tensor=True)
    sim_matrix        = util.cos_sim(note_embeddings, output_embeddings)

    for i, t in enumerate(typed):
        t["max_sim"] = round(float(sim_matrix[i].max()), 4)

    # Aggregate score
    all_sims  = [t["max_sim"] for t in typed]
    aggregate = round(sum(all_sims) / len(all_sims), 4)

    # Per-component-type means
    by_component = {}
    for ctype in ("claim", "evidence", "logic", "implication"):
        group = [t["max_sim"] for t in typed if t["component_type"] == ctype]
        by_component[ctype] = round(sum(group) / len(group), 4) if group else None

    # Per-claim-type means
    by_claimtype = {}
    for ctype in ("opinion", "fact", "connection", "question"):
        group = [t["max_sim"] for t in typed if t["claim_type"] == ctype]
        by_claimtype[ctype] = round(sum(group) / len(group), 4) if group else None

    # Flattening: high claim preservation + low logic/implication
    claim_score  = by_component.get("claim")  if by_component.get("claim")  is not None else 0.0
    logic_score  = by_component.get("logic")  if by_component.get("logic")  is not None else 0.0
    impl_score   = by_component.get("implication") if by_component.get("implication") is not None else 0.0
    logic_impl_mean = (logic_score + impl_score) / 2

    flattening = (
        claim_score  >= FLATTENING_CLAIM_MIN and
        logic_impl_mean < FLATTENING_LOGIC_MAX and
        # Only flag if we actually have logic/implication components to measure
        any(t["component_type"] in ("logic", "implication") for t in typed)
    )

    return {
        "aggregate":         aggregate,
        "by_component":      by_component,
        "by_claimtype":      by_claimtype,
        "per_sentence":      typed,
        "n_components":      len(typed),
        "substance_flagged": aggregate < SUBSTANCE_THRESHOLD,
        "flattening_flagged": flattening,
    }


def _empty_result() -> dict:
    return {
        "aggregate":          0.0,
        "by_component":       {k: None for k in ("claim", "evidence", "logic", "implication")},
        "by_claimtype":       {k: None for k in ("opinion", "fact", "connection", "question")},
        "per_sentence":       [],
        "n_components":       0,
        "substance_flagged":  True,
        "flattening_flagged": False,
    }
