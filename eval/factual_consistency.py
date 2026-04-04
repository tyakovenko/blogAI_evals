"""
Factual Consistency — Hallucination Floor Check
Checks whether the output contradicts the source article.
This is a pass/fail floor constraint, not a primary metric.

Method: NLI (roberta-large-mnli) — for each output sentence, check
entailment against the article. Flag outputs with >20% contradiction rate.
"""

from transformers import pipeline
import re

NLI_MODEL = "roberta-large-mnli"
CONTRADICTION_THRESHOLD = 0.20

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = pipeline("text-classification", model=NLI_MODEL)
    return _pipe


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]


def _truncate(text: str, max_chars: int = 400) -> str:
    """Truncate to fit NLI model token limits."""
    return text[:max_chars]


def score(article_text: str, output: str) -> dict:
    """
    Check output for factual contradictions against the article.

    Returns:
        {
            "contradiction_rate": float,  # fraction of sentences flagged
            "flagged": bool,              # True if rate > threshold
            "n_sentences": int,
            "contradicted": list[str]     # sentences flagged for review
        }
    """
    pipe = _get_pipe()
    output_sentences = _split_sentences(output)

    if not output_sentences:
        return {"contradiction_rate": 0.0, "flagged": False,
                "n_sentences": 0, "contradicted": []}

    # Use first 1500 chars of article as premise (NLI token limit)
    article_premise = _truncate(article_text, 1500)
    contradicted = []

    for sentence in output_sentences:
        # NLI input: premise = article snippet, hypothesis = output sentence
        result = pipe(
            f"{article_premise} [SEP] {_truncate(sentence, 200)}",
            truncation=True,
            max_length=512,
        )
        label = result[0]["label"].lower()
        if label == "contradiction":
            contradicted.append(sentence)

    rate = len(contradicted) / len(output_sentences)

    return {
        "contradiction_rate": round(rate, 4),
        "flagged": rate > CONTRADICTION_THRESHOLD,
        "n_sentences": len(output_sentences),
        "contradicted": contradicted,
    }
