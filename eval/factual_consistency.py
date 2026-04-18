"""
Factual Consistency — Hallucination Floor Check
Checks whether the output contradicts the source article.
Floor constraint only — not a primary metric.

Method: for each output sentence, retrieve top-3 most relevant article
passages via BM25, then run NLI (roberta-large-mnli) against each.
Replaces the original 1500-char truncation approach which created bias
against content later in the article.
"""

import re
from transformers import pipeline
from rank_bm25 import BM25Okapi

NLI_MODEL              = "cross-encoder/nli-deberta-v3-small"
CONTRADICTION_THRESHOLD = 0.20
TOP_K_PASSAGES          = 3
MIN_PASSAGE_WORDS       = 10

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = pipeline("text-classification", model=NLI_MODEL)
    return _pipe


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]


def _split_passages(article_text: str) -> list[str]:
    """Split article into sentence-level passages for BM25 indexing."""
    return _split_sentences(article_text)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _retrieve_passages(query: str, passages: list[str], bm25: BM25Okapi) -> list[str]:
    """Return top-K most relevant article passages for a query sentence."""
    scores  = bm25.get_scores(_tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K_PASSAGES]
    return [passages[i] for i in top_idx if len(passages[i].split()) >= MIN_PASSAGE_WORDS]


def score(article_text: str, output: str) -> dict:
    """
    Check output for factual contradictions against the article.

    Returns:
        {
            contradiction_rate: float,
            flagged: bool,
            n_sentences: int,
            contradicted: list[str],
        }
    """
    pipe             = _get_pipe()
    output_sentences = _split_sentences(output)
    article_passages = _split_passages(article_text)

    if not output_sentences or not article_passages:
        return {"contradiction_rate": 0.0, "flagged": False,
                "n_sentences": 0, "contradicted": []}

    # Build BM25 index over article passages
    tokenized_passages = [_tokenize(p) for p in article_passages]
    bm25               = BM25Okapi(tokenized_passages)

    contradicted = []
    for sentence in output_sentences:
        top_passages = _retrieve_passages(sentence, article_passages, bm25)
        if not top_passages:
            continue

        # Score sentence against each retrieved passage, take the most charitable
        # (lowest contradiction score) — we want to flag only clear contradictions
        is_contradiction = False
        for passage in top_passages:
            result = pipe(
                f"{passage} [SEP] {sentence}",
                truncation=True,
                max_length=512,
            )
            if result[0]["label"].lower() == "contradiction":
                is_contradiction = True
                break  # one clear contradiction is enough to flag the sentence

        if is_contradiction:
            contradicted.append(sentence)

    rate = len(contradicted) / len(output_sentences)
    return {
        "contradiction_rate": round(rate, 4),
        "flagged":            rate > CONTRADICTION_THRESHOLD,
        "n_sentences":        len(output_sentences),
        "contradicted":       contradicted,
    }
