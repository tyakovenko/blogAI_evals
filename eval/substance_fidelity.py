"""
Substance Fidelity Scorer
Measures whether the user's ideas from notes appear in the output,
or were replaced by article summary.

Score = mean of per-claim max cosine similarities between notes and output.
Low score (<0.5) = model replaced user's ideas with article content.
"""

from sentence_transformers import SentenceTransformer, util
import re

MODEL_NAME = "all-mpnet-base-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _split_claims(text: str) -> list[str]:
    """Split notes or output into individual claims."""
    # Split on newlines first (for bullet-style notes), then sentence boundaries
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    claims = []
    for line in lines:
        # Further split long lines on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", line)
        claims.extend([s.strip() for s in sentences if len(s.strip()) > 10])
    return claims


def score(notes: str, output: str) -> dict:
    """
    Compute substance fidelity between user notes and model output.

    Returns:
        {
            "score": float,           # mean of per-claim max similarities [0, 1]
            "per_claim": list[float], # max similarity per note claim
            "flagged": bool,          # True if score < 0.5 (substance likely replaced)
            "n_claims": int
        }
    """
    model = _get_model()

    note_claims = _split_claims(notes)
    output_sentences = _split_claims(output)

    if not note_claims or not output_sentences:
        return {"score": 0.0, "per_claim": [], "flagged": True, "n_claims": 0}

    note_embeddings = model.encode(note_claims, convert_to_tensor=True)
    output_embeddings = model.encode(output_sentences, convert_to_tensor=True)

    # For each note claim, find max similarity with any output sentence
    similarity_matrix = util.cos_sim(note_embeddings, output_embeddings)
    per_claim_max = similarity_matrix.max(dim=1).values.tolist()

    mean_score = sum(per_claim_max) / len(per_claim_max)

    return {
        "score": round(mean_score, 4),
        "per_claim": [round(s, 4) for s in per_claim_max],
        "flagged": mean_score < 0.5,
        "n_claims": len(note_claims),
    }
