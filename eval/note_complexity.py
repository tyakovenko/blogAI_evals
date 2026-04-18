"""
Note Complexity Scorer
Computes a complexity score per input and writes it back to inputs.jsonl in-place.

complexity = (number of claims) × (mean claim length in words) × (1 + has_connecting_logic)

has_connecting_logic = 1 if any sentence matches logic keywords, else 0.
Tercile thresholds are computed from the full input set so stratification is relative.

Usage:
    python eval/note_complexity.py          # reads/writes data/inputs.jsonl
"""

import json
import re
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
INPUTS_FILE = ROOT / "data" / "inputs.jsonl"

LOGIC_KEYWORDS = [
    "because", "which means", "this is why", "as a result",
    "therefore", "that's why", "so ", "thus",
]

# Same keywords used in substance_fidelity for consistency
LOGIC_RE = re.compile(
    r"\b(" + "|".join(re.escape(k.strip()) for k in LOGIC_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def split_claims(notes: str) -> list[str]:
    """Split notes into individual claim sentences."""
    lines = [l.strip() for l in notes.split("\n") if l.strip()]
    claims = []
    for line in lines:
        sentences = re.split(r"(?<=[.!?])\s+", line)
        claims.extend([s.strip() for s in sentences if len(s.strip()) > 10])
    return claims


def score_complexity(notes: str) -> float:
    claims = split_claims(notes)
    if not claims:
        return 0.0
    n_claims         = len(claims)
    mean_len         = sum(len(c.split()) for c in claims) / n_claims
    has_logic        = int(bool(LOGIC_RE.search(notes)))
    return round(n_claims * mean_len * (1 + has_logic), 4)


def assign_tiers(scores: list[float]) -> list[str]:
    """Assign simple/moderate/complex based on tercile thresholds."""
    if len(scores) < 3:
        # Not enough data for meaningful terciles — assign moderate to all
        return ["moderate"] * len(scores)
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    t1 = sorted_scores[n // 3]
    t2 = sorted_scores[(2 * n) // 3]
    tiers = []
    for s in scores:
        if s <= t1:
            tiers.append("simple")
        elif s <= t2:
            tiers.append("moderate")
        else:
            tiers.append("complex")
    return tiers


def main():
    if not INPUTS_FILE.exists():
        raise FileNotFoundError(f"inputs.jsonl not found at {INPUTS_FILE}")

    records = []
    with open(INPUTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    scores = [score_complexity(r["notes"]) for r in records]
    tiers  = assign_tiers(scores)

    for record, score, tier in zip(records, scores, tiers):
        record["note_complexity"] = score
        record["complexity_tier"] = tier

    with open(INPUTS_FILE, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Complexity scores written to {INPUTS_FILE}")
    for r in records:
        print(f"  {r['id']}: {r['note_complexity']} ({r['complexity_tier']})")


if __name__ == "__main__":
    main()
