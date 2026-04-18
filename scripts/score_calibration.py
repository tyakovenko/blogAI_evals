"""
Rubric validation gate — scores calibration set and checks Spearman ρ ≥ 0.5.

Scores:
  - data/calibration/gold_standard.jsonl  (manual ceiling — Taya's edited posts)
  - data/calibration/null_baseline.jsonl  (automated floor — article summary only)

For each sample+mode pair, gold should score higher than null.
Spearman ρ is computed between rubric scores and manual quality labels (gold=1, null=0).

Writes results/calibration.csv and reports pass/fail per metric.

Usage:
    python3 scripts/score_calibration.py
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from substance_fidelity import score as substance_score
from voice_rubric        import score as voice_score

GOLD_FILE = ROOT / "data" / "calibration" / "gold_standard.jsonl"
NULL_FILE = ROOT / "data" / "calibration" / "null_baseline.jsonl"
OUT_FILE  = ROOT / "results" / "calibration.csv"

RHO_THRESHOLD = 0.5


def load_calibration(path: Path) -> list[dict]:
    """Load multi-line JSON records using raw_decode to handle records spanning multiple lines."""
    with open(path) as f:
        content = f.read().strip()
    decoder = json.JSONDecoder()
    records = []
    idx = 0
    while idx < len(content):
        # Skip whitespace between records
        while idx < len(content) and content[idx] in " \t\n\r":
            idx += 1
        if idx >= len(content):
            break
        obj, end = decoder.raw_decode(content, idx)
        records.append(obj)
        idx = end
    return records


def score_record(rec: dict) -> dict:
    notes = rec.get("notes", "")
    output = rec.get("output", "")
    mode = rec["mode"]

    s = substance_score(notes, output)
    v = voice_score(output, mode)

    if mode == "linkedin":
        voice_combined = v.get("score")
        voice_tier1 = v.get("score")
        voice_tier2 = None
    else:
        voice_combined = v.get("combined")
        voice_tier1 = v.get("tier1")
        voice_tier2 = v.get("tier2")

    return {
        "id":                    rec["id"],
        "mode":                  mode,
        "label":                 rec.get("baseline_type", "unknown"),
        "substance_aggregate":   s["aggregate"],
        "substance_flagged":     int(s["substance_flagged"]),
        "flattening_flagged":    int(s["flattening_flagged"]),
        "voice_tier1":           voice_tier1,
        "voice_tier2":           voice_tier2,
        "voice_combined":        voice_combined,
    }


def main():
    ROOT.joinpath("results").mkdir(exist_ok=True)

    gold_records = load_calibration(GOLD_FILE)
    null_records = load_calibration(NULL_FILE)
    print(f"Gold standard: {len(gold_records)} records")
    print(f"Null baseline: {len(null_records)} records")

    print("\nScoring gold standard...")
    gold_scored = []
    for rec in gold_records:
        print(f"  {rec['id']}/{rec['mode']}...", end=" ", flush=True)
        scored = score_record(rec)
        gold_scored.append(scored)
        print(f"substance={scored['substance_aggregate']:.3f}  voice={scored['voice_combined']:.3f}")

    print("\nScoring null baseline...")
    null_scored = []
    for rec in null_records:
        print(f"  {rec['id']}/{rec['mode']}...", end=" ", flush=True)
        scored = score_record(rec)
        null_scored.append(scored)
        print(f"substance={scored['substance_aggregate']:.3f}  voice={scored['voice_combined']:.3f}")

    # --- Spearman ρ validation ---
    # Build paired arrays: for each (id, mode) pair, gold=1, null=0
    gold_by_key = {(r["id"], r["mode"]): r for r in gold_scored}
    null_by_key = {(r["id"], r["mode"]): r for r in null_scored}

    pairs = sorted(set(gold_by_key) & set(null_by_key))
    if not pairs:
        sys.exit("No overlapping (id, mode) pairs between gold and null sets.")

    manual_labels = []
    substance_scores = []
    voice_scores_list = []

    for key in pairs:
        g = gold_by_key[key]
        n = null_by_key[key]
        # gold = 1 (better), null = 0 (worse)
        manual_labels.extend([1, 0])
        substance_scores.extend([g["substance_aggregate"], n["substance_aggregate"]])
        voice_scores_list.extend([g["voice_combined"], n["voice_combined"]])

    rho_substance, p_substance = spearmanr(manual_labels, substance_scores)
    rho_voice,     p_voice     = spearmanr(manual_labels, voice_scores_list)

    print("\n--- Rubric Validation Gate ---")
    print(f"Substance Spearman ρ = {rho_substance:.3f}  (p={p_substance:.4f})  {'✓ PASS' if rho_substance >= RHO_THRESHOLD else '✗ FAIL'}")
    print(f"Voice     Spearman ρ = {rho_voice:.3f}  (p={p_voice:.4f})  {'✓ PASS' if rho_voice >= RHO_THRESHOLD else '✗ FAIL'}")

    passed = rho_substance >= RHO_THRESHOLD and rho_voice >= RHO_THRESHOLD
    print(f"\nGate: {'✓ PASS — proceed to full generation' if passed else '✗ FAIL — rubric must be revised before full study'}")

    # --- Pairwise direction check ---
    print("\n--- Pairwise direction (gold should score > null) ---")
    substance_correct = 0
    voice_correct = 0
    for key in pairs:
        g = gold_by_key[key]
        n = null_by_key[key]
        s_ok = g["substance_aggregate"] > n["substance_aggregate"]
        v_ok = g["voice_combined"] > n["voice_combined"]
        substance_correct += int(s_ok)
        voice_correct += int(v_ok)
        print(f"  {key[0]}/{key[1]:10s}  sub: {'✓' if s_ok else '✗'} ({g['substance_aggregate']:.3f} vs {n['substance_aggregate']:.3f})  "
              f"voice: {'✓' if v_ok else '✗'} ({g['voice_combined']:.3f} vs {n['voice_combined']:.3f})")

    print(f"\nSubstance correct direction: {substance_correct}/{len(pairs)}")
    print(f"Voice correct direction:     {voice_correct}/{len(pairs)}")

    # --- Write CSV ---
    all_rows = gold_scored + null_scored
    cols = ["id", "mode", "label", "substance_aggregate", "substance_flagged",
            "flattening_flagged", "voice_tier1", "voice_tier2", "voice_combined"]
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults written to {OUT_FILE}")
    return passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
