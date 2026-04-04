"""
Run full evaluation pipeline across all conditions and modes.
Reads data/inputs.jsonl + data/outputs/*.jsonl
Writes results/scores.csv
"""

import json
import csv
import os
from pathlib import Path

from substance_fidelity import score as substance_score
from voice_rubric import score as voice_score
from factual_consistency import score as factual_score

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

CONDITIONS = ["qwen_raw", "haiku_only", "qwen_haiku"]
MODES = ["blog", "linkedin"]

OUTPUT_COLS = [
    "id", "condition", "mode", "domain",
    "substance_score", "substance_flagged",
    "voice_score",
    "contradiction_rate", "contradiction_flagged",
    "latency_s",
]


def load_jsonl(path: Path) -> dict:
    """Load jsonl into {id: record} dict."""
    records = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            records[r["id"]] = r
    return records


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    inputs = load_jsonl(DATA_DIR / "inputs.jsonl")

    rows = []
    for condition in CONDITIONS:
        for mode in MODES:
            output_file = DATA_DIR / "outputs" / f"{condition}_{mode}.jsonl"
            if not output_file.exists():
                print(f"Skipping {condition}_{mode} — file not found")
                continue

            outputs = load_jsonl(output_file)
            print(f"Evaluating {condition}/{mode} ({len(outputs)} samples)...")

            for sample_id, out in outputs.items():
                if sample_id not in inputs:
                    print(f"  Warning: {sample_id} not in inputs, skipping")
                    continue

                inp = inputs[sample_id]
                generated = out.get("output", "")

                s_result = substance_score(inp["notes"], generated)
                v_result = voice_score(generated, mode)
                f_result = factual_score(inp["article_text"], generated)

                rows.append({
                    "id": sample_id,
                    "condition": condition,
                    "mode": mode,
                    "domain": inp.get("domain", ""),
                    "substance_score": s_result["score"],
                    "substance_flagged": int(s_result["flagged"]),
                    "voice_score": v_result["score"],
                    "contradiction_rate": f_result["contradiction_rate"],
                    "contradiction_flagged": int(f_result["flagged"]),
                    "latency_s": out.get("latency_s", ""),
                })

    out_path = RESULTS_DIR / "scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to {out_path}")


if __name__ == "__main__":
    main()
