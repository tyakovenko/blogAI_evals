"""
Evaluation pipeline — scores all condition/mode output files and writes results/scores.csv.

Conditions scored:
  qwen, haiku, qwen_pre_edit, qwen_haiku

Each output JSONL must have fields: id, output, article_text
inputs.jsonl must have: id, notes, domain, note_complexity, complexity_tier

Usage:
    python eval/run_all.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "data" / "outputs"
INPUTS_FILE = ROOT / "data" / "inputs.jsonl"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT / "eval"))

from substance_fidelity import score as substance_score
from voice_rubric        import score as voice_score
from factual_consistency import score as factual_score

# Conditions and modes are derived from output files at runtime — not hardcoded.
# Any {condition}_{mode}.jsonl in OUTPUTS_DIR is scored automatically.
# Ordering here is for display only; missing files are skipped without error.
KNOWN_CONDITIONS = ["qwen", "haiku", "qwen_pre_edit", "qwen_haiku"]
MODES            = ["blog", "linkedin"]

OUTPUT_COLS = [
    "id", "condition", "mode", "domain", "note_complexity", "complexity_tier",
    # Substance
    "substance_aggregate",
    "substance_claim", "substance_evidence", "substance_logic", "substance_implication",
    "substance_flagged", "flattening_flagged",
    # Claim-type survival
    "claimtype_opinion_score", "claimtype_fact_score",
    "claimtype_connection_score", "claimtype_question_score",
    # Voice
    "voice_tier1", "voice_tier2", "voice_combined", "voice_delta",
    # Factual consistency
    "contradiction_rate", "contradiction_flagged",
    # Cost / perf
    "edit_scope",
    "haiku_input_tokens", "haiku_output_tokens", "haiku_cost_usd",
    "latency_ms",
]


def load_jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["id"]] = r
    return records


def compute_edit_scope(pre_edit_output: str, edited_output: str) -> float:
    """Sentence-level change rate: (added + removed) / total pre-edit sentences."""
    def sentences(text):
        import re
        return set(s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10)

    pre  = sentences(pre_edit_output)
    post = sentences(edited_output)
    if not pre:
        return 0.0
    changed = len(pre.symmetric_difference(post))
    return round(changed / len(pre), 4)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    if not INPUTS_FILE.exists():
        sys.exit(f"inputs.jsonl not found at {INPUTS_FILE}")

    inputs = load_jsonl(INPUTS_FILE)

    # Pre-load pre-edit outputs for edit_scope calculation
    pre_edit_outputs = {
        mode: load_jsonl(OUTPUTS_DIR / f"qwen_pre_edit_{mode}.jsonl")
        for mode in MODES
    }

    # Discover all conditions present in the outputs directory
    present_conditions = []
    for cond in KNOWN_CONDITIONS:
        if any((OUTPUTS_DIR / f"{cond}_{mode}.jsonl").exists() for mode in MODES):
            present_conditions.append(cond)
    if not present_conditions:
        sys.exit(f"No output files found in {OUTPUTS_DIR}")
    print(f"Conditions found: {present_conditions}")

    rows = []
    skipped_missing_input = 0
    for condition in present_conditions:
        for mode in MODES:
            output_file = OUTPUTS_DIR / f"{condition}_{mode}.jsonl"
            if not output_file.exists():
                print(f"  Skipping {condition}/{mode} — file not found")
                continue

            outputs = load_jsonl(output_file)
            print(f"Scoring {condition}/{mode} ({len(outputs)} samples)...")

            for sample_id, out in outputs.items():
                if sample_id not in inputs:
                    print(f"  Warning: {sample_id} not in inputs — skipping")
                    skipped_missing_input += 1
                    continue

                inp       = inputs[sample_id]
                generated = out.get("output", "")

                s = substance_score(inp["notes"], generated)
                v = voice_score(generated, mode)
                f = factual_score(inp.get("article_text", out.get("article_text", "")), generated)

                # Edit scope — only for qwen_haiku, requires pre-edit output
                edit_scope = None
                if condition == "qwen_haiku":
                    pre = pre_edit_outputs[mode].get(sample_id, {})
                    if pre.get("output"):
                        edit_scope = compute_edit_scope(pre["output"], generated)

                # Voice — linkedin rubric returns flat score dict, blog returns tiered
                if mode == "linkedin":
                    voice_tier1  = v.get("score")
                    voice_tier2  = None
                    voice_combined = v.get("score")
                    voice_delta  = None
                else:
                    voice_tier1   = v.get("tier1")
                    voice_tier2   = v.get("tier2")
                    voice_combined = v.get("combined")
                    voice_delta   = v.get("delta")

                rows.append({
                    "id":              sample_id,
                    "condition":       condition,
                    "mode":            mode,
                    "domain":          inp.get("domain", ""),
                    "note_complexity": inp.get("note_complexity"),
                    "complexity_tier": inp.get("complexity_tier"),
                    # Substance
                    "substance_aggregate":   s["aggregate"],
                    "substance_claim":       s["by_component"].get("claim"),
                    "substance_evidence":    s["by_component"].get("evidence"),
                    "substance_logic":       s["by_component"].get("logic"),
                    "substance_implication": s["by_component"].get("implication"),
                    "substance_flagged":     int(s["substance_flagged"]),
                    "flattening_flagged":    int(s["flattening_flagged"]),
                    # Claim-type
                    "claimtype_opinion_score":    s["by_claimtype"].get("opinion"),
                    "claimtype_fact_score":       s["by_claimtype"].get("fact"),
                    "claimtype_connection_score": s["by_claimtype"].get("connection"),
                    "claimtype_question_score":   s["by_claimtype"].get("question"),
                    # Voice
                    "voice_tier1":    voice_tier1,
                    "voice_tier2":    voice_tier2,
                    "voice_combined": voice_combined,
                    "voice_delta":    voice_delta,
                    # Factual
                    "contradiction_rate":    f["contradiction_rate"],
                    "contradiction_flagged": int(f["flagged"]),
                    # Cost
                    "edit_scope":         edit_scope,
                    "haiku_input_tokens":  out.get("input_tokens"),
                    "haiku_output_tokens": out.get("output_tokens"),
                    "haiku_cost_usd":      out.get("estimated_cost_usd"),
                    "latency_ms":          out.get("latency_ms"),
                })

    out_path = RESULTS_DIR / "scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        writer.writeheader()
        writer.writerows(rows)

    flagged_substance = sum(r["substance_flagged"] for r in rows)
    flagged_factual   = sum(r["contradiction_flagged"] for r in rows)
    print(f"\nDone. {len(rows)} rows → {out_path}")
    print(f"  Substance flagged:    {flagged_substance}/{len(rows)}")
    print(f"  Contradiction flagged: {flagged_factual}/{len(rows)}")
    if skipped_missing_input:
        print(f"  WARNING: {skipped_missing_input} outputs skipped — sample ID not found in inputs.jsonl")


if __name__ == "__main__":
    main()
