"""
Generate null baseline outputs for calibration set.
Null baseline: article summary only — notes are ignored.
This establishes the floor: a model that ignores the user's voice entirely.

Generates blog + linkedin for each calibration sample using Haiku.
Output: data/calibration/null_baseline.jsonl

Usage:
    python3 scripts/gen_null_baseline.py
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import anthropic
import trafilatura

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    sys.exit("ANTHROPIC_API_KEY not set in .env")

HAIKU_MODEL = "claude-haiku-4-5-20251001"

from constants import HAIKU_INPUT_COST_PER_M as INPUT_COST_PER_M, HAIKU_OUTPUT_COST_PER_M as OUTPUT_COST_PER_M

INPUTS_FILE   = ROOT / "data" / "inputs.jsonl"
OUTPUT_FILE   = ROOT / "data" / "calibration" / "null_baseline.jsonl"

# Calibration sample IDs — first 5 inputs
CALIBRATION_IDS = ["sample_001", "sample_002", "sample_003", "sample_004", "sample_005"]

NULL_BLOG_SYSTEM = """You are writing a blog post summarizing a source article.
Write a clear, informative summary in prose. Short paragraphs. No bullet points. No headers.
300–500 words. Include a title."""

NULL_BLOG_USER = """Write a blog post summarizing the following article.

Article:
{article_text}

Blog post:"""

NULL_LINKEDIN_SYSTEM = """You are writing a LinkedIn post summarizing a source article.
150–250 words. Open with the main point. End with a question. No hashtags. No bullet points."""

NULL_LINKEDIN_USER = """Write a LinkedIn post summarizing the following article.

Article:
{article_text}

LinkedIn post:"""

MODES = {
    "blog":     {"system": NULL_BLOG_SYSTEM,     "user": NULL_BLOG_USER,     "max_tokens": 900},
    "linkedin": {"system": NULL_LINKEDIN_SYSTEM, "user": NULL_LINKEDIN_USER, "max_tokens": 400},
}


def fetch_article(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch: {url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError("Could not extract article text.")
    return text


def call_haiku(system: str, user_prompt: str, max_tokens: int) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    start  = time.time()
    resp   = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    input_tokens  = resp.usage.input_tokens
    output_tokens = resp.usage.output_tokens
    cost = round(
        (input_tokens  / 1_000_000) * INPUT_COST_PER_M +
        (output_tokens / 1_000_000) * OUTPUT_COST_PER_M,
        6,
    )
    return {
        "output":             resp.content[0].text.strip(),
        "input_tokens":       input_tokens,
        "output_tokens":      output_tokens,
        "estimated_cost_usd": cost,
        "latency_ms":         round((time.time() - start) * 1000),
    }


def load_inputs() -> dict:
    records = {}
    with open(INPUTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["id"]] = r
    return records


def load_existing() -> set:
    if not OUTPUT_FILE.exists():
        return set()
    seen = set()
    with open(OUTPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                seen.add((r["id"], r["mode"]))
    return seen


def main():
    inputs   = load_inputs()
    existing = load_existing()
    total_cost = 0.0

    for sample_id in CALIBRATION_IDS:
        if sample_id not in inputs:
            print(f"  {sample_id} not found in inputs — skipping")
            continue

        inp = inputs[sample_id]
        print(f"\nSample: {sample_id}")

        print("  Fetching article...", end=" ", flush=True)
        try:
            article_text = fetch_article(inp["article_url"])
            print(f"{len(article_text)} chars")
        except Exception as e:
            print(f"FAILED — {e}")
            continue

        for mode, cfg in MODES.items():
            if (sample_id, mode) in existing:
                print(f"  [skip] {mode} already done")
                continue

            user_prompt = cfg["user"].format(article_text=article_text[:4000])
            print(f"  Calling haiku/{mode}...", end=" ", flush=True)
            try:
                result = call_haiku(cfg["system"], user_prompt, cfg["max_tokens"])
                record = {
                    "id":                 sample_id,
                    "mode":               mode,
                    "article_text":       article_text,
                    "output":             result["output"],
                    "input_tokens":       result["input_tokens"],
                    "output_tokens":      result["output_tokens"],
                    "estimated_cost_usd": result["estimated_cost_usd"],
                    "latency_ms":         result["latency_ms"],
                    "baseline_type":      "null",
                }
                with open(OUTPUT_FILE, "a") as f:
                    f.write(json.dumps(record) + "\n")
                total_cost += result["estimated_cost_usd"]
                print(f"{len(result['output'])} chars  {result['latency_ms']}ms  ${result['estimated_cost_usd']:.6f}")
            except Exception as e:
                print(f"FAILED — {e}")

    print(f"\nDone. Total cost: ${total_cost:.6f}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
