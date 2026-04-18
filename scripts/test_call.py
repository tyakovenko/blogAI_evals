"""
Smoke test: real calls across active models (Qwen + Haiku) × 2 modes.
Verifies full response capture including token costs for Haiku.

Usage:
    python scripts/test_call.py <article_url> "<notes>"
"""

import sys
import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import trafilatura
import anthropic
from huggingface_hub import InferenceClient

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_TOKEN          = os.getenv("HF_TOKEN")

if not ANTHROPIC_API_KEY:
    sys.exit("ANTHROPIC_API_KEY not set in .env")
if not HF_TOKEN:
    sys.exit("HF_TOKEN not set in .env")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

from constants import HAIKU_INPUT_COST_PER_M as INPUT_COST_PER_M, HAIKU_OUTPUT_COST_PER_M as OUTPUT_COST_PER_M

MODELS = {
    "qwen":  {"id": "Qwen/Qwen2.5-7B-Instruct",  "provider": "hf"},
    "haiku": {"id": "claude-haiku-4-5-20251001", "provider": "anthropic"},
}

MODES = {
    "blog":     {"max_tokens": 900},
    "linkedin": {"max_tokens": 400},
}


def fetch_article(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch: {url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError("Could not extract article text.")
    return text


def load_prompt(model_label: str, mode: str) -> tuple[str, str]:
    """Load system + user prompt templates. HF models share haiku prompts for now."""
    prefix = "haiku"  # all models use the same prompt format until per-model prompts are written
    system   = (PROMPTS_DIR / f"{prefix}_{mode}_system.txt").read_text().strip()
    user_tmpl = (PROMPTS_DIR / f"{prefix}_{mode}_user.txt").read_text().strip()
    return system, user_tmpl


def call_hf(model_id: str, system: str, user_prompt: str, max_tokens: int) -> dict:
    client = InferenceClient(token=HF_TOKEN)
    start = time.time()
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_prompt},
        ],
        model=model_id,
        max_tokens=max_tokens,
        temperature=0.85,
    )
    latency_ms = round((time.time() - start) * 1000)
    return {
        "output":             response.choices[0].message.content.strip(),
        "input_tokens":       None,
        "output_tokens":      None,
        "estimated_cost_usd": None,
        "latency_ms":         latency_ms,
    }


def call_haiku(system: str, user_prompt: str, max_tokens: int) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    start = time.time()
    response = client.messages.create(
        model=MODELS["haiku"]["id"],
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    latency_ms    = round((time.time() - start) * 1000)
    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = round(
        (input_tokens  / 1_000_000) * INPUT_COST_PER_M +
        (output_tokens / 1_000_000) * OUTPUT_COST_PER_M,
        6,
    )
    return {
        "output":             response.content[0].text.strip(),
        "input_tokens":       input_tokens,
        "output_tokens":      output_tokens,
        "estimated_cost_usd": cost,
        "latency_ms":         latency_ms,
    }


def call_model(model_label: str, system: str, user_prompt: str, max_tokens: int) -> dict:
    cfg = MODELS[model_label]
    if cfg["provider"] == "anthropic":
        return call_haiku(system, user_prompt, max_tokens)
    return call_hf(cfg["id"], system, user_prompt, max_tokens)


def main():
    if len(sys.argv) < 3:
        sys.exit('Usage: python scripts/test_call.py <article_url> "<notes>"')

    url   = sys.argv[1]
    notes = sys.argv[2]

    print(f"\nFetching article: {url}")
    article_text = fetch_article(url)
    print(f"Article fetched — {len(article_text)} chars\n")

    results = {}
    for model_label in MODELS:
        results[model_label] = {}
        for mode, cfg in MODES.items():
            system, user_tmpl = load_prompt(model_label, mode)
            user_prompt = user_tmpl.format(article_text=article_text[:4000], notes=notes)

            print(f"Calling {model_label} — {mode}...", end=" ", flush=True)
            try:
                result = call_model(model_label, system, user_prompt, cfg["max_tokens"])
                results[model_label][mode] = result

                cost_str = f"  cost ${result['estimated_cost_usd']:.6f}" if result["estimated_cost_usd"] else ""
                tokens_str = f"  in={result['input_tokens']} out={result['output_tokens']}" if result["input_tokens"] else ""
                print(f"{len(result['output'])} chars  {result['latency_ms']}ms{tokens_str}{cost_str}")

            except Exception as e:
                print(f"FAILED — {e}")
                results[model_label][mode] = {"error": str(e)}

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'model':<10} {'mode':<10} {'chars':<8} {'latency':>10}  {'cost':>12}")
    print("-" * 60)
    for model_label, modes in results.items():
        for mode, r in modes.items():
            if "error" in r:
                print(f"{model_label:<10} {mode:<10} ERROR: {r['error']}")
            else:
                cost = f"${r['estimated_cost_usd']:.6f}" if r["estimated_cost_usd"] else "free tier"
                print(f"{model_label:<10} {mode:<10} {len(r['output']):<8} {r['latency_ms']:>8}ms  {cost:>12}")

    # Full outputs
    print("\n" + "=" * 60)
    for model_label, modes in results.items():
        for mode, r in modes.items():
            if "error" not in r:
                print(f"\n--- {model_label} / {mode} ---\n{r['output'][:300]}...\n")

    out = {
        "id":           "smoke_test_001",
        "article_url":  url,
        "notes":        notes,
        "article_text": article_text,
        "results":      results,
    }
    out_path = Path(__file__).resolve().parent.parent / "data" / "smoke_test_output.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull output written to {out_path}")


if __name__ == "__main__":
    main()
