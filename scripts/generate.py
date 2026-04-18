"""
Generation script — runs all 3 conditions × 2 modes for every input in data/inputs.jsonl.

Conditions:
  qwen        — Qwen 2.5 7B standalone
  haiku       — Claude Haiku standalone
  qwen_haiku  — Qwen output → Haiku edit pass (saves pre-edit intermediate too)

Outputs written to data/outputs/{condition}_{mode}.jsonl.
Resumes safely — skips samples already present in the output file.

Usage:
    python scripts/generate.py [--dry-run]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import trafilatura
import anthropic
from huggingface_hub import InferenceClient

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_TOKEN          = os.getenv("HF_TOKEN")

if not ANTHROPIC_API_KEY:
    sys.exit("ANTHROPIC_API_KEY not set in .env")
if not HF_TOKEN:
    sys.exit("HF_TOKEN not set in .env")

PROMPTS_DIR  = ROOT / "prompts"
DATA_DIR     = ROOT / "data"
OUTPUTS_DIR  = DATA_DIR / "outputs"
INPUTS_FILE  = DATA_DIR / "inputs.jsonl"

MODELS = {
    "qwen":  {"id": "Qwen/Qwen2.5-7B-Instruct", "provider": "hf"},
    "haiku": {"id": "claude-haiku-4-5-20251001", "provider": "anthropic"},
}

MODES = {
    "blog":     {"max_tokens": 900},
    "linkedin": {"max_tokens": 400},
}

from constants import HAIKU_INPUT_COST_PER_M as INPUT_COST_PER_M, HAIKU_OUTPUT_COST_PER_M as OUTPUT_COST_PER_M


# --- Prompt loading ---

def load_prompt(model: str, mode: str) -> tuple[str, str]:
    """Load system + user prompt by model and mode (e.g. model='haiku', mode='blog')."""
    system    = (PROMPTS_DIR / f"{model}_{mode}_system.txt").read_text().strip()
    user_tmpl = (PROMPTS_DIR / f"{model}_{mode}_user.txt").read_text().strip()
    return system, user_tmpl


# --- Article fetching ---

def fetch_article(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch: {url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError("Could not extract article text.")
    return text


# --- Model calls ---

def call_hf(model_id: str, system: str, user_prompt: str, max_tokens: int) -> dict:
    client = InferenceClient(token=HF_TOKEN)
    start  = time.time()
    resp   = client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_prompt},
        ],
        model=model_id,
        max_tokens=max_tokens,
        temperature=0.85,
    )
    return {
        "output":             resp.choices[0].message.content.strip(),
        "input_tokens":       None,
        "output_tokens":      None,
        "estimated_cost_usd": None,
        "latency_ms":         round((time.time() - start) * 1000),
    }


def call_haiku(system: str, user_prompt: str, max_tokens: int) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    start  = time.time()
    resp   = client.messages.create(
        model=MODELS["haiku"]["id"],
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


def call_model(model_label: str, system: str, user_prompt: str, max_tokens: int) -> dict:
    cfg = MODELS[model_label]
    if cfg["provider"] == "anthropic":
        return call_haiku(system, user_prompt, max_tokens)
    return call_hf(cfg["id"], system, user_prompt, max_tokens)


# --- JSONL helpers ---

def load_jsonl(path: Path) -> dict:
    """Load existing output file as {id: record}. Returns empty dict if file missing."""
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


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print plan without making API calls")
    args = parser.parse_args()

    if not INPUTS_FILE.exists():
        sys.exit(f"inputs.jsonl not found at {INPUTS_FILE}")

    inputs = []
    with open(INPUTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                inputs.append(json.loads(line))

    print(f"Loaded {len(inputs)} inputs\n")

    # Build generation plan: condition → mode → output path
    plan = {
        "qwen":           {m: OUTPUTS_DIR / f"qwen_{m}.jsonl"          for m in MODES},
        "haiku":          {m: OUTPUTS_DIR / f"haiku_{m}.jsonl"         for m in MODES},
        "qwen_pre_edit":  {m: OUTPUTS_DIR / f"qwen_pre_edit_{m}.jsonl" for m in MODES},
        "qwen_haiku":     {m: OUTPUTS_DIR / f"qwen_haiku_{m}.jsonl"    for m in MODES},
    }

    if args.dry_run:
        total = len(inputs) * len(MODES) * 3  # qwen + haiku + qwen_haiku
        print(f"Dry run — would generate {total} outputs across {len(inputs)} inputs × 2 modes × 3 conditions")
        for condition, modes in plan.items():
            for mode, path in modes.items():
                existing = load_jsonl(path)
                print(f"  {condition:<16} {mode:<10} → {path.name}  ({len(existing)} already done)")
        return

    total_haiku_cost = 0.0

    for inp in inputs:
        sample_id   = inp["id"]
        article_url = inp["article_url"]
        notes       = inp["notes"]

        print(f"\n{'='*60}")
        print(f"Sample: {sample_id}  ({article_url})")

        # Fetch article once per sample — prefer local txt if present
        # (trafilatura blocked by paywalls/bot-detection on some sites;
        #  article text was manually copied to data/articles/{id}.txt in those cases)
        article_file = DATA_DIR / "articles" / f"{sample_id}.txt"
        print("  Fetching article...", end=" ", flush=True)
        try:
            if article_file.exists():
                article_text = article_file.read_text().strip()
                print(f"{len(article_text)} chars (local txt)")
            else:
                article_text = fetch_article(article_url)
                print(f"{len(article_text)} chars")
        except Exception as e:
            print(f"FAILED — {e}")
            continue

        for mode, mode_cfg in MODES.items():
            max_tokens = mode_cfg["max_tokens"]

            # --- qwen standalone ---
            qwen_path    = plan["qwen"][mode]
            qwen_done    = load_jsonl(qwen_path)
            qwen_output  = None

            if sample_id in qwen_done:
                print(f"  [skip] qwen/{mode} already done")
                qwen_output = qwen_done[sample_id]["output"]
            else:
                print(f"  Calling qwen/{mode}...", end=" ", flush=True)
                try:
                    system, user_tmpl = load_prompt("qwen", mode)
                    user_prompt = user_tmpl.format(article_text=article_text[:4000], notes=notes)
                    result = call_model("qwen", system, user_prompt, max_tokens)
                    record = {"id": sample_id, "article_text": article_text, **result}
                    append_jsonl(qwen_path, record)
                    qwen_output = result["output"]
                    print(f"{len(qwen_output)} chars  {result['latency_ms']}ms")
                except Exception as e:
                    print(f"FAILED — {e}")

            # --- haiku standalone ---
            haiku_path = plan["haiku"][mode]
            haiku_done = load_jsonl(haiku_path)

            if sample_id in haiku_done:
                print(f"  [skip] haiku/{mode} already done")
            else:
                print(f"  Calling haiku/{mode}...", end=" ", flush=True)
                try:
                    system, user_tmpl = load_prompt("haiku", mode)
                    user_prompt = user_tmpl.format(article_text=article_text[:4000], notes=notes)
                    result = call_model("haiku", system, user_prompt, max_tokens)
                    record = {"id": sample_id, "article_text": article_text, **result}
                    append_jsonl(haiku_path, record)
                    total_haiku_cost += result["estimated_cost_usd"] or 0
                    print(f"{len(result['output'])} chars  {result['latency_ms']}ms  ${result['estimated_cost_usd']:.6f}")
                except Exception as e:
                    print(f"FAILED — {e}")

            # --- qwen_haiku edit pass (requires qwen output) ---
            if qwen_output is None:
                print(f"  [skip] qwen_haiku/{mode} — no qwen output available")
                continue

            pre_edit_path    = plan["qwen_pre_edit"][mode]
            pre_edit_done    = load_jsonl(pre_edit_path)
            qwen_haiku_path  = plan["qwen_haiku"][mode]
            qwen_haiku_done  = load_jsonl(qwen_haiku_path)

            # Save pre-edit intermediate (just Qwen's output, relabelled)
            if sample_id not in pre_edit_done:
                record = {"id": sample_id, "article_text": article_text, "output": qwen_output,
                          "input_tokens": None, "output_tokens": None,
                          "estimated_cost_usd": None, "latency_ms": None}
                append_jsonl(pre_edit_path, record)

            if sample_id in qwen_haiku_done:
                print(f"  [skip] qwen_haiku/{mode} already done")
            else:
                edit_system, edit_tmpl = load_prompt("haiku", f"edit_{mode}")
                edit_prompt = edit_tmpl.format(output=qwen_output)

                print(f"  Calling qwen_haiku/{mode}...", end=" ", flush=True)
                try:
                    result = call_haiku(edit_system, edit_prompt, max_tokens)
                    record = {"id": sample_id, "article_text": article_text, **result}
                    append_jsonl(qwen_haiku_path, record)
                    total_haiku_cost += result["estimated_cost_usd"] or 0
                    print(f"{len(result['output'])} chars  {result['latency_ms']}ms  ${result['estimated_cost_usd']:.6f}")
                except Exception as e:
                    print(f"FAILED — {e}")

    print(f"\n{'='*60}")
    print(f"Done. Total Haiku cost: ${total_haiku_cost:.6f}")


if __name__ == "__main__":
    main()
