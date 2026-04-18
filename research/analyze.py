"""
Platform style analysis: extract structural + linguistic features from blog and LinkedIn samples.
Produces analysis_results.json for visualization and rubric design.
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev, median
import spacy


def _load_samples(jsonl_path: str) -> list[dict]:
    """Load JSONL samples. Returns list of {id, text, ...}"""
    samples = []
    path = Path(jsonl_path)
    if not path.exists():
        return []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def _get_nlp():
    """Lazy load spaCy model."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Error: spaCy model 'en_core_web_sm' not found. Install with:")
        print("  python -m spacy download en_core_web_sm")
        raise


def _extract_features(text: str, nlp) -> dict:
    """Extract structural + linguistic features from a single text."""
    if not text:
        return {}

    features = {}

    # --- Hook velocity: words until first sentence ends ---
    doc = nlp(text[:1000])
    sents = list(doc.sents)
    if sents:
        first_sent_words = len(sents[0].text.split())
        features["hook_velocity_words"] = first_sent_words
        # Check if first sentence has reaction/claim markers
        first_lower = sents[0].text.lower()
        reaction_markers = r"\b(i|we)\s+(found|learned|realized|noticed|tried|discovered|saw|felt|think|believe|know)"
        features["first_sentence_has_reaction"] = bool(re.search(reaction_markers, first_lower))
    else:
        features["hook_velocity_words"] = None
        features["first_sentence_has_reaction"] = False

    # --- Paragraph structure ---
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    features["paragraph_count"] = len(paragraphs)

    para_sentence_counts = []
    for para in paragraphs:
        para_doc = nlp(para)
        para_sents = list(para_doc.sents)
        para_sentence_counts.append(len(para_sents))

    if para_sentence_counts:
        features["sentences_per_paragraph_mean"] = round(mean(para_sentence_counts), 2)
        features["sentences_per_paragraph_median"] = median(para_sentence_counts)
        features["sentences_per_paragraph_max"] = max(para_sentence_counts)
    else:
        features["sentences_per_paragraph_mean"] = None

    # --- Line break density: blank lines per 100 words ---
    word_count = len(text.split())
    blank_lines = len([l for l in text.split("\n") if not l.strip()])
    features["line_break_density"] = round((blank_lines / word_count * 100) if word_count > 0 else 0, 2)

    # --- Sentence length ---
    if sents:
        sent_words = [len(s.text.split()) for s in sents if s.text.strip()]
        if sent_words:
            features["sentence_length_mean"] = round(mean(sent_words), 2)
            features["sentence_length_median"] = median(sent_words)
            features["sentence_length_max"] = max(sent_words)

    # --- Conversational markers (presence) ---
    markers = {
        "honestly": r"\bhonestly\b",
        "here_is_the_thing": r"\b(here'?s? the thing|here goes)\b",
        "to_be_fair": r"\bto be fair\b",
        "lowkey": r"\blowkey\b",
        "this_is_why": r"\bthis is why\b",
    }
    text_lower = text.lower()
    features["conversational_markers"] = {k: bool(re.search(v, text_lower)) for k, v in markers.items()}

    # --- Formal transitions (presence) ---
    formal_transitions = {
        "moreover": r"\bmoreover\b",
        "furthermore": r"\bfurthermore\b",
        "in_addition": r"\bin addition\b",
        "in_conclusion": r"\bin conclusion\b",
        "thus": r"\bthus\b",
        "therefore": r"\btherefore\b",
    }
    features["formal_transitions"] = {k: bool(re.search(v, text_lower)) for k, v in formal_transitions.items()}

    # --- Direct address (you) ---
    features["has_direct_address"] = bool(re.search(r"\byou\b", text_lower))

    # --- Ending: question or CTA ---
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    last_line = lines[-1] if lines else ""
    features["ends_with_question"] = last_line.endswith("?")
    cta_markers = r"\b(share|comment|let me know|i'd love to hear|what do you think|your thoughts?)\b"
    features["ends_with_cta"] = bool(re.search(cta_markers, last_line.lower()))

    # --- Word count ---
    features["word_count"] = word_count

    return features


def analyze_samples(blog_samples: list[dict], linkedin_samples: list[dict]) -> dict:
    """
    Compare features across blog and LinkedIn samples.
    Returns aggregated statistics for visualization.
    """
    nlp = _get_nlp()

    print("Extracting features from blog samples...")
    blog_features = []
    for i, sample in enumerate(blog_samples):
        text = sample.get("text", "")
        features = _extract_features(text, nlp)
        features["id"] = sample.get("id", f"blog_{i}")
        features["platform"] = "blog"
        blog_features.append(features)

    print("Extracting features from LinkedIn samples...")
    linkedin_features = []
    for i, sample in enumerate(linkedin_samples):
        text = sample.get("text", "")
        features = _extract_features(text, nlp)
        features["id"] = sample.get("id", f"linkedin_{i}")
        features["platform"] = "linkedin"
        linkedin_features.append(features)

    # Aggregate statistics
    def agg_numeric(samples: list[dict], key: str) -> dict:
        """Aggregate numeric feature across samples."""
        values = [s.get(key) for s in samples if s.get(key) is not None]
        if not values:
            return {}
        return {
            "mean": round(mean(values), 2),
            "median": median(values),
            "min": min(values),
            "max": max(values),
            "stdev": round(stdev(values), 2) if len(values) > 1 else 0,
        }

    def agg_boolean(samples: list[dict], key: str) -> dict:
        """Aggregate boolean feature across samples."""
        values = [s.get(key, False) for s in samples]
        count_true = sum(1 for v in values if v)
        return {
            "present_count": count_true,
            "present_pct": round(count_true / len(values) * 100, 1) if values else 0,
        }

    result = {
        "blog": {
            "n_samples": len(blog_features),
            "hook_velocity_words": agg_numeric(blog_features, "hook_velocity_words"),
            "sentences_per_paragraph_mean": agg_numeric(blog_features, "sentences_per_paragraph_mean"),
            "line_break_density": agg_numeric(blog_features, "line_break_density"),
            "sentence_length_mean": agg_numeric(blog_features, "sentence_length_mean"),
            "word_count": agg_numeric(blog_features, "word_count"),
            "has_direct_address": agg_boolean(blog_features, "has_direct_address"),
            "ends_with_question": agg_boolean(blog_features, "ends_with_question"),
            "ends_with_cta": agg_boolean(blog_features, "ends_with_cta"),
            "first_sentence_has_reaction": agg_boolean(blog_features, "first_sentence_has_reaction"),
        },
        "linkedin": {
            "n_samples": len(linkedin_features),
            "hook_velocity_words": agg_numeric(linkedin_features, "hook_velocity_words"),
            "sentences_per_paragraph_mean": agg_numeric(linkedin_features, "sentences_per_paragraph_mean"),
            "line_break_density": agg_numeric(linkedin_features, "line_break_density"),
            "sentence_length_mean": agg_numeric(linkedin_features, "sentence_length_mean"),
            "word_count": agg_numeric(linkedin_features, "word_count"),
            "has_direct_address": agg_boolean(linkedin_features, "has_direct_address"),
            "ends_with_question": agg_boolean(linkedin_features, "ends_with_question"),
            "ends_with_cta": agg_boolean(linkedin_features, "ends_with_cta"),
            "first_sentence_has_reaction": agg_boolean(linkedin_features, "first_sentence_has_reaction"),
        },
        "per_sample": {
            "blog": blog_features,
            "linkedin": linkedin_features,
        }
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze platform style features")
    parser.add_argument(
        "--blog-input",
        default="blog_samples.jsonl",
        help="Path to blog samples JSONL"
    )
    parser.add_argument(
        "--linkedin-input",
        default="linkedin_samples.jsonl",
        help="Path to LinkedIn samples JSONL"
    )
    parser.add_argument(
        "--output",
        default="analysis_results.json",
        help="Output analysis results JSON"
    )
    parser.add_argument(
        "--linkedin-only",
        action="store_true",
        help="Skip blog samples — analyze LinkedIn corpus only"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test with minimal data"
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run: creating test samples...")
        test_blog = [{
            "id": "blog_test",
            "text": "This is an interesting observation. I found that when you combine A and B, something surprising happens. "
                    "The real implication is that C matters more than we thought. Here's why: the data shows D. "
                    "To be fair, E is also true. However, F complicates things. What do you think?"
        }]
        test_linkedin = [{
            "id": "linkedin_test",
            "text": "I just realized something.\n\nA + B = surprising.\n\nWhy?\nBecause C.\nWhich means D.\n\nComment below."
        }]
        result = analyze_samples(test_blog, test_linkedin)
    else:
        linkedin_samples = _load_samples(args.linkedin_input)
        blog_samples = [] if args.linkedin_only else _load_samples(args.blog_input)

        if not args.linkedin_only and not blog_samples:
            print(f"Warning: no blog samples found in {args.blog_input}")
        if not linkedin_samples:
            print(f"Warning: no LinkedIn samples found in {args.linkedin_input}")
            return

        print(f"Loaded {len(blog_samples)} blog samples, {len(linkedin_samples)} LinkedIn samples")
        result = analyze_samples(blog_samples, linkedin_samples)

    # Write output
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Analysis saved to {output_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    for platform in ["blog", "linkedin"]:
        print(f"\n{platform.upper()}:")
        print(f"  Samples: {result[platform]['n_samples']}")
        if result[platform]['hook_velocity_words']:
            print(f"  Hook velocity (words): {result[platform]['hook_velocity_words']['mean']:.1f} "
                  f"(median: {result[platform]['hook_velocity_words']['median']})")
        if result[platform]['sentences_per_paragraph_mean']:
            print(f"  Sentences/para: {result[platform]['sentences_per_paragraph_mean']['mean']:.1f}")
        if result[platform]['line_break_density']:
            print(f"  Line break density: {result[platform]['line_break_density']['mean']:.2f}% per 100 words")
        if result[platform]['has_direct_address']:
            print(f"  Has 'you': {result[platform]['has_direct_address']['present_pct']}%")
        if result[platform]['ends_with_question']:
            print(f"  Ends with question: {result[platform]['ends_with_question']['present_pct']}%")


if __name__ == "__main__":
    main()
