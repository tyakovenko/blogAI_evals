"""
Blog post scraper for platform style research.
Fetches blog posts via trafilatura and saves to blog_samples.jsonl
"""

import sys
import json
import argparse
from pathlib import Path
import trafilatura


def scrape_url(url: str) -> dict | None:
    """
    Fetch and extract clean text from a URL.
    Returns {url, title, text, error} or None on critical failure.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {"url": url, "title": None, "text": None, "error": "fetch failed"}

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            output_format='plaintext'
        )

        if not text:
            return {"url": url, "title": None, "text": None, "error": "extract failed"}

        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata else None

        return {
            "url": url,
            "title": title,
            "text": text,
            "error": None
        }
    except Exception as e:
        return {"url": url, "title": None, "text": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Scrape blog posts for platform style research")
    parser.add_argument(
        "--input",
        default="blog_urls.txt",
        help="Path to file with blog URLs (one per line)"
    )
    parser.add_argument(
        "--output",
        default="blog_samples.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test with a single sample URL"
    )
    args = parser.parse_args()

    # Load URLs
    input_path = Path(args.input)
    if not input_path.exists():
        if args.dry_run:
            print("Dry run: testing with example URL...")
            test_url = "https://www.paulgraham.com/avg.html"  # Known good URL
            result = scrape_url(test_url)
            if result and result.get("text"):
                print(f"✓ Scrape successful for {test_url}")
                print(f"  Title: {result.get('title', 'N/A')}")
                print(f"  Text preview: {result['text'][:200]}...")
            else:
                print(f"✗ Scrape failed: {result.get('error', 'unknown error')}")
            return
        else:
            print(f"Error: {input_path} not found. Create it with one blog URL per line.")
            sys.exit(1)

    with open(input_path) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not urls:
        print(f"No URLs found in {input_path}")
        sys.exit(1)

    print(f"Scraping {len(urls)} URLs...")

    output_path = Path(args.output)
    succeeded = 0
    failed = 0

    with open(output_path, 'w') as out:
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {url}...", end=" ", flush=True)
            result = scrape_url(url)

            if result and result.get("text"):
                # Write as JSONL: {id, url, title, text}
                record = {
                    "id": f"blog_{i:03d}",
                    "url": url,
                    "title": result.get("title"),
                    "text": result["text"]
                }
                out.write(json.dumps(record) + "\n")
                print("✓")
                succeeded += 1
            else:
                print(f"✗ ({result.get('error', 'unknown')})")
                failed += 1

    print(f"\nDone: {succeeded} succeeded, {failed} failed")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
