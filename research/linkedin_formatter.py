#!/usr/bin/env python3
"""
LinkedIn post formatter helper.
Converts raw LinkedIn post text into proper JSONL format for the research dataset.

Two modes:
1. Interactive mode: paste posts one by one, script formats in real-time
2. File mode: paste all posts into a text file separated by "---", then convert

Usage:
    python3 linkedin_formatter.py --interactive    # Interactive mode
    python3 linkedin_formatter.py input.txt         # File mode (posts separated by ---)
"""

import sys
import json
import argparse
from pathlib import Path


def interactive_mode(output_file: str):
    """
    Interactive mode: user pastes posts one-by-one.
    Each post separated by typing "---" on its own line.
    """
    print("=" * 60)
    print("LinkedIn Post Formatter — Interactive Mode")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Copy a LinkedIn post from the browser")
    print("2. Paste it below and press Enter (multiple lines OK)")
    print("3. When done with the post, type '---' on a new line")
    print("4. Repeat for all posts")
    print("5. Type 'DONE' when finished\n")

    posts = []
    current_post = []
    post_num = 1

    while True:
        try:
            line = input()
        except EOFError:
            # Handle piped input ending
            if current_post:
                posts.append("\n".join(current_post).strip())
            break

        if line.strip() == "---":
            if current_post:
                post_text = "\n".join(current_post).strip()
                if post_text:
                    posts.append(post_text)
                    print(f"✓ Post {post_num} recorded ({len(post_text)} chars)")
                    post_num += 1
                current_post = []
        elif line.strip().upper() == "DONE":
            if current_post:
                post_text = "\n".join(current_post).strip()
                if post_text:
                    posts.append(post_text)
                    print(f"✓ Post {post_num} recorded ({len(post_text)} chars)")
            break
        else:
            current_post.append(line)

    # Write JSONL
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        for i, post in enumerate(posts, 1):
            record = {
                "id": f"linkedin_{i:03d}",
                "text": post
            }
            f.write(json.dumps(record) + "\n")

    print(f"\n✓ Done! Saved {len(posts)} posts to {output_path}")
    print(f"  Use for analysis: python3 analyze.py --linkedin-input {output_path}")


def file_mode(input_file: str, output_file: str):
    """
    File mode: read posts from a file separated by '---' on its own line.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)

    with open(input_path) as f:
        content = f.read()

    # Split on --- delimiter
    raw_posts = content.split("---")
    posts = [p.strip() for p in raw_posts if p.strip()]

    if not posts:
        print(f"Error: no posts found in {input_file}")
        print("Make sure posts are separated by '---' on its own line")
        sys.exit(1)

    # Write JSONL
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        for i, post in enumerate(posts, 1):
            record = {
                "id": f"linkedin_{i:03d}",
                "text": post
            }
            f.write(json.dumps(record) + "\n")

    print(f"✓ Converted {len(posts)} posts from {input_file}")
    print(f"✓ Output: {output_path}")
    print(f"  Use for analysis: python3 analyze.py --linkedin-input {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Format LinkedIn posts into JSONL for research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (paste posts one-by-one)
  python3 linkedin_formatter.py --interactive

  # File mode (posts in file.txt separated by ---)
  python3 linkedin_formatter.py linkedin_raw.txt

File format for mode 2:
  [post 1 text here, multiple lines OK]
  ---
  [post 2 text here]
  ---
  [post 3 text here]
  ...
        """
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Text file with posts separated by '---' (if not using --interactive)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: paste posts one-by-one"
    )
    parser.add_argument(
        "-o", "--output",
        default="linkedin_samples.jsonl",
        help="Output JSONL file (default: linkedin_samples.jsonl)"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.output)
    elif args.input_file:
        file_mode(args.input_file, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
