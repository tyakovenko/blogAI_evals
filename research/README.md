# Platform Style Research

## Goal

Extract structural and linguistic differences between blog posts and LinkedIn posts to build a **data-grounded LinkedIn rubric** for voice fidelity evaluation in Study 2.

The current LinkedIn rubric in `eval/voice_rubric.py` is a placeholder (routes to `score_blog()`). This research produces the real one.

---

## Files

- **`blog_urls.txt`** — list of blog post URLs (one per line). Populated by you manually.
- **`linkedin_samples.jsonl`** — LinkedIn post texts collected manually. Populated by you.
- **`scraper.py`** — fetches blog URLs via trafilatura, saves cleaned text to `blog_samples.jsonl`
- **`analyze.py`** — extracts 10+ features from both sets, writes `analysis_results.json`
- **`blog_samples.jsonl`** — output of `scraper.py`
- **`analysis_results.json`** — output of `analyze.py`, data for the rubric design

---

## Workflow

### Step 1: Collect blog URLs

Create `blog_urls.txt` with ~50 blog post URLs, one per line. Aim for:
- Tech/AI/research blogs (match your LinkedIn industry)
- Writing quality comparable to what you'd post
- Mix of personal essays, analysis, opinion, tutorials

Example:
```
https://paulgraham.com/avg.html
https://www.ribbonfarm.com/2023/...
https://example.com/blog/article-name
# Comment lines are ignored
```

Run scraper to fetch:
```bash
cd ~/Desktop/blogAI_evals
python research/scraper.py --input research/blog_urls.txt --output research/blog_samples.jsonl
```

Expect: ~20-30 successful fetches (some URLs will fail due to paywalls, JS-heavy pages, etc.)

### Step 2: Collect LinkedIn posts

Create `linkedin_samples.jsonl` with ~50 LinkedIn posts. Format (one JSON object per line):

```json
{"id": "linkedin_001", "text": "post text here"}
{"id": "linkedin_002", "text": "post text here"}
...
```

How to collect:
1. Find high-engagement LinkedIn posts in your field
2. Click the `...` menu and select "Copy post link"
3. Open in browser, copy all the text (Ctrl+A from the post body)
4. Paste into a text editor with the template above

Template for faster entry (copy-paste into VS Code, use multi-cursor):
```
{"id": "linkedin_001", "text": ""},
{"id": "linkedin_002", "text": ""},
...
```

Then fill in the `text` fields from your copied LinkedIn posts.

**Why manual collection?** LinkedIn blocks automated scraping. This is the only reliable, ToS-compliant way.

**Minimum quality:** 50 posts, across 2-3 industries/niches. The more diverse, the better the rubric will generalize.

### Step 3: Analyze

Run the analyzer:
```bash
python research/analyze.py \
  --blog-input research/blog_samples.jsonl \
  --linkedin-input research/linkedin_samples.jsonl \
  --output research/analysis_results.json
```

This extracts:
- **Hook velocity** — words until first sentence ends (does LinkedIn lead faster?)
- **Paragraph structure** — sentences per paragraph, max length (are LinkedIn paras shorter?)
- **Line breaks** — blank lines per 100 words (visual rhythm)
- **Sentence length** — mean/median (does LinkedIn use shorter sentences?)
- **Conversational markers** — "honestly", "here's the thing", "to be fair", etc.
- **Formal transitions** — "moreover", "furthermore", "thus", etc. (should be absent on LinkedIn)
- **Direct address** — "you" presence (connection signal)
- **Endings** — questions vs. CTAs ("comment below" vs. rhetorical questions)
- **Reaction openings** — first sentence has "I found", "I realized", etc.

Output: `analysis_results.json` with aggregated stats (means, medians, presence percentages) per platform.

### Step 4: Visualize (Optional but recommended)

### Step 5: Update the rubric

Based on `analysis_results.json`, the `score_linkedin()` function in `eval/voice_rubric.py` may need tuning. The proposed rubric has 7 checks:

1. `hook_in_first_sentence` — first sentence <30 words AND has reaction marker
2. `short_paragraphs` — no paragraph >3 sentences
3. `high_line_break_density` — ≥1 blank line per 80 words
4. `conversational_markers` — at least one present
5. `no_formal_transitions` — none of the 6 formal words present
6. `direct_address` — "you" present
7. `ends_with_question_or_cta` — ends with `?` OR CTA verb

If analysis shows, e.g., LinkedIn posts average 2.2 sentences/paragraph (not 3), adjust the check. If 90% of posts have a conversational marker vs. 40% of blogs, keep that check. Data drives the rubric.

---

## Testing

Dry-run the scraper (no URLs needed):
```bash
python research/scraper.py --dry-run
```

Dry-run the analyzer (generates test data):
```bash
python research/analyze.py --dry-run
```

---

## Output Format Details

### `blog_samples.jsonl` and `linkedin_samples.jsonl`

Each line is one JSON object:
```json
{"id": "blog_001", "url": "https://...", "title": "Title", "text": "full post text"}
{"id": "linkedin_001", "text": "full post text"}
```

For LinkedIn, you may not have `url` or `title` — that's fine, `text` is what matters.

### `analysis_results.json`

Structure:
```json
{
  "blog": {
    "n_samples": 30,
    "hook_velocity_words": {"mean": 15.3, "median": 14, ...},
    "sentences_per_paragraph_mean": {"mean": 2.8, ...},
    ...
  },
  "linkedin": {
    "n_samples": 50,
    "hook_velocity_words": {"mean": 12.1, "median": 11, ...},
    ...
  },
  "per_sample": {
    "blog": [{...}, ...],
    "linkedin": [{...}, ...]
  }
}
```

Use the `blog` and `linkedin` sections to compare platforms. Use `per_sample` to inspect individual outliers.

---

## Next Steps

1. Populate `blog_urls.txt` and `linkedin_samples.jsonl`
2. Run scraper + analyzer
3. Read `analysis_results.json`
4. Adjust `score_linkedin()` in `eval/voice_rubric.py` based on findings
5. Return findings to `blogAI/app/config.py` to tighten the LinkedIn prompt
