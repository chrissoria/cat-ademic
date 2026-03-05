![cat-ademic Logo](https://github.com/chrissoria/cat-ademic/blob/main/images/logo.png?raw=True)

# cat-ademic

LLM-powered category analysis for academic paper abstracts via OpenAlex.

[![PyPI - Version](https://img.shields.io/pypi/v/cat-ademic.svg)](https://pypi.org/project/cat-ademic)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cat-ademic.svg)](https://pypi.org/project/cat-ademic)

-----

## The Problem

If you study a research field, you know the challenge: hundreds of papers need to be characterized before you can map what a journal publishes, how methods are evolving, or where the gaps are. Manual reading doesn't scale. Keyword search misses nuance.

## The Solution

cat-ademic fetches paper abstracts directly from [OpenAlex](https://openalex.org/) and uses LLMs to classify, extract, and explore categories across them. It handles:

- **Category Assignment** (`classify`): Classify papers into your predefined categories (multi-label supported)
- **Category Extraction** (`extract`): Automatically discover and extract categories from abstracts when you don't have a predefined scheme
- **Category Exploration** (`explore`): Analyze category stability and saturation through repeated raw extraction

No manual downloading. Point it at a journal ISSN (or OpenAlex topic), set a date range, and get back a structured CSV.

-----

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Best Practices for Classification](#best-practices-for-classification)
- [Configuration](#configuration)
- [Supported Models](#supported-models)
- [API Reference](#api-reference)
  - [classify()](#classify)
  - [extract()](#extract)
  - [explore()](#explore)
- [Related Projects](#related-projects)
- [Academic Research](#academic-research)
- [Contributing & Support](#contributing--support)
- [License](#license)

## Installation

```console
pip install cat-ademic
```

-----

## Quick Start

```python
import catademic as cat
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]

# Classify 250 recent papers from Social Science Computer Review
results = cat.classify(
    categories=[
        "Introduces new computational tool or method",
        "Applies LLM/AI to social science data",
        "Evaluates or benchmarks a method",
        "Improves survey or data collection",
        "Theory-driven / conceptual",
        "Other",
    ],
    journal_issn="0894-4393",
    paper_limit=250,
    date_from="2023-01-01",
    description="Academic papers from Social Science Computer Review",
    api_key=api_key,
    filename="out/sscr_classified.csv",
)

print(results.head(10))
```

```python
# Discover emergent categories without a predefined scheme
raw_categories = cat.explore(
    journal_issn="0894-4393",
    paper_limit=250,
    date_from="2023-01-01",
    description="Academic papers from Social Science Computer Review",
    api_key=api_key,
    filename="out/sscr_categories_raw.csv",
)

print(f"Total raw category strings extracted: {len(raw_categories)}")
```

-----

## Best Practices for Classification

These recommendations are based on empirical testing across multiple datasets and models (7B to frontier-class).

### What works

- **Detailed category descriptions**: The single biggest lever for accuracy. Instead of short labels like `"Methods paper"`, use verbose descriptions like `"Introduces a new computational tool or method, including software packages, algorithms, or pipelines."` This consistently improves accuracy across all models.
- **Include an "Other" category**: Adding a catch-all category prevents the model from forcing ambiguous papers into ill-fitting categories.
- **Low temperature** (`creativity=0`): For classification tasks, deterministic output is generally preferable.

### What doesn't help (or hurts)

- **Chain of Thought** (`chain_of_thought`): Does not reliably improve classification accuracy and adds cost.
- **Chain of Verification** (`chain_of_verification`): Uses ~4x the API calls. Tends to retract correct classifications during the verification step. Not recommended.
- **Step-back prompting** (`step_back_prompt`): Inconsistent results across datasets. Not recommended as a default.

### Summary

The most effective approach is: **write detailed category descriptions, include an "Other" category, and use a capable model at low temperature.**

-----

## Configuration

### Get Your API Key

Get an API key from your preferred provider:

- **OpenAI**: [platform.openai.com](https://platform.openai.com)
- **Anthropic**: [console.anthropic.com](https://console.anthropic.com)
- **Google**: [aistudio.google.com](https://aistudio.google.com)
- **Huggingface**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **xAI**: [console.x.ai](https://console.x.ai)
- **Mistral**: [console.mistral.ai](https://console.mistral.ai)
- **Perplexity**: [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)

OpenAlex is unauthenticated and requires no key. Providing a `polite_email` in your requests is recommended for higher rate limits.

## Supported Models

- **OpenAI**: GPT-4o, GPT-4, GPT-5, etc.
- **Anthropic**: Claude Sonnet 4, Claude 3.5 Sonnet, Claude Haiku, etc.
- **Google**: Gemini 2.5 Flash, Gemini 2.5 Pro, etc.
- **Huggingface**: Qwen, Llama 4, DeepSeek, and thousands of community models
- **xAI**: Grok models
- **Mistral**: Mistral Large, etc.
- **Perplexity**: Sonar Large, Sonar Small, etc.

**Note:** For best results, start with OpenAI or Anthropic.

-----

## API Reference

### `classify()`

Classify academic paper abstracts into predefined categories. Abstracts are fetched automatically from OpenAlex when you provide a `journal_issn` or `topic_id`.

**Parameters:**
- `categories` (list): List of category descriptions for classification
- `journal_issn` (str): Journal ISSN to pull abstracts from via OpenAlex
- `journal_name` (str, optional): Journal name for filtering by name instead of ISSN
- `topic_id` (str, optional): OpenAlex topic ID to pull papers by research topic
- `topic_name` (str, optional): OpenAlex topic name to pull papers by research topic
- `paper_limit` (int, default=50): Number of papers to fetch
- `date_from` (str, optional): Start date filter as `"YYYY-MM-DD"`
- `date_to` (str, optional): End date filter as `"YYYY-MM-DD"`
- `polite_email` (str, optional): Email for OpenAlex polite pool (higher rate limits)
- `api_key` (str): API key for the LLM service
- `description` (str): Description of the corpus context
- `user_model` (str, default="gpt-4o"): Model to use
- `model_source` (str, default="auto"): Provider — "auto", "openai", "anthropic", "google", "mistral", "perplexity", "huggingface", "xai"
- `creativity` (float, optional): Temperature setting (0.0–1.0)
- `chain_of_thought` (bool, default=False): Enable step-by-step reasoning
- `filename` (str, optional): Output CSV filename
- `save_directory` (str, optional): Directory to save results

**Returns:**
- `pandas.DataFrame`: Results with one binary column per category (`category_1`, `category_2`, …), plus `title`, `doi`, `publication_date`, `cited_by_count`

**Example:**

```python
import catademic as cat

results = cat.classify(
    categories=[
        "Introduces new computational tool or method",
        "Applies LLM/AI to social science data",
        "Theory-driven / conceptual",
        "Other",
    ],
    journal_issn="0894-4393",
    paper_limit=100,
    date_from="2023-01-01",
    description="Social Science Computer Review papers",
    api_key=api_key,
    filename="sscr_classified.csv",
)

# Add readable column names
results["new_tool"] = results["category_1"]
results["applies_ai"] = results["category_2"]
results.to_csv("sscr_classified.csv", index=False)
```

---

### `extract()`

Automatically discover and extract categories from paper abstracts when you don't have a predefined scheme. Returns a clean, deduplicated, semantically merged set of categories.

**Parameters:**
- `journal_issn` (str): Journal ISSN to pull abstracts from via OpenAlex
- `journal_name` (str, optional): Journal name for filtering
- `topic_id` (str, optional): OpenAlex topic ID
- `topic_name` (str, optional): OpenAlex topic name
- `paper_limit` (int, default=50): Number of papers to fetch
- `date_from` (str, optional): Start date filter as `"YYYY-MM-DD"`
- `date_to` (str, optional): End date filter as `"YYYY-MM-DD"`
- `polite_email` (str, optional): Email for OpenAlex polite pool
- `api_key` (str): API key for the LLM service
- `description` (str): Description of the corpus
- `max_categories` (int, default=12): Maximum number of categories to return
- `categories_per_chunk` (int, default=10): Categories to extract per chunk
- `divisions` (int, default=12): Number of chunks to divide data into
- `iterations` (int, default=8): Number of extraction passes over the data
- `user_model` (str, default="gpt-4o"): Model to use
- `specificity` (str, default="broad"): `"broad"` or `"specific"` category granularity
- `research_question` (str, optional): Research context to guide extraction
- `focus` (str, optional): Focus instruction (e.g., `"methodological contributions"`)
- `filename` (str, optional): Output CSV filename
- `model_source` (str, default="auto"): Provider

**Returns:**
- `dict` with keys:
  - `counts_df`: DataFrame of categories with counts
  - `top_categories`: List of top category names
  - `raw_top_text`: Raw model output

**Example:**

```python
import catademic as cat

results = cat.extract(
    journal_issn="0894-4393",
    paper_limit=250,
    date_from="2023-01-01",
    description="Social Science Computer Review papers",
    api_key=api_key,
    max_categories=10,
    focus="methodological contributions",
)

print(results["top_categories"])
# ['Computational text analysis', 'Survey methodology', 'Network analysis', ...]
```

---

### `explore()`

Raw category extraction for frequency and saturation analysis. Unlike `extract()`, which normalizes and merges categories into a clean final set, `explore()` returns **every category string from every chunk across every iteration** — with duplicates intact.

This is useful for analyzing which categories are robust (consistently discovered across runs) versus noise (appearing only once or twice). Increasing `iterations` lets you build saturation curves showing when category discovery converges.

**Parameters:**
- `journal_issn` (str): Journal ISSN to pull abstracts from via OpenAlex
- `journal_name` (str, optional): Journal name for filtering
- `topic_id` (str, optional): OpenAlex topic ID
- `topic_name` (str, optional): OpenAlex topic name
- `paper_limit` (int, default=50): Number of papers to fetch
- `date_from` (str, optional): Start date filter as `"YYYY-MM-DD"`
- `date_to` (str, optional): End date filter as `"YYYY-MM-DD"`
- `polite_email` (str, optional): Email for OpenAlex polite pool
- `api_key` (str): API key for the LLM service
- `description` (str): Description of the corpus
- `categories_per_chunk` (int, default=10): Categories to extract per chunk
- `divisions` (int, default=12): Number of chunks to divide data into
- `iterations` (int, default=8): Number of passes over the data
- `user_model` (str, default="gpt-4o"): Model to use
- `specificity` (str, default="broad"): `"broad"` or `"specific"` category granularity
- `research_question` (str, optional): Research context to guide extraction
- `focus` (str, optional): Focus instruction for extraction
- `random_state` (int, optional): Random seed for reproducibility
- `filename` (str, optional): Output CSV filename (one category per row)
- `model_source` (str, default="auto"): Provider

**Returns:**
- `list[str]`: Every category extracted from every chunk across every iteration. Length ≈ `iterations × divisions × categories_per_chunk`.

**Example:**

```python
import catademic as cat
from collections import Counter

raw_categories = cat.explore(
    journal_issn="0894-4393",
    paper_limit=250,
    date_from="2023-01-01",
    description="Social Science Computer Review papers",
    api_key=api_key,
    iterations=20,
    filename="sscr_categories_raw.csv",
)

counts = Counter(raw_categories)
for category, freq in counts.most_common(15):
    print(f"{freq:3d}x  {category}")
```

-----

## Related Projects

- **[cat-llm](https://github.com/chrissoria/cat-llm)**: The survey response version of this tool — classifies and extracts categories from open-ended survey responses, images, and PDFs.
- **[llm-web-research](https://github.com/chrissoria/llm-web-research)**: LLM-powered web research with a Funnel of Verification methodology.

## Academic Research

If you use this package for research, please cite:

Soria, C. (2025). cat-ademic (0.1.0). GitHub. https://github.com/chrissoria/cat-ademic

## Contributing & Support

- **Report bugs or request features**: [Open a GitHub Issue](https://github.com/chrissoria/cat-ademic/issues)
- **Research collaboration**: Email [ChrisSoria@Berkeley.edu](mailto:ChrisSoria@Berkeley.edu)

## License

`cat-ademic` is distributed under the terms of the [GNU GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
