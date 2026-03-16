![cat-ademic Logo](https://github.com/chrissoria/cat-ademic/blob/main/images/logo.png?raw=True)

# cat-ademic

LLM-powered category analysis for academic paper abstracts via OpenAlex.

[![PyPI - Version](https://img.shields.io/pypi/v/cat-ademic.svg)](https://pypi.org/project/cat-ademic)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cat-ademic.svg)](https://pypi.org/project/cat-ademic)

-----

## The Problem

If you study a research field, you know the challenge: hundreds of papers need to be characterized before you can map what a journal publishes, how methods are evolving, or where the gaps are. Manual reading doesn't scale. Keyword search misses nuance.

## The Solution

cat-ademic fetches paper abstracts directly from [OpenAlex](https://openalex.org/) and uses LLMs to classify, extract, explore, and summarize them. It handles:

- **Category Assignment** (`classify`): Classify papers into your predefined categories (multi-label supported)
- **Category Extraction** (`extract`): Automatically discover and extract categories from abstracts when you don't have a predefined scheme
- **Category Exploration** (`explore`): Analyze category stability and saturation through repeated raw extraction
- **Summarization** (`summarize`): Summarize text or PDF documents using single or multiple models

cat-ademic also works with arbitrary text, images, and PDFs — not just OpenAlex papers. No manual downloading. Point it at a journal ISSN (or OpenAlex topic), set a date range, and get back a structured CSV.

Built on [cat-stack](https://github.com/chrissoria/cat-stack), which provides the core LLM classification engine across 8 providers.

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
  - [summarize()](#summarize)
  - [Discovery Functions](#discovery-functions)
- [Multi-Model Ensemble](#multi-model-ensemble)
- [Batch Mode](#batch-mode)
- [Related Projects](#related-projects)
- [Academic Research](#academic-research)
- [Contributing & Support](#contributing--support)
- [License](#license)

## Installation

```console
pip install cat-ademic
```

For PDF support:

```console
pip install "cat-ademic[pdf]"
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
results = cat.extract(
    journal_issn="0894-4393",
    paper_limit=250,
    date_from="2023-01-01",
    description="Academic papers from Social Science Computer Review",
    api_key=api_key,
)

print(results["top_categories"])
```

```python
# Raw category extraction for saturation analysis
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

```python
# Summarize text
results = cat.summarize(
    input_data=df["abstracts"],
    description="Academic paper abstracts",
    instructions="Summarize the key findings in 2-3 sentences",
    api_key=api_key,
)
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
- **Thinking/reasoning** (`thinking_budget`): Negligible accuracy gains (<1 pp) with significantly increased latency and cost.
- **Few-shot examples** (`example1`–`example6`): Degrades accuracy ~1 pp on average. Encourages over-classification (more false positives). Use verbose category definitions instead.

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

- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-5, o-series (o1, o3, o4), etc.
- **Anthropic**: Claude Sonnet 4, Claude 3.5 Sonnet, Claude Haiku, etc.
- **Google**: Gemini 2.5 Flash, Gemini 2.5 Pro, etc.
- **Huggingface**: Qwen, Llama 4, DeepSeek, and thousands of community models
- **xAI**: Grok models
- **Mistral**: Mistral Large, Mistral Small, etc.
- **Perplexity**: Sonar Large, Sonar Small, etc.
- **Ollama**: Any locally hosted model

The provider is auto-detected from the model name, or you can set it explicitly with `model_source=`.

**Note:** For best results, start with OpenAI or Anthropic.

-----

## API Reference

### `classify()`

Classify text, image, or PDF inputs into predefined categories. When you provide a journal/topic source, abstracts are fetched automatically from OpenAlex.

#### Academic source parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `journal_issn` | str | None | Journal ISSN to pull abstracts from via OpenAlex (e.g. `"0894-4393"`) |
| `journal_name` | str | None | Journal name — auto-resolved to ISSN via `find_journal()` |
| `journal_field` | str | None | Discipline name to fetch across all matching journals |
| `topic_name` | str | None | Filter papers by research topic (content-based) |
| `topic_id` | str | None | OpenAlex concept ID for exact topic match |
| `paper_limit` | int | 50 | Number of papers to fetch |
| `date_from` | str | None | Start date filter (`"YYYY-MM-DD"`) |
| `date_to` | str | None | End date filter (`"YYYY-MM-DD"`) |
| `polite_email` | str | None | Email for OpenAlex polite pool (higher rate limits) |

#### Academic context parameters

These are injected into the classification prompt to give the model context:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `journal` | str | None | Journal name (e.g. `"Nature Human Behaviour"`) |
| `field` | str | None | Academic field (e.g. `"computational social science"`) |
| `research_focus` | str | None | Research focus string |
| `paper_metadata` | dict | None | Additional key-value context injected into the prompt |

#### Core parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `categories` | list | *required* | List of category names/descriptions for classification |
| `input_data` | list/Series | None | Text, image paths, or PDF paths. Omit when using a journal source. |
| `api_key` | str | None | API key for the LLM provider (single-model mode) |
| `description` | str | `""` | Description of the corpus context |
| `user_model` | str | `"gpt-4o"` | Model to use |
| `model_source` | str | `"auto"` | Provider: `"auto"`, `"openai"`, `"anthropic"`, `"google"`, `"mistral"`, `"perplexity"`, `"huggingface"`, `"xai"`, `"ollama"` |
| `creativity` | float | None | Temperature (0.0–1.0). `None` = provider default. `0.0` is valid and recommended for classification. |
| `multi_label` | bool | True | If True, multiple categories can be assigned per item. If False, forces single-label. |
| `filename` | str | None | Output CSV filename |
| `save_directory` | str | None | Directory to save results |

#### Multi-model ensemble parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | list | None | List of `(model, provider, api_key)` tuples. Overrides `user_model`/`api_key`/`model_source`. |
| `consensus_threshold` | str/float | `"unanimous"` | Agreement threshold: `"unanimous"` (100%), `"majority"` (50%), `"two-thirds"` (67%), or a float 0–1 |

#### Batch mode parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_mode` | bool | False | Use async batch API (50% cost savings). Text input only. |
| `batch_poll_interval` | float | 30.0 | Seconds between batch job status checks |
| `batch_timeout` | float | 86400.0 | Max seconds to wait for batch completion (default 24h) |

#### Prompt strategy parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chain_of_thought` | bool | False | Step-by-step reasoning (no measurable accuracy gain) |
| `chain_of_verification` | bool | False | CoVe verification step (~4x cost, degrades accuracy ~2 pp) |
| `step_back_prompt` | bool | False | Step-back prompting (~2x cost, inconsistent gains) |
| `context_prompt` | bool | False | Add expert context prefix |
| `thinking_budget` | int | 0 | Provider-specific reasoning budget (Google/OpenAI/Anthropic) |
| `example1`–`example6` | str | None | Few-shot examples (degrades accuracy ~1 pp — use verbose categories instead) |

#### Advanced parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | `"image"` | PDF processing mode: `"image"`, `"text"`, or `"both"` |
| `pdf_dpi` | int | 150 | DPI for PDF page rendering |
| `use_json_schema` | bool | True | Use JSON schema for structured output |
| `safety` | bool | False | Save progress after each item |
| `max_workers` | int | None | Max parallel workers (None = auto) |
| `parallel` | bool | None | Force parallel (True) or sequential (False) processing |
| `fail_strategy` | str | `"partial"` | `"partial"` (keep successful rows) or `"strict"` (fail entirely) |
| `max_retries` | int | 5 | Max retries per API call |
| `batch_retries` | int | 2 | Max retries for batch-level failures |
| `retry_delay` | float | 1.0 | Delay between retries (seconds) |
| `row_delay` | float | 0.0 | Delay between rows (seconds) — useful for rate-limited providers |
| `auto_download` | bool | False | Auto-download Ollama models |
| `embeddings` | bool | False | Add embedding similarity scores per category |
| `embedding_tiebreaker` | bool | False | Use embedding centroids to break ensemble ties |
| `json_formatter` | bool | False | Use local fine-tuned model fallback for malformed JSON |
| `categories_per_call` | int | None | Split large category lists into chunks of this size |
| `progress_callback` | callable | None | Callback for progress updates: `callback(current, total, label)` |

**Returns:** `pandas.DataFrame` with one binary column per category (`category_1`, `category_2`, …), plus OpenAlex metadata columns (`title`, `doi`, `publication_date`, `cited_by_count`, `authors`, etc.) when using a journal source.

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

#### Academic source parameters

Same as [`classify()`](#academic-source-parameters): `journal_issn`, `journal_name`, `journal_field`, `topic_name`, `topic_id`, `paper_limit`, `date_from`, `date_to`, `polite_email`.

#### Academic context parameters

Same as [`classify()`](#academic-context-parameters): `journal`, `field`, `research_focus`, `paper_metadata`.

#### Core parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data` | list/Series | None | Text data. Omit when using a journal source. |
| `api_key` | str | *required* | API key for the LLM provider |
| `description` | str | `""` | Description of the corpus |
| `input_type` | str | `"text"` | Input type: `"text"`, `"image"`, or `"pdf"` |
| `max_categories` | int | 12 | Maximum number of final categories to return |
| `categories_per_chunk` | int | 10 | Categories to extract per chunk |
| `divisions` | int | 12 | Number of chunks to divide data into |
| `iterations` | int | 8 | Number of extraction passes over the data |
| `user_model` | str | `"gpt-4o"` | Model to use |
| `model_source` | str | `"auto"` | Provider |
| `creativity` | float | None | Temperature |
| `specificity` | str | `"broad"` | `"broad"` or `"specific"` category granularity |
| `research_question` | str | None | Research context to guide extraction |
| `focus` | str | None | Focus instruction (e.g. `"methodological contributions"`) |
| `mode` | str | `"text"` | PDF mode: `"text"`, `"image"`, or `"both"` |
| `filename` | str | None | Output CSV filename |
| `random_state` | int | None | Random seed for reproducibility |
| `chunk_delay` | float | 0.0 | Delay between API calls (seconds) |
| `progress_callback` | callable | None | Callback for progress updates |

**Returns:** `dict` with keys:
- `counts_df`: DataFrame of categories with counts
- `top_categories`: List of top category names
- `raw_top_text`: Raw model output from final merge step

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

#### Academic source parameters

Same as [`classify()`](#academic-source-parameters): `journal_issn`, `journal_name`, `journal_field`, `topic_name`, `topic_id`, `paper_limit`, `date_from`, `date_to`, `polite_email`.

#### Core parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data` | list/Series | None | Text data. Omit when using a journal source. |
| `api_key` | str | *required* | API key for the LLM provider |
| `description` | str | `""` | Description of the corpus |
| `max_categories` | int | 12 | Max categories per chunk |
| `categories_per_chunk` | int | 10 | Categories to extract per chunk |
| `divisions` | int | 12 | Number of chunks to divide data into |
| `iterations` | int | 8 | Number of passes over the data |
| `user_model` | str | `"gpt-4o"` | Model to use |
| `model_source` | str | `"auto"` | Provider |
| `creativity` | float | None | Temperature |
| `specificity` | str | `"broad"` | `"broad"` or `"specific"` category granularity |
| `research_question` | str | None | Research context |
| `focus` | str | None | Focus instruction for extraction |
| `random_state` | int | None | Random seed for reproducibility |
| `filename` | str | None | Output CSV filename (one category per row) |
| `chunk_delay` | float | 0.0 | Delay between API calls (seconds) |
| `progress_callback` | callable | None | Callback for progress updates |

**Returns:** `list[str]` — every category extracted from every chunk across every iteration. Length ≈ `iterations × divisions × categories_per_chunk`.

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

---

### `summarize()`

Summarize text or PDF documents using LLMs. Supports single-model and multi-model (ensemble) summarization. In multi-model mode, summaries from all models are synthesized into a consensus summary. Input type is auto-detected.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data` | list/Series/str | *required* | Text list, pandas Series, single string, PDF directory, or list of PDF paths |
| `api_key` | str | None | API key (single-model mode) |
| `description` | str | `""` | Description of the content (provides context) |
| `instructions` | str | `""` | Specific summarization instructions (e.g. `"bullet points"`) |
| `max_length` | int | None | Maximum summary length in words |
| `focus` | str | None | What to focus on (e.g. `"key findings"`, `"methodology"`) |
| `user_model` | str | `"gpt-4o"` | Model to use |
| `model_source` | str | `"auto"` | Provider |
| `mode` | str | `"image"` | PDF processing mode: `"image"`, `"text"`, or `"both"` |
| `pdf_dpi` | int | 150 | DPI for PDF rendering |
| `creativity` | float | None | Temperature |
| `thinking_budget` | int | 0 | Provider-specific reasoning budget |
| `chain_of_thought` | bool | True | Enable step-by-step reasoning |
| `context_prompt` | bool | False | Add expert context prefix |
| `step_back_prompt` | bool | False | Step-back prompting |
| `filename` | str | None | Output CSV filename |
| `save_directory` | str | None | Directory to save results |
| `progress_callback` | callable | None | Callback for progress updates |
| `models` | list | None | List of `(model, provider, api_key)` tuples for multi-model mode |
| `max_workers` | int | None | Max parallel workers |
| `safety` | bool | False | Save progress after each item |
| `max_retries` | int | 5 | Max retries per API call |
| `fail_strategy` | str | `"partial"` | `"partial"` or `"strict"` |
| `row_delay` | float | 0.0 | Delay between rows (seconds) |
| `batch_mode` | bool | False | Use async batch API (text input only) |
| `batch_poll_interval` | float | 30.0 | Seconds between batch status checks |
| `batch_timeout` | float | 86400.0 | Max seconds to wait for batch completion |

**Returns:** `pandas.DataFrame` with columns:
- `survey_input`: Original text or page label (PDFs)
- `summary`: Generated summary (or consensus for multi-model)
- `summary_<model>`: Per-model summaries (multi-model only)
- `processing_status`: `"success"`, `"error"`, or `"skipped"`
- `failed_models`: Comma-separated list (multi-model only)
- `pdf_path`, `page_index`: PDF metadata (PDF mode only)

**Example:**

```python
import catademic as cat

# Single model
results = cat.summarize(
    input_data=df["abstracts"],
    description="Academic paper abstracts",
    instructions="Summarize the key findings in 2-3 bullet points",
    max_length=100,
    api_key=api_key,
)

# Multi-model ensemble
results = cat.summarize(
    input_data=df["abstracts"],
    description="Academic paper abstracts",
    models=[
        ("gpt-4o", "openai", openai_key),
        ("claude-sonnet-4-5-20250929", "anthropic", anthropic_key),
    ],
)

# PDF summarization
results = cat.summarize(
    input_data="/path/to/papers/",
    description="Research papers",
    mode="text",
    api_key=api_key,
)
```

---

### Discovery Functions

These functions help you find journals and topics in OpenAlex before running classification or extraction.

#### `find_journal(name)`

Search OpenAlex for journals matching a name string.

```python
>>> cat.find_journal("socius")
                                      display_name       issn  works_count
0  Socius Sociological Research for a Dynamic World  2378-0231         1039
1                                     Jurnal Socius  2089-9661          315
```

#### `find_journals_by_field(field)`

Search for journals whose name or focus matches a field.

```python
>>> cat.find_journals_by_field("demography")
# Returns DataFrame with display_name, issn, works_count, publisher

# Then fetch across all demography journals:
>>> cat.classify(categories=[...], journal_field="demography", api_key=api_key)
```

#### `find_topic(name)`

Search for academic topics/concepts in OpenAlex.

```python
>>> cat.find_topic("machine learning")
        display_name  level  works_count
0  Machine learning      2      1234567
1    Deep learning       3       567890

# Then filter papers by topic:
>>> cat.classify(categories=[...], topic_name="machine learning", api_key=api_key)
```

#### `fetch_academic_papers()`

Fetch papers directly for custom analysis.

```python
papers = cat.fetch_academic_papers(
    journal_issn="0894-4393",
    limit=100,
    date_from="2023-01-01",
    polite_email="you@example.com",
)
# Returns DataFrame with: text, title, doi, publication_year, authors,
# cited_by_count, fwci, keywords, concepts, topic, ...
```

-----

## Multi-Model Ensemble

Use multiple models to classify the same data and aggregate their decisions via consensus voting. This reduces false positives and improves reliability.

```python
results = cat.classify(
    categories=["Empirical", "Theoretical", "Methodological", "Other"],
    journal_issn="0894-4393",
    paper_limit=50,
    models=[
        ("gpt-4o", "openai", openai_key),
        ("claude-sonnet-4-5-20250929", "anthropic", anthropic_key),
        ("gemini-2.5-flash", "google", google_key),
    ],
    consensus_threshold="unanimous",  # or "majority", "two-thirds", or 0.75
)
```

The result DataFrame includes:
- Per-model columns (e.g. `category_1_gpt_4o`, `category_1_claude_sonnet_...`)
- Consensus columns (`category_1_consensus`, `category_2_consensus`, …)
- `agreement_score`: Proportion of models that agreed

Thresholds:
- `"unanimous"` (default): All models must agree — highest precision, most conservative
- `"majority"`: >50% agreement
- `"two-thirds"`: >67% agreement
- Any float between 0 and 1

Per-model temperature can be set with 4-tuples:

```python
models=[
    ("gpt-4o", "openai", openai_key, {"creativity": 0.0}),
    ("gpt-4o", "openai", openai_key, {"creativity": 0.3}),
]
```

-----

## Batch Mode

For large classification jobs, batch mode uses the provider's async batch API — typically **50% cheaper** with higher rate limits. Supported by OpenAI, Anthropic, Google, Mistral, and xAI.

```python
results = cat.classify(
    categories=categories,
    journal_issn="0894-4393",
    paper_limit=500,
    api_key=api_key,
    batch_mode=True,
    batch_poll_interval=60,   # check every 60 seconds
    batch_timeout=86400,      # wait up to 24 hours
)
```

Limitations: text input only (no PDF/image), single model only (no ensemble), no `progress_callback`.

-----

## Related Projects

- **[cat-stack](https://github.com/chrissoria/cat-stack)**: The shared classification engine that powers cat-ademic.
- **[cat-llm](https://github.com/chrissoria/cat-llm)**: The full CatLLM ecosystem meta-package for text classification.
- **[cat-vader](https://github.com/chrissoria/cat-vader)**: Social media text classification with platform-aware context injection.

## Academic Research

If you use this package for research, please cite:

Soria, C. (2025). cat-ademic (0.1.0). GitHub. https://github.com/chrissoria/cat-ademic

## Contributing & Support

- **Report bugs or request features**: [Open a GitHub Issue](https://github.com/chrissoria/cat-ademic/issues)
- **Research collaboration**: Email [ChrisSoria@Berkeley.edu](mailto:ChrisSoria@Berkeley.edu)

## License

`cat-ademic` is distributed under the terms of the [GNU GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
