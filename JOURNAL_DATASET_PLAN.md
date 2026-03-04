# Plan: Empirical Journal Field Dataset

## Motivation

The current `journal_field` parameter in cat-ademic searches journal names via
OpenAlex's text search, which misses journals that cover a field without naming it.
For example, searching "demography" won't reliably return "Population Studies" or
"Demography and Social Economy."

The fix is to build an empirical field → journal mapping by pulling the full OpenAlex
journal catalogue, LLM-classifying each journal's field, and publishing the result as
a reusable dataset.

---

## Step 1 — Pull the full journal catalogue

Script: `social_science_computer_review/pull_all_journals.py`

- **Filter**: `type:journal` (no impact factor cutoff — pull all ~250k)
- **Fields**: display_name, abbreviated_title, issn, all_issns, publisher,
  country_code, works_count, oa_works_count, cited_by_count,
  first_publication_year, last_publication_year, impact_factor, h_index, i10_index,
  is_oa, is_in_doaj, is_high_oa_rate, is_indexed_in_scopus, is_core, apc_usd,
  primary_field, primary_subfield, primary_domain, all_fields
- **Output**: `out/all_journals.csv`

The current script fetches ~17k journals (impact_factor > 1). For the full dataset,
remove the `summary_stats.2yr_mean_citedness` filter.

---

## Step 2 — LLM-classify journals by field

Use cat-ademic itself to classify all 250k journals into a standardized field
taxonomy (e.g. OpenAlex's own domain/field/subfield hierarchy, or a custom one).

Input per journal: `display_name`, `primary_field`, `all_fields`, `primary_subfield`

Output: a `field_label` column — normalized, human-readable field name that a
researcher would use (e.g. "Demography", "Computational Social Science", "Ecology").

Consider:
- Running in batch mode (Anthropic/OpenAI batch API) to reduce cost
- Using the `primary_field` from OpenAlex topics as the ground truth where confident,
  and only sending ambiguous/missing cases to the LLM
- Handling multidisciplinary journals with a comma-separated `field_labels` column

---

## Step 3 — Publish as a HuggingFace dataset

Publish `all_journals.csv` (with LLM-derived field labels) as a public HuggingFace
dataset under `chrissoria/openalex-journals`.

Dataset card should describe:
- Source: OpenAlex (open catalogue, CC0)
- How field labels were derived
- Column definitions
- How to use it with cat-ademic

This lets the research community use it independently of cat-ademic.

---

## Step 4 — Bundle a filtered version into cat-ademic

For the `journal_field` parameter, ship a lightweight lookup table inside the package:

```
src/catademic/data/field_journal_map.parquet
```

Contents: `field_label` (normalized), `issn` (linking ISSN), `display_name`

Filtered to journals with `impact_factor > 1` to keep file size manageable (~17k rows).

Replace the name-search in `find_journals_by_field()` with a local lookup:

```python
import importlib.resources
import pandas as pd

def find_journals_by_field(field: str, limit: int = 20) -> pd.DataFrame:
    with importlib.resources.open_binary("catademic.data", "field_journal_map.parquet") as f:
        df = pd.read_parquet(f)
    mask = df["field_label"].str.contains(field, case=False, na=False)
    return df[mask].head(limit)
```

Add a `refresh=True` parameter to re-fetch from the HuggingFace dataset if the user
wants the latest version.

---

## Step 5 — Wire `journal_field` into `fetch_academic_papers()`

Currently `journal_field` does a name search and fetches the top-matching journal.
After bundling the lookup table, it should:

1. Look up all journals matching the field label in `field_journal_map.parquet`
2. Build an OR filter across their ISSNs: `primary_location.source.issn:issn1|issn2|...`
3. Fetch works from all matching journals in one OpenAlex query

This gives true cross-journal field coverage.

---

## File Locations

| File | Purpose |
|---|---|
| `social_science_computer_review/pull_all_journals.py` | Fetch full catalogue |
| `social_science_computer_review/out/all_journals.csv` | Raw journal data |
| `src/catademic/_academic.py` | `find_journals_by_field()` — replace name search |
| `src/catademic/data/field_journal_map.parquet` | Bundled lookup table (to create) |
| `JOURNAL_DATASET_PLAN.md` | This file |

---

## Cost Estimate (rough)

- 250k journals × ~50 tokens input = 12.5M tokens
- At Anthropic batch rates (~$0.75/1M input tokens for Haiku): ~$9
- Output tokens negligible (just a field label per journal)
