"""
Category exploration functions for CatAdemic.

This module provides raw category extraction from text inputs,
returning unprocessed category lists for frequency/saturation analysis.
"""

import pandas as pd

from ._academic import fetch_academic_papers, SUPPORTED_SOURCES

__all__ = [
    "explore",
]

from .text_functions import explore_common_categories


def explore(
    input_data=None,
    api_key=None,
    description="",
    # Academic source — when set, input_data is fetched automatically
    journal_issn: str = None,
    journal_name: str = None,
    journal_field: str = None,
    topic_name: str = None,
    topic_id: str = None,
    paper_limit: int = 50,
    date_from: str = None,
    date_to: str = None,
    polite_email: str = None,
    max_categories=12,
    categories_per_chunk=10,
    divisions=12,
    user_model="gpt-4o",
    creativity=None,
    specificity="broad",
    research_question=None,
    filename=None,
    model_source="auto",
    iterations=8,
    random_state=None,
    focus=None,
    progress_callback=None,
    chunk_delay: float = 0.0,
):
    """
    Explore categories in text data, returning the raw extracted list.

    Unlike extract(), which normalizes, deduplicates, and semantically merges
    categories, explore() returns every category string from every chunk across
    every iteration — with duplicates intact. This is useful for analyzing
    category stability and saturation across repeated extraction runs.

    Args:
        input_data: List of text responses or pandas Series.
            Omit when using journal_issn (abstracts are fetched automatically).
        api_key (str): API key for the model provider.
        journal_issn (str): Journal ISSN to pull abstracts from via OpenAlex.
            When set, input_data is fetched automatically.
        paper_limit (int): Number of papers to fetch. Default 50.
        date_from (str): Optional start date filter as "YYYY-MM-DD".
        date_to (str): Optional end date filter as "YYYY-MM-DD".
        description (str): The survey question or description of the data.
        max_categories (int): Maximum categories per chunk (passed through).
        categories_per_chunk (int): Categories to extract per chunk.
        divisions (int): Number of chunks to divide data into.
        user_model (str): Model name to use. Default "gpt-4o".
        creativity (float): Temperature setting. None uses model default.
        specificity (str): "broad" or "specific" category granularity.
        research_question (str): Optional research context.
        filename (str): Optional CSV filename to save raw category list.
        model_source (str): Provider - "auto", "openai", "anthropic", etc.
        iterations (int): Number of passes over the data.
        random_state (int): Random seed for reproducibility.
        focus (str): Optional focus instruction for category extraction.
        progress_callback (callable): Optional callback for progress updates.
        chunk_delay (float): Delay in seconds between API calls to avoid rate
            limits. Default 0.0 (no delay).

    Returns:
        list[str]: Every category string extracted from every chunk across
        every iteration. Length ≈ iterations × divisions × categories_per_chunk.

    Examples:
        >>> import catademic as cat
        >>>
        >>> raw_categories = cat.explore(
        ...     input_data=df['responses'],
        ...     description="Why did you move?",
        ...     api_key="your-api-key",
        ...     iterations=3,
        ...     divisions=5,
        ... )
        >>> print(len(raw_categories))  # ~150
        >>> print(raw_categories[:5])
    """
    # Early validation
    if api_key is None:
        raise ValueError(
            "[CatAdemic] api_key is required. Pass api_key='sk-...'."
        )

    # Fetch abstracts from OpenAlex when journal_issn is set
    if journal_issn is not None or journal_name is not None or journal_field is not None or topic_name is not None or topic_id is not None:
        if input_data is not None:
            raise ValueError("Pass either input_data or a journal/field source, not both.")
        _papers_df = fetch_academic_papers(
            journal_issn=journal_issn, journal_name=journal_name, journal_field=journal_field,
            topic_name=topic_name, topic_id=topic_id,
            limit=paper_limit, date_from=date_from, date_to=date_to,
            polite_email=polite_email,
        )
        input_data = _papers_df["text"].tolist()
        print(f"[CatAdemic] Fetched {len(input_data)} paper abstracts.")
    elif input_data is None:
        raise ValueError(
            "Provide either input_data, journal_issn, or journal_name."
        )

    raw_items = explore_common_categories(
        survey_input=input_data,
        api_key=api_key,
        survey_question=description,
        max_categories=max_categories,
        categories_per_chunk=categories_per_chunk,
        divisions=divisions,
        user_model=user_model,
        creativity=creativity,
        specificity=specificity,
        research_question=research_question,
        filename=None,  # We handle saving ourselves
        model_source=model_source,
        iterations=iterations,
        random_state=random_state,
        focus=focus,
        progress_callback=progress_callback,
        return_raw=True,
        chunk_delay=chunk_delay,
    )

    if filename:
        df = pd.DataFrame(raw_items, columns=["Category"])
        df.to_csv(filename, index=False)
        print(f"Raw categories saved to {filename}")

    return raw_items
