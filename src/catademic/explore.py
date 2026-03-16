"""
Category exploration functions for CatAdemic.

Thin wrapper around cat_stack.explore() that adds academic-specific features:
- OpenAlex paper fetching (journal_issn, journal_name, journal_field, topic_name/id)
"""

import cat_stack

from ._academic import fetch_academic_papers, SUPPORTED_SOURCES

__all__ = [
    "explore",
]


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
    **kwargs,
):
    """
    Explore categories in text data, returning the raw extracted list.

    Unlike extract(), which normalizes, deduplicates, and semantically merges
    categories, explore() returns every category string from every chunk across
    every iteration — with duplicates intact. Useful for saturation analysis.

    Wraps cat_stack.explore() and adds OpenAlex paper fetching.

    Args:
        input_data: List of text responses or pandas Series.
            Omit when using journal_issn (abstracts are fetched automatically).
        api_key (str): API key for the model provider.
        description (str): The survey question or description of the data.
        journal_issn (str): Journal ISSN to pull abstracts from via OpenAlex.
        journal_name (str): Journal name to search for (resolved to ISSN).
        journal_field (str): Discipline name to pull papers from matching journals.
        topic_name (str): Filter papers by content topic.
        topic_id (str): OpenAlex concept ID for exact topic match.
        paper_limit (int): Number of papers to fetch. Default 50.
        date_from (str): Optional start date filter as "YYYY-MM-DD".
        date_to (str): Optional end date filter as "YYYY-MM-DD".
        polite_email (str): Optional email for OpenAlex polite pool.
        **kwargs: All other parameters passed through to cat_stack.explore()
            (e.g. max_categories, categories_per_chunk, divisions, user_model,
            creativity, specificity, iterations, focus, filename, etc.)

    Returns:
        list[str]: Every category string extracted from every chunk across
        every iteration. Length ~ iterations x divisions x categories_per_chunk.

    Examples:
        >>> import catademic as cat
        >>>
        >>> raw_categories = cat.explore(
        ...     journal_issn="0894-4393",
        ...     paper_limit=50,
        ...     description="Academic papers",
        ...     api_key="your-api-key",
        ...     iterations=3,
        ...     divisions=5,
        ... )
        >>> print(len(raw_categories))
    """
    # Early validation
    if api_key is None:
        raise ValueError(
            "[CatAdemic] api_key is required. Pass api_key='sk-...'."
        )

    # Fetch abstracts from OpenAlex when an academic source is set
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

    return cat_stack.explore(
        input_data=input_data,
        api_key=api_key,
        description=description,
        **kwargs,
    )
