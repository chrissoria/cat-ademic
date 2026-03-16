"""
Category extraction functions for CatAdemic.

Thin wrapper around cat_stack.extract() that adds academic-specific features:
- OpenAlex paper fetching (journal_issn, journal_name, journal_field, topic_name/id)
- Academic context injection (journal, field, research_focus, paper_metadata)
"""

import cat_stack

from ._academic import fetch_academic_papers, SUPPORTED_SOURCES

__all__ = [
    "extract",
]


def _build_academic_context(journal, field, research_focus, paper_metadata):
    """Build a context block string from academic paper metadata fields."""
    parts = []
    if journal:
        parts.append(f"Journal: {journal}")
    if field:
        parts.append(f"Field: {field}")
    if research_focus:
        parts.append(f"Research focus: {research_focus}")
    if paper_metadata:
        for k, v in paper_metadata.items():
            parts.append(f"{k.capitalize()}: {v}")
    return "\n".join(parts)


def extract(
    input_data=None,
    api_key=None,
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
    # Academic context fields — injected into the extraction prompt
    journal: str = None,
    field: str = None,
    research_focus: str = None,
    paper_metadata: dict = None,
    description="",
    **kwargs,
):
    """
    Extract/discover categories from text, image, PDF, or academic inputs.

    Wraps cat_stack.extract() and adds:
    - OpenAlex paper fetching via journal_issn/journal_name/journal_field/topic_name
    - Academic context injection into the extraction prompt

    Args:
        input_data: The data to explore. Can be:
            - For text: list of text responses or pandas Series
            - For image: directory path, single file, or list of image paths
            - For pdf: directory path, single file, or list of PDF paths
            - Omit when using journal_issn (abstracts are fetched automatically).
        api_key (str): API key for the model provider.
        journal_issn (str): Journal ISSN to pull abstracts from via OpenAlex.
        journal_name (str): Journal name to search for (resolved to ISSN).
        journal_field (str): Discipline name to pull papers from matching journals.
        topic_name (str): Filter papers by content topic.
        topic_id (str): OpenAlex concept ID for exact topic match.
        paper_limit (int): Number of papers to fetch. Default 50.
        date_from (str): Optional start date filter as "YYYY-MM-DD".
        date_to (str): Optional end date filter as "YYYY-MM-DD".
        polite_email (str): Optional email for OpenAlex polite pool.
        journal (str): Journal name — injected into the prompt as context.
        field (str): Academic field/discipline.
        research_focus (str): Optional research focus string.
        paper_metadata (dict): Additional context injected into the prompt.
        description (str): Description of the input data.
        **kwargs: All other parameters passed through to cat_stack.extract()
            (e.g. input_type, max_categories, categories_per_chunk, divisions,
            user_model, creativity, specificity, iterations, focus, etc.)

    Returns:
        dict with keys:
            - counts_df: DataFrame of categories with counts
            - top_categories: List of top category names
            - raw_top_text: Raw model output from final merge step

    Examples:
        >>> import catademic as cat
        >>>
        >>> # Extract categories from journal abstracts
        >>> results = cat.extract(
        ...     journal_issn="0894-4393",
        ...     paper_limit=50,
        ...     description="Academic papers from Social Science Computer Review",
        ...     api_key="your-api-key",
        ... )
        >>> print(results['top_categories'])
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

    # Prepend academic context to description if any fields provided
    academic_context = _build_academic_context(journal, field, research_focus, paper_metadata)
    if academic_context:
        description = f"{academic_context}\n{description}".strip() if description else academic_context

    return cat_stack.extract(
        input_data=input_data,
        api_key=api_key,
        description=description,
        **kwargs,
    )
