"""
Classification functions for CatAdemic.

Thin wrapper around cat_stack.classify() that adds academic-specific features:
- OpenAlex paper fetching (journal_issn, journal_name, journal_field, topic_name/id)
- Academic context injection (journal, field, research_focus, paper_metadata)
- Post-classification metadata attachment (paper metadata columns)
"""

import cat_stack

from ._academic import fetch_academic_papers, SUPPORTED_SOURCES

__all__ = [
    "classify",
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


def classify(
    categories,
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
    # Academic context fields — injected into the classification prompt
    journal: str = None,
    field: str = None,
    research_focus: str = None,
    paper_metadata: dict = None,
    description="",
    filename=None,
    save_directory=None,
    **kwargs,
):
    """
    Classify text, image, or PDF inputs with academic-specific features.

    Wraps cat_stack.classify() and adds:
    - OpenAlex paper fetching via journal_issn/journal_name/journal_field/topic_name
    - Academic context injection into the classification prompt
    - Post-classification attachment of paper metadata columns

    Args:
        categories (list): List of category names for classification.
        input_data: The data to classify. Can be:
            - For text: list of text responses or pandas Series
            - For image: directory path or list of image file paths
            - For pdf: directory path or list of PDF file paths
            - Omit when using journal_issn (abstracts are fetched automatically).
        api_key (str): API key for the model provider (single-model mode).
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
        description (str): Description of the input data context.
        filename (str): Output filename for CSV.
        save_directory (str): Directory to save results.
        **kwargs: All other parameters passed through to cat_stack.classify()
            (e.g. user_model, models, creativity, batch_mode, consensus_threshold,
            chain_of_thought, thinking_budget, embeddings, etc.)

    Returns:
        pd.DataFrame: Results with classification columns.

    Examples:
        >>> import catademic as cat
        >>>
        >>> # Classify papers from a journal
        >>> results = cat.classify(
        ...     categories=["Quantitative", "Qualitative", "Mixed Methods"],
        ...     journal_issn="0894-4393",
        ...     paper_limit=50,
        ...     api_key="your-api-key",
        ... )
        >>>
        >>> # Classify with academic context
        >>> results = cat.classify(
        ...     input_data=df['abstracts'],
        ...     categories=["Empirical", "Theoretical", "Review"],
        ...     journal="Social Science Computer Review",
        ...     field="computational social science",
        ...     api_key="your-api-key",
        ... )
    """
    # Early validation
    if api_key is None and kwargs.get("models") is None:
        raise ValueError(
            "[CatAdemic] api_key is required. Pass api_key='sk-...' or use the "
            "models= parameter for multi-model mode."
        )

    # Fetch abstracts from OpenAlex when an academic source is set
    _papers_df = None
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

    # When using academic source, suppress internal save so we can attach metadata first
    _academic = _papers_df is not None
    result = cat_stack.classify(
        input_data=input_data,
        categories=categories,
        api_key=api_key,
        description=description,
        add_other=False,
        check_verbosity=False,
        filename=None if _academic else filename,
        save_directory=None if _academic else save_directory,
        **kwargs,
    )

    # Attach paper metadata and rename survey_input → abstract
    if _academic:
        result = result.reset_index(drop=True)
        if "survey_input" in result.columns:
            result = result.rename(columns={"survey_input": "abstract"})
        meta_cols = [c for c in _papers_df.columns if c != "text"]
        for col in meta_cols:
            result[col] = _papers_df[col].reset_index(drop=True)
        # Save with enriched columns if filename was requested
        if filename or save_directory:
            import os as _os
            out = filename
            if save_directory and filename:
                out = _os.path.join(save_directory, _os.path.basename(filename))
            elif save_directory:
                out = _os.path.join(save_directory, "results.csv")
            if out:
                result.to_csv(out, index=False)
                print(f"Combined results saved to {out}")

    return result
