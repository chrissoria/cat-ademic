"""
Category extraction functions for CatAdemic.

This module provides unified category extraction from text, image, and PDF inputs,
with built-in support for fetching academic paper abstracts from OpenAlex.
"""

import warnings

from ._academic import fetch_academic_papers, SUPPORTED_SOURCES

__all__ = [
    # Main entry point
    "extract",
    # Input-specific functions (for backward compatibility)
    "explore_common_categories",
    "explore_corpus",
    "explore_image_categories",
    "explore_pdf_categories",
]

# Import provider infrastructure
from ._providers import (
    UnifiedLLMClient,
    detect_provider,
)

# Import the implementation functions from existing modules
from .text_functions import (
    explore_common_categories,
    explore_corpus,
)

from .image_functions import (
    explore_image_categories,
)

from .pdf_functions import (
    explore_pdf_categories,
)


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
    input_type="text",
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
    max_categories=12,
    categories_per_chunk=10,
    divisions=12,
    user_model="gpt-4o",
    creativity=None,
    specificity="broad",
    research_question=None,
    mode="text",
    filename=None,
    model_source="auto",
    iterations=8,
    random_state=None,
    focus=None,
    progress_callback=None,
    chunk_delay: float = 0.0,
):
    """
    Unified category extraction function for text, image, PDF, and academic inputs.

    This function dispatches to the appropriate specialized explore function
    based on the `input_type` parameter, providing a single entry point for
    discovering categories in your data.

    Args:
        input_data: The data to explore. Can be:
            - For text: list of text responses or pandas Series
            - For image: directory path, single file, or list of image paths
            - For pdf: directory path, single file, or list of PDF paths
            - Omit when using journal_issn (abstracts are fetched automatically).
        api_key (str): API key for the model provider.
        input_type (str): Type of input data. Options:
            - "text" (default): Text/survey responses
            - "image": Image files
            - "pdf": PDF documents
        journal_issn (str): Journal ISSN to pull abstracts from via OpenAlex
            (e.g. "0894-4393"). When set, input_data is fetched automatically.
        paper_limit (int): Number of papers to fetch. Default 50.
        date_from (str): Optional start date filter as "YYYY-MM-DD".
        date_to (str): Optional end date filter as "YYYY-MM-DD".
        journal (str): Journal name — injected into the extraction prompt as context.
        field (str): Academic field/discipline (e.g. "computational social science").
        research_focus (str): Optional research focus string (e.g. "survey methods").
        paper_metadata (dict): Additional context injected into the prompt
            (e.g. {"cited_by_count": "varies", "keywords": "see abstracts"}).
        description (str): Description of the input data. Used as:
            - survey_question for text
            - image_description for images
            - pdf_description for PDFs
        max_categories (int): Maximum number of final categories to return.
        categories_per_chunk (int): Categories to extract per chunk.
        divisions (int): Number of chunks to divide data into.
        user_model (str): Model name to use. Default "gpt-4o".
        creativity (float): Temperature setting. None uses model default.
        specificity (str): "broad" or "specific" category granularity.
        research_question (str): Optional research context.
        mode (str): Processing mode:
            - For text: Not used
            - For image: "image" (default) or "both"
            - For pdf: "text" (default), "image", or "both"
        filename (str): Optional CSV filename to save results.
        model_source (str): Provider - "auto", "openai", "anthropic", "google",
            "mistral", "huggingface", "xai".
        iterations (int): Number of passes over the data.
        random_state (int): Random seed for reproducibility.
        focus (str): Optional focus instruction for category extraction (e.g.,
            "decisions to move", "emotional responses"). When provided, the model
            will prioritize extracting categories related to this focus.
        progress_callback (callable): Optional callback function for progress updates.
            Called as progress_callback(current_step, total_steps, step_label).
        chunk_delay (float): Delay in seconds between API calls to avoid rate
            limits. Default 0.0 (no delay).

    Returns:
        dict with keys:
            - counts_df: DataFrame of categories with counts
            - top_categories: List of top category names
            - raw_top_text: Raw model output from final merge step

    Examples:
        >>> import catademic as cat
        >>>
        >>> # Extract categories from OpenAlex journal abstracts
        >>> results = cat.extract(
        ...     journal_issn="0894-4393",
        ...     paper_limit=50,
        ...     description="Academic papers from Social Science Computer Review",
        ...     api_key="your-api-key"
        ... )
        >>> print(results['top_categories'])
        >>>
        >>> # Extract categories from text responses
        >>> results = cat.extract(
        ...     input_data=df['responses'],
        ...     description="Why did you move?",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> # Extract categories from PDFs
        >>> results = cat.extract(
        ...     input_data="/path/to/pdfs/",
        ...     description="Research papers",
        ...     input_type="pdf",
        ...     mode="text",
        ...     api_key="your-api-key"
        ... )
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

    # Prepend academic context to description if any fields provided
    academic_context = _build_academic_context(journal, field, research_focus, paper_metadata)
    if academic_context:
        description = f"{academic_context}\n{description}".strip() if description else academic_context

    input_type = input_type.lower().rstrip('s')  # Normalize: "texts" -> "text", "images" -> "image", "pdfs" -> "pdf"

    if input_type == "text":
        return explore_common_categories(
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
            filename=filename,
            model_source=model_source,
            iterations=iterations,
            random_state=random_state,
            focus=focus,
            progress_callback=progress_callback,
            chunk_delay=chunk_delay,
        )

    elif input_type == "image":
        return explore_image_categories(
            image_input=input_data,
            api_key=api_key,
            image_description=description,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions,
            user_model=user_model,
            creativity=creativity,
            specificity=specificity,
            research_question=research_question,
            mode=mode if mode in ["image", "both"] else "image",
            filename=filename,
            model_source=model_source,
            iterations=iterations,
            random_state=random_state,
            progress_callback=progress_callback,
        )

    elif input_type == "pdf":
        return explore_pdf_categories(
            pdf_input=input_data,
            api_key=api_key,
            pdf_description=description,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions,
            user_model=user_model,
            creativity=creativity,
            specificity=specificity,
            research_question=research_question,
            mode=mode if mode in ["text", "image", "both"] else "text",
            filename=filename,
            model_source=model_source,
            iterations=iterations,
            random_state=random_state,
            progress_callback=progress_callback,
        )

    else:
        raise ValueError(
            f"input_type '{input_type}' is not supported. "
            f"Please use one of: 'text', 'image', or 'pdf'.\n\n"
            f"Examples:\n"
            f"  - For survey responses or text data: input_type='text'\n"
            f"  - For image files (.jpg, .png, etc.): input_type='image'\n"
            f"  - For PDF documents: input_type='pdf'"
        )
