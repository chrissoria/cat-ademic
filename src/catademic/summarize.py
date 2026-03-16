"""
Summarization functions for CatAdemic.

Thin wrapper around cat_stack.summarize() — passes through all parameters.
"""

import cat_stack

__all__ = [
    "summarize",
]


def summarize(input_data, **kwargs):
    """
    Summarize text or PDF data using LLMs.

    Passes through directly to cat_stack.summarize(). Supports single-model
    and multi-model (ensemble) summarization with auto-detected input type.

    Args:
        input_data: Data to summarize. Can be:
            - Text: list of strings, pandas Series, or single string
            - PDF: directory path, single PDF path, or list of PDF paths
        **kwargs: All parameters passed through to cat_stack.summarize()
            (e.g. api_key, description, instructions, max_length, focus,
            user_model, models, mode, creativity, batch_mode, etc.)

    Returns:
        pd.DataFrame: Results with summary column(s).

    Examples:
        >>> import catademic as cat
        >>>
        >>> results = cat.summarize(
        ...     input_data=df['abstracts'],
        ...     description="Academic paper abstracts",
        ...     api_key="your-api-key",
        ... )
    """
    return cat_stack.summarize(
        input_data=input_data,
        **kwargs,
    )
