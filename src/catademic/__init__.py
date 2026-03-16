# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: MIT

from .__about__ import (
    __version__,
    __author__,
    __description__,
    __title__,
    __url__,
    __license__,
)

# =============================================================================
# Public API — catademic entry points (thin wrappers around cat_stack)
# =============================================================================
from .classify import classify
from .extract import extract
from .explore import explore
from .summarize import summarize

# =============================================================================
# Academic data source (catademic-specific)
# =============================================================================
from ._academic import fetch_academic_papers, find_journal, find_journals_by_field, find_topic, SUPPORTED_SOURCES

# =============================================================================
# Re-exports from cat_stack (backward compatibility + provider utilities)
# =============================================================================
from cat_stack import (
    # Category analysis
    has_other_category,
    check_category_verbosity,
    # Batch exceptions
    BatchJobExpiredError,
    BatchJobFailedError,
    # Provider utilities
    UnifiedLLMClient,
    detect_provider,
    set_ollama_endpoint,
    check_ollama_running,
    list_ollama_models,
    check_ollama_model,
    pull_ollama_model,
    PROVIDER_CONFIG,
    # Deprecated backward-compat functions
    explore_common_categories,
    explore_corpus,
    explore_image_categories,
    explore_pdf_categories,
    classify_ensemble,
    multi_class,
    image_multi_class,
    pdf_multi_class,
    summarize_ensemble,
    # Utilities
    build_json_schema,
    extract_json,
    validate_classification_json,
    image_score_drawing,
    image_features,
)

# Define public API
__all__ = [
    # Main entry points (catademic wrappers)
    "classify",
    "extract",
    "explore",
    "summarize",
    # Academic data source
    "fetch_academic_papers",
    "find_journal",
    "find_journals_by_field",
    "find_topic",
    "SUPPORTED_SOURCES",
    # Category analysis (from cat_stack)
    "has_other_category",
    "check_category_verbosity",
    # Batch exceptions (from cat_stack)
    "BatchJobExpiredError",
    "BatchJobFailedError",
    # Provider utilities (from cat_stack)
    "UnifiedLLMClient",
    "detect_provider",
    "set_ollama_endpoint",
    "check_ollama_running",
    "list_ollama_models",
    "check_ollama_model",
    "pull_ollama_model",
    "PROVIDER_CONFIG",
    # Deprecated backward-compat (from cat_stack)
    "explore_common_categories",
    "explore_corpus",
    "explore_image_categories",
    "explore_pdf_categories",
    "classify_ensemble",
    "summarize_ensemble",
    "multi_class",
    "image_multi_class",
    "pdf_multi_class",
    "image_score_drawing",
    "image_features",
    "build_json_schema",
    "extract_json",
    "validate_classification_json",
]
