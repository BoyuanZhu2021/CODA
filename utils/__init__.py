"""Utility functions for fake video detection."""

from .text_processing import (
    clean_text,
    detect_language,
    get_language_name,
    extract_text_content,
    detect_content_language,
    is_debunking_content,
    extract_claims_text
)

from .logger import (
    setup_logger,
    get_main_logger,
    get_training_logger,
    get_llm_logger,
    get_pipeline_logger,
    TrainingLogger,
    LLMDecisionLogger,
    PipelineLogger,
    log_info,
    log_error
)

__all__ = [
    'clean_text',
    'detect_language',
    'get_language_name',
    'extract_text_content',
    'detect_content_language',
    'is_debunking_content',
    'extract_claims_text',
    'setup_logger',
    'get_main_logger',
    'get_training_logger',
    'get_llm_logger',
    'get_pipeline_logger',
    'TrainingLogger',
    'LLMDecisionLogger',
    'PipelineLogger',
    'log_info',
    'log_error'
]

