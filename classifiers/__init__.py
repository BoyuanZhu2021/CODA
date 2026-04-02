"""Classifier models for fake video detection."""

from .models import (
    EmbeddingClassifier,
    MLPClassifier,
    TransformerClassifier,
    get_embeddings
)

__all__ = [
    'EmbeddingClassifier',
    'MLPClassifier', 
    'TransformerClassifier',
    'get_embeddings'
]

