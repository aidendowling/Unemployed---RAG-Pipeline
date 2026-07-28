"""Knowledge-obsolescence scoring for retrieved documents."""

from .freshness import (
    FreshnessScorer,
    freshness_score,
    is_obsolete,
    query_intent_vintage_score,
)

__all__ = [
    "FreshnessScorer",
    "freshness_score",
    "is_obsolete",
    "query_intent_vintage_score",
]
