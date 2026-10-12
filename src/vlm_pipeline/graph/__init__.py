"""Workflow model, pre-run validation, cache keys and the executor. Knows NodeSpec/NodeImpl, never concrete nodes."""

from vlm_pipeline.graph.validation import Issue, validate

__all__ = ["Issue", "validate"]
