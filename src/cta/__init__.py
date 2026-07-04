"""Compositional Token Algebra (CTA): reversible, session-scoped span deduplication."""
from .algebra import compose, decompose, score_fn
from .detector import select_collapse_spans, build_segments, find_repeated_spans
from .learned_gate import GateMLP, gumbel_sigmoid

__all__ = [
    "compose", "decompose", "score_fn",
    "select_collapse_spans", "build_segments", "find_repeated_spans",
    "GateMLP", "gumbel_sigmoid",
]
__version__ = "0.1.0"
