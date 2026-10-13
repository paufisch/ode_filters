"""Square-root Gaussian objects and algebra."""

from .algebra import compose_backward_conditionals, sqr_inversion, sqr_marginalization
from .types import AffineModel, Conditional, Gaussian

__all__ = [
    "AffineModel",
    "Conditional",
    "Gaussian",
    "compose_backward_conditionals",
    "sqr_inversion",
    "sqr_marginalization",
]
