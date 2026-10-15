"""Square-root Gaussian objects and algebra."""

from .algebra import compose, invert, log_likelihood, marginalize, update
from .types import AffineModel, Conditional, Gaussian

__all__ = [
    "AffineModel",
    "Conditional",
    "Gaussian",
    "compose",
    "invert",
    "log_likelihood",
    "marginalize",
    "update",
]
