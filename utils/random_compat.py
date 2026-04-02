"""Compatibility helper for RNG objects.

Both legacy ``RandomState`` (which provides ``rand()``) and modern
``Generator`` (which provides ``random()``) are supported.
"""

from __future__ import annotations


def rng_random(rng, size=None):
	"""Draw uniform [0, 1) samples from *rng*, handling both API styles."""
	if rng is None:
		return None
	if hasattr(rng, "random"):
		return rng.random(size) if size is not None else rng.random()
	if hasattr(rng, "rand"):
		return rng.rand(*size) if size is not None else rng.rand()
	raise AttributeError("RNG must provide random() or rand()")
