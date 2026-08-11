"""Minimal tracing helpers (kept for optional shape asserts)."""

try:
    from torch import _assert
except ImportError:

    def _assert(condition: bool, message: str):
        assert condition, message
