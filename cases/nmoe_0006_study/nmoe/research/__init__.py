"""nmoe research tools.

Usage:
    from nmoe.research import lab
"""

import importlib

__all__: list[str] = ['lab']


def __getattr__(name: str):
    """Lazy import for lab module."""
    if name == 'lab':
        return importlib.import_module('.lab', __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
