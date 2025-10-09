"""
CsioParserPro package
---------------------
A small package for parsing and exploring CSIO EDI-style flat files.

Entrypoints:
- Streamlit UI: `python -m CsioParserPro` (or the installed package name) which launches Streamlit.
- Library usage: you can access parsing helpers via the package; attributes are loaded lazily.
"""
from __future__ import annotations

from typing import Any

# Public API names that will be lazily resolved from core.parser
__all__ = [
    "parse_lines",
    "enrich_by_code",
    "load_text",
    "save_outputs",
    "print_summary",
    "find_dates",
    "find_amounts",
    "find_vin",
    "find_year",
    "find_postal",
    "guess_coverage_code",
]

__version__ = "0.1.0"

# PEP 562: Lazy attribute access to avoid importing heavy deps (yaml/pandas) at package import time
def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in __all__:
        from .core import parser as _parser
        return getattr(_parser, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
