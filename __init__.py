"""
csio_explorer package
---------------------
A small package for parsing and exploring CSIO EDI-style flat files.

Entrypoints:
- Streamlit UI: `python -m csio_explorer` or `streamlit run -m csio_explorer.app`.
- Library usage: import parsing helpers from `csio_explorer.parser`.
"""

from .core.parser import (
    parse_lines,
    enrich_by_code,
    load_text,
    save_outputs,
    print_summary,
    find_dates,
    find_amounts,
    find_vin,
    find_year,
    find_postal,
    guess_coverage_code,
)

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
