#!/usr/bin/env python3
"""
csio_explorer.core.tokenizer
----------------------------
Header/length-aware tokenizer for CSIO flat file blobs.
This module scans the entire string (no newline dependence) and yields
records with level, code, length, and body.
"""
from __future__ import annotations

import re
from typing import Iterator, Dict, Any, List

HEADER_RE = re.compile(r"(\d)([A-Z]{3})(\d{3})")


def tokenize_records(raw: str) -> Iterator[Dict[str, Any]]:
    """Yield dicts {level, record_code, record_len, body} by scanning the blob."""
    i, n = 0, len(raw)
    while i < n:
        m = HEADER_RE.search(raw, i)
        if not m:
            break
        level, code, rlen = m.group(1), m.group(2), int(m.group(3))
        body_start = m.end()
        body_end = body_start + rlen
        if body_end > n:
            body_end = min(n, body_start + rlen)
        body = raw[body_start:body_end]
        # Clean payload by removing '?' runs and trimming whitespace as per UI requirements
        try:
            from .csio_utils import clean_payload as _clean_payload
            body = _clean_payload(body)
        except Exception:
            body = body.strip()
        yield {"level": level, "record_code": code, "record_len": f"{rlen:03d}", "body": body}
        i = body_end


def parse_lines(lines: List[str]):
    """Parse by scanning the entire content as a single string; no newline dependency."""
    if not lines:
        import pandas as pd
        return pd.DataFrame([])
    # Join without inserting extra characters so byte positions remain correct
    raw = "".join([ln.rstrip("\r\n") for ln in lines])
    rows: List[Dict[str, Any]] = []
    for rec in tokenize_records(raw):
        rows.append(rec)
    import pandas as pd
    return pd.DataFrame(rows)
