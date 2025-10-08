#!/usr/bin/env python3
"""
csio_explorer.core.csio_utils
-----------------------------
Utility functions and heuristics shared across the CSIO Parser Pro app.
"""
from __future__ import annotations

import re
from typing import Dict, Any, Tuple

# --- Lightweight text finders / heuristics ---
VIN_RE = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
POSTAL_RE = re.compile(r"\b([ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]\s?\d[ABCEGHJ-NPRSTV-Z]\d)\b", re.I)
ISO_DATE_RE = re.compile(r"\b(20\d{2}|19\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b")
AMOUNT_RE = re.compile(r"(?:\$?\s*)(\d{1,3}(?:,\d{3})*(?:\.\d{2})|\d+\.\d{2})")

# Right-edge money blocks (digits= cents, trailing sign)
MONEY_BLOCK = re.compile(r"(?<!\d)(\d{9,12})([+-])")
DATE8_RE = re.compile(r"\b(20\d{2}|19\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b")


def guess_coverage_code(text: str) -> str | None:
    tokens = re.findall(r"\b[A-Z0-9]{2,6}\b", text or "")
    for t in tokens[:6]:
        if not t.isdigit():
            return t
    return tokens[0] if tokens else None


def find_postal(text: str) -> str | None:
    m = POSTAL_RE.search(text or "")
    return m.group(1).upper() if m else None


def find_dates(text: str) -> str | None:
    dates = ISO_DATE_RE.findall(text or "")
    out = [f"{y}-{m}-{d}" for (y, m, d) in dates]
    return ", ".join(out) if out else None


def find_amounts(text: str) -> str | None:
    amts = AMOUNT_RE.findall(text or "")
    return ", ".join(amts) if amts else None


def find_vin(text: str) -> str | None:
    m = VIN_RE.search(text or "")
    return m.group(1) if m else None


def find_year(text: str) -> str | None:
    # Find plausible vehicle/policy years 1990–2029 and return the last occurrence
    yrs = re.findall(r"\b(199\d|20[0-2]\d)\b", text or "")
    return yrs[-1] if yrs else None


def find_make_model(text: str) -> tuple[str, str]:
    text = text or ""
    yr = find_year(text)
    if not yr:
        return ("", "")
    vin_m = VIN_RE.search(text)
    vin_pos = vin_m.start() if vin_m else len(text)
    # take subsection after year, up to VIN or end
    idx = text.find(yr)
    tail = text[idx + len(yr) : vin_pos]
    # Split into uppercase tokens
    toks = re.findall(r"\b[A-Z0-9][A-Z0-9-]+\b", tail)
    if not toks:
        return ("", "")
    make = toks[0]
    model = " ".join(toks[1:]) if len(toks) > 1 else ""
    return (make, model)


# --- Normalizers ---

# Generic trailing-sign amount token: 8–12 digits followed by +/-
TRAILING_SIGN_AMOUNT_RE = re.compile(r"\b(\d{8,12})([+-])\b")


def decode_trailing_amount(digits: str, sign: str) -> str:
    """Decode cents with trailing sign into dollars string with two decimals.
    Example: '00000001600', '+' -> '16.00' ; '00000003400','-' -> '-34.00'
    """
    try:
        cents = int(digits)
        val = cents / 100.0
        if sign == '-':
            val = -val
        return f"{val:.2f}"
    except Exception:
        return ""


def extract_trailing_amounts(text: str) -> list[tuple[str, str, str]]:
    """Scan left-to-right and collect all trailing-sign amount tokens.
    Returns list of tuples: (raw_token, digits+sign, decoded_dollars)
    """
    out: list[tuple[str, str, str]] = []
    for m in TRAILING_SIGN_AMOUNT_RE.finditer(text or ""):
        digits, sign = m.groups()
        raw_tok = m.group(0)
        out.append((raw_tok, digits + sign, decode_trailing_amount(digits, sign)))
    return out


def collapse_spaces(text: str) -> str:
    """Collapse runs of 2+ spaces to a single space for regex parsing contexts.
    Does not modify tabs or newlines (rare in our tokenizer flow)."""
    return re.sub(r" {2,}", " ", text or "")


# Flexible date parsing
DATE_TOKEN_RE = re.compile(r"\b(\d{6,8}|\d{4}-\d{2}-\d{2})\b")


def _infer_century(yy: int) -> int:
    import datetime as _dt
    now = _dt.datetime.now().year
    y20 = 2000 + yy
    y19 = 1900 + yy
    return y20 if abs(now - y20) <= abs(now - y19) else y19


def decode_date_token(tok: str) -> str | None:
    tok = tok.strip()
    # YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", tok)
    if m:
        return tok
    # YYYYMMDD (8 digits)
    if re.match(r"^\d{8}$", tok):
        return f"{tok[0:4]}-{tok[4:6]}-{tok[6:8]}"
    # YYMMDD (6 digits)
    if re.match(r"^\d{6}$", tok):
        yy = int(tok[0:2])
        yyyy = _infer_century(yy)
        return f"{yyyy}-{tok[2:4]}-{tok[4:6]}"
    return None


def extract_dates_generic(text: str) -> list[tuple[str, str | None]]:
    """Collect date tokens and their decoded forms (or None when undecodable)."""
    out: list[tuple[str, str | None]] = []
    for m in DATE_TOKEN_RE.finditer(text or ""):
        raw = m.group(1)
        out.append((raw, decode_date_token(raw)))
    return out


def _norm_date(s: str) -> str | None:
    if not s:
        return None
    s = s.strip()
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return s
    return None


def _norm_amount(s: str) -> str | None:
    if not s:
        return None
    s = s.strip()
    m = re.match(r"^(\d{1,})([+-])$", s)
    if m:
        val = int(m.group(1))
        sign = -1 if m.group(2) == '-' else 1
        return f"{sign * (val/100):.2f}"
    if s.isdigit():
        try:
            return f"{int(s)/100:.2f}"
        except Exception:
            return None
    m2 = re.match(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+\.\d{2}$", s)
    if m2:
        return s.replace(',', '')
    return None


def normalize_yymmdd(s: str) -> str:
    nd = _norm_date(s or "")
    return nd or ""


def normalize_money(s: str) -> str:
    na = _norm_amount(s or "")
    return na or ""


def _normalize_date8(s: str) -> str:
    """Normalize YYYYMMDD to YYYY-MM-DD with validity checks.
    Returns empty string if the token is not a valid calendar-like date
    (year 1900–2099, month 01–12, day 01–31). This allows callers to
    detect junk slices and fall back to alternatives.
    """
    s = (s or '').strip()
    if len(s) == 8 and s.isdigit():
        # Validate using DATE8_RE (YYYY MM DD with ranges)
        m = DATE8_RE.fullmatch(s)
        if m:
            y, mth, day = m.groups()
            return f"{y}-{mth}-{day}"
        return ""
    return ""


def _normalize_money_block(digits: str, sign: str) -> str:
    try:
        val = int(digits) / 100.0
        if sign == '-':
            val = -val
        return f"{val:.2f}"
    except Exception:
        return ""


def _first_date_in(body: str) -> str:
    m = DATE8_RE.search(body or "")
    if m:
        s = ''.join(m.groups())
        return _normalize_date8(s)
    return ""


# Public normalization helpers

def normalize_date8(s: str) -> str:
    return _normalize_date8(s)


def normalize_money_block(digits: str, sign: str) -> str:
    return _normalize_money_block(digits, sign)


def first_date_in(text: str) -> str:
    return _first_date_in(text)


def try_money_from(text: str, pick: int = -1) -> str:
    text = text or ""
    blocks = list(MONEY_BLOCK.finditer(text))
    if not blocks:
        return ""
    try:
        d, sgn = blocks[pick].groups()
    except Exception:
        return ""
    return normalize_money_block(d, sgn)


def guess_deductible(text: str) -> str:
    text = text or ""
    blocks = list(MONEY_BLOCK.finditer(text))
    if not blocks:
        return ""
    start = blocks[-1].start()
    m = re.search(r"(\d{3,5})\s*$", text[:start])
    if not m:
        return ""
    try:
        return f"{int(m.group(1)):.2f}"
    except Exception:
        return ""


# Liability helpers
LIAB_CODES = {"TPPD", "TPBI", "UA", "44", "AB", "23A", "CMP", "TPP", "TPB"}


def is_liability_coverage(code: str) -> bool:
    code = (code or "").strip().upper()
    return code in LIAB_CODES or code.startswith("TP")


def find_bare_limit_near(body: str, coverage_code: str) -> str:
    """Search for a 7–8 digit bare limit near/right of coverage token and return dollars string.
    Example: 01000000 -> 1000000.00
    """
    try:
        cov = (coverage_code or "").strip()
        pos = body.find(cov) if cov else -1
        search_seg = body[pos:] if pos >= 0 else body
        m = re.search(r"\b(0?\d{7,8})\b", search_seg)
        if m:
            digits = m.group(1)
            return f"{int(digits):.2f}"
    except Exception:
        pass
    return ""


def safe_slice(body: str, start: int, length: int) -> str:
    if start < 0 or length < 0:
        return ""
    if body is None:
        return ""
    end = start + length
    seg = body[start:end] if start < len(body) else ""
    return seg.rstrip()


# --- Display columns canonicalizer ---

def canonicalize_display_columns(cols: list[str], available: list[str] | set[str] | None = None) -> list[str]:
    """Return a strictly unique list of column names preserving order and valid existence.
    - Keep only the first occurrence of any name; drop later duplicates.
    - If available is provided, drop names not present in available.
    - Never repeat base columns: level, record_code, record_len, link_ref (if included).
    """
    if cols is None:
        return []
    seen: set[str] = set()
    base_dupes = {"level", "record_code", "record_len", "link_ref"}
    out: list[str] = []
    avail_set = set(available) if available is not None else None
    for c in cols:
        if not c:
            continue
        c2 = str(c)
        if avail_set is not None and c2 not in avail_set:
            # skip columns that don't exist (e.g., amount_1_norm placeholders)
            continue
        if c2 in seen:
            continue
        # ensure we never add base duplicates twice
        if c2 in base_dupes and c2 in seen:
            continue
        out.append(c2)
        seen.add(c2)
    return out
