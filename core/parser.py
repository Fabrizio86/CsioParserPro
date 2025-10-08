#!/usr/bin/env python3
"""
Legacy module removed.
Use core/* modules instead (core.tokenizer, core.decoder, core.csio_utils).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import yaml

# -----------------------------
# Meaning mapping (shared)
# -----------------------------
from .constants import CODE_MEANING

# -----------------------------
# Helpers (heuristic extractors)
# -----------------------------

VIN_RE = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
POSTAL_RE = re.compile(r"\b([ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]\s?\d[ABCEGHJ-NPRSTV-Z]\d)\b", re.I)
ISO_DATE_RE = re.compile(r"\b(20\d{2}|19\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b")
AMOUNT_RE = re.compile(r"(?:\$?\s*)(\d{1,3}(?:,\d{3})*(?:\.\d{2})|\d+\.\d{2})")


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


def guess_name_from_bis(body: str) -> str | None:
    m = re.match(r"^([A-Z0-9 ,&.'/-]+?)(?:\s{2,}|$)", (body or "").strip())
    return m.group(1).strip() if m else None


# -----------------------------
# Parsing (header/length aware tokenizer)
# -----------------------------

HEADER_RE = re.compile(r"(\d)([A-Z]{3})(\d{3})")

def tokenize_records(raw: str):
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
        yield {"level": level, "record_code": code, "record_len": f"{rlen:03d}", "body": body}
        i = body_end

def parse_lines(lines: List[str]) -> pd.DataFrame:
    """Parse by scanning the entire content as a single string; no newline dependency."""
    if not lines:
        return pd.DataFrame([])
    # Join without inserting extra characters so byte positions remain correct
    raw = "".join([ln.rstrip("\r\n") for ln in lines])
    rows: List[Dict[str, Any]] = []
    for rec in tokenize_records(raw):
        rows.append(rec)
    return pd.DataFrame(rows)


def enrich_by_code(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    if df.empty:
        return dfs
    dfs["ALL_RECORDS"] = df.copy()

    def maybe_add(name: str, frame: pd.DataFrame):
        if not frame.empty:
            dfs[name] = frame.copy()

    if (df["record_code"] == "BIS").any():
        d = df[df["record_code"] == "BIS"].copy()
        d["insured_name_guess"] = d["body"].apply(guess_name_from_bis)
        maybe_add("BIS", d)

    for code in ["ISI", "BPI"]:
        mask = df["record_code"] == code
        if mask.any():
            d = df[mask].copy()
            d["dates_found"] = d["body"].apply(find_dates)
            maybe_add(code, d)

    if (df["record_code"] == "RMK").any():
        maybe_add("RMK", df[df["record_code"] == "RMK"])

    for code in ["PAY", "EFT", "ACH"]:
        mask = df["record_code"] == code
        if mask.any():
            d = df[mask].copy()
            d["amounts_found"] = d["body"].apply(find_amounts)
            d["dates_found"] = d["body"].apply(find_dates)
            maybe_add(code, d)

    if (df["record_code"] == "LAG").any():
        d = df[df["record_code"] == "LAG"].copy()
        d["postal_guess"] = d["body"].apply(find_postal)
        maybe_add("LAG", d)

    if (df["record_code"] == "HRU").any():
        maybe_add("HRU", df[df["record_code"] == "HRU"])

    if (df["record_code"] == "AOI").any():
        d = df[df["record_code"] == "AOI"].copy()
        d["postal_guess"] = d["body"].apply(find_postal)
        maybe_add("AOI", d)

    if (df["record_code"] == "CVH").any():
        d = df[df["record_code"] == "CVH"].copy()
        d["coverage_code_guess"] = d["body"].apply(guess_coverage_code)
        d["amounts_found"] = d["body"].apply(find_amounts)
        maybe_add("CVH", d)

    if (df["record_code"] == "SAV").any():
        d = df[df["record_code"] == "SAV"].copy()
        d["vin_guess"] = d["body"].apply(find_vin)
        d["year_guess"] = d["body"].apply(find_year)
        maybe_add("SAV", d)

    if (df["record_code"] == "SAC").any():
        d = df[df["record_code"] == "SAC"].copy()
        d["coverage_code_guess"] = d["body"].apply(guess_coverage_code)
        d["amounts_found"] = d["body"].apply(find_amounts)
        maybe_add("SAC", d)

    if (df["record_code"] == "SAD").any():
        d = df[df["record_code"] == "SAD"].copy()
        d["dates_found"] = d["body"].apply(find_dates)
        maybe_add("SAD", d)

    if (df["record_code"] == "CHG").any():
        d = df[df["record_code"] == "CHG"].copy()
        d["amounts_found"] = d["body"].apply(find_amounts)
        maybe_add("CHG", d)

    return dfs

# -----------------------------
# Schema-driven parsing
# -----------------------------

def load_schema(src: Any) -> Dict[str, Any]:
    """Load schema from a Path, str path, bytes/str YAML, or a dict that already looks like schema.
    Returns a dict with keys: version (str), layouts (dict[code]-> {description, fields}).
    Each field is a dict: {name, start, length, meaning}.
    """
    if isinstance(src, dict):
        schema = src
    else:
        text: str
        if isinstance(src, (str, Path)):
            p = Path(src)
            text = p.read_text(encoding="utf-8")
        elif isinstance(src, (bytes, bytearray)):
            text = src.decode("utf-8", errors="ignore")
        elif isinstance(src, str):
            text = src
        else:
            raise TypeError("Unsupported schema source type")
        schema = yaml.safe_load(text) or {}
    # Basic validation/sanitization
    version = str(schema.get("version", "csio-240-v1"))
    layouts = schema.get("layouts") or {}
    if not isinstance(layouts, dict):
        layouts = {}
    # Ensure each layout has fields list
    clean_layouts: Dict[str, Any] = {}
    for code, spec in layouts.items():
        if not isinstance(spec, dict):
            continue
        desc = spec.get("description") or ""
        fields = spec.get("fields") or []
        cleaned_fields = []
        for f in fields:
            try:
                name = str(f.get("name", ""))
                start = int(f.get("start", 0))
                length = int(f.get("length", 0))
                meaning = str(f.get("meaning", ""))
                if name and length >= 0 and start >= 0:
                    cleaned_fields.append({"name": name, "start": start, "length": length, "meaning": meaning})
            except Exception:
                continue
        clean_layouts[str(code)] = {"description": desc, "fields": cleaned_fields}
    return {"version": version, "layouts": clean_layouts}


def save_schema(schema: Dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(schema, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _safe_slice(body: str, start: int, length: int) -> str:
    if start < 0 or length < 0:
        return ""
    if body is None:
        return ""
    end = start + length
    seg = body[start:end] if start < len(body) else ""
    return seg.rstrip()


def _norm_date(s: str) -> str | None:
    if not s:
        return None
    s = s.strip()
    # Accept YYYYMMDD or YYYY-MM-DD
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
    # Handle fixed with trailing sign, e.g., 00000006850+ or 00000006850-
    m = re.match(r"^(\d{1,})([+-])$", s)
    if m:
        val = int(m.group(1))
        sign = -1 if m.group(2) == '-' else 1
        return f"{sign * (val/100):.2f}"
    # Handle pure digits (assume cents)
    if s.isdigit():
        try:
            return f"{int(s)/100:.2f}"
        except Exception:
            return None
    # Handle 1,234.56 or 123.45
    m2 = re.match(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+\.\d{2}$", s)
    if m2:
        return s.replace(',', '')
    return None

# Public normalization utilities required by UI

def normalize_yymmdd(s: str) -> str:
    nd = _norm_date(s or "")
    return nd or ""


def normalize_money(s: str) -> str:
    na = _norm_amount(s or "")
    return na or ""


# --- Money/date/VIN helpers required by spec ---
MONEY_BLOCK_RE = re.compile(r"(?<!\d)(\d{9,12})([+-])")
DATE8_RE = re.compile(r"\b(20\d{2}|19\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b")

# Public names as specified
MONEY_BLOCK = MONEY_BLOCK_RE

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

# Vehicle make/model heuristics

def find_make_model(text: str) -> tuple[str, str]:
    text = text or ""
    yr = find_year(text)
    if not yr:
        return ("", "")
    vin_m = VIN_RE.search(text)
    vin_pos = vin_m.start() if vin_m else len(text)
    # take subsection after year, up to VIN or end
    idx = text.find(yr)
    tail = text[idx+len(yr):vin_pos]
    # Split into uppercase tokens
    toks = re.findall(r"\b[A-Z0-9][A-Z0-9-]+\b", tail)
    if not toks:
        return ("", "")
    make = toks[0]
    model = " ".join(toks[1:]) if len(toks) > 1 else ""
    return (make, model)

# --- SAC decoding helpers (for auto coverages) ---

def _normalize_date8(s: str) -> str:
    s = (s or '').strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
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


def sac_decode_from_body(body: str) -> dict:
    """Extract premium/limit/deductible from SAC body text by scanning numeric blocks.
    This is a heuristic: last money block is premium, previous is limit; deductible is a 3-5 digit number just before last money block (assumed dollars)."""
    out = {"premium": "", "limit": "", "deductible": ""}
    body = body or ""
    blocks = list(MONEY_BLOCK_RE.finditer(body))
    if blocks:
        # premium as last block
        prem = blocks[-1].groups()
        out["premium"] = _normalize_money_block(*prem)
        # limit as second last block if present
        if len(blocks) >= 2:
            lim = blocks[-2].groups()
            out["limit"] = _normalize_money_block(*lim)
        # deductible: digits before premium
        start = blocks[-1].start()
        m = re.search(r"(\d{3,5})\s*$", body[:start])
        if m:
            try:
                out["deductible"] = f"{int(m.group(1)):.2f}"
            except Exception:
                pass
    return out


def parse_with_schema(lines: List[str], schema: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Schema-driven parse. Returns dict of code->DataFrame. Includes keys 'ALL' and 'ALL_RECORDS'.
    Each DataFrame includes: level, record_code, record_len, body, parsed fields (if any), provenance (dict), meaning, hints.
    """
    base_df = parse_lines(lines)
    if base_df.empty:
        return {"ALL": base_df, "ALL_RECORDS": base_df}
    layouts = (schema or {}).get("layouts", {})
    # Start with base columns
    rows: List[Dict[str, Any]] = []
    for _, r in base_df.iterrows():
        level = r.get("level")
        code = r.get("record_code")
        rlen = r.get("record_len")
        body = r.get("body") or ""
        out: Dict[str, Any] = {"level": level, "record_code": code, "record_len": rlen, "body": body, "link_key": (_safe_slice(body, 0, 6) or "").strip()}
        prov: Dict[str, Tuple[int, int]] = {}
        layout = layouts.get(str(code))
        if layout and layout.get("fields"):
            for f in layout["fields"]:
                name = f.get("name")
                start = int(f.get("start", 0))
                length = int(f.get("length", 0))
                val = _safe_slice(body, start, length)
                out[name] = val
                prov[name] = (start, start + length)
            # compute readable columns from *_raw names per spec
            for k in list(out.keys()):
                if k in ("level", "record_code", "record_len", "body", "meaning", "hints", "provenance"):
                    continue
                raw_src = None
                base = None
                if k.endswith("_raw"):
                    raw_src = str(out.get(k, "")).strip()
                    base = k[:-4]
                # Dates
                if base and ("date" in base or base in ("effective", "expiry") or base.startswith("date_")):
                    nd = _norm_date(raw_src)
                    if nd is not None:
                        out[base] = nd
                # Money amounts
                if base and any(tok in base for tok in ["amount", "amt", "premium", "limit", "ded", "deduct"]):
                    na = _norm_amount(raw_src)
                    if na is not None:
                        target = base
                        if base in ("ded", "deduct"):
                            target = "deductible"
                        out[target] = na
            # legacy light normalizations to _norm for older schemas
            for k in list(out.keys()):
                if k.endswith("date") or k.endswith("_date"):
                    nd2 = _norm_date(str(out.get(k, "")).strip())
                    if nd2 and not out.get(k + "_norm"):
                        out[k + "_norm"] = nd2
                if any(tok in k for tok in ["amount", "premium", "limit", "deduct"]):
                    na2 = _norm_amount(str(out.get(k, "")).strip())
                    if na2 and not out.get(k + "_norm"):
                        out[k + "_norm"] = na2
            # SAC specific decoding from body when *_raw not provided
            if str(code) == "SAC":
                sac_vals = sac_decode_from_body(body)
                # Only fill if missing to avoid overwriting schema-derived values
                for key, val in sac_vals.items():
                    if val and not out.get(key):
                        out[key] = val
            # CVH computed/heuristic columns
            if str(code) == "CVH":
                # premium/limit from right-edge money blocks if present
                prem = try_money_from(body, -1)
                lim = try_money_from(body, -2)
                ded_guess = guess_deductible(body)
                if prem and not out.get("premium"):
                    out["premium"] = prem
                if lim and not out.get("limit"):
                    out["limit"] = lim
                if ded_guess and not out.get("deductible"):
                    out["deductible"] = ded_guess
                # also try cents from *_raw with special case: ded_raw is dollars
                if not out.get("premium"):
                    out["premium"] = _norm_amount(str(out.get("premium_raw","")) or "") or ""
                if not out.get("limit"):
                    out["limit"] = _norm_amount(str(out.get("limit_raw","")) or "") or ""
                if not out.get("deductible"):
                    dr = str(out.get("ded_raw",""))
                    if dr and dr.strip().isdigit():
                        try:
                            out["deductible"] = f"{int(dr):.2f}"
                        except Exception:
                            pass
            # PAY amounts/dates normalized
            if str(code) == "PAY":
                for i in (1,2):
                    ar = str(out.get(f"amount_{i}_raw",""))
                    if ar and not out.get(f"amount_{i}"):
                        out[f"amount_{i}"] = _norm_amount(ar) or ""
                    dr = str(out.get(f"date_{i}_raw",""))
                    if dr and not out.get(f"date_{i}"):
                        out[f"date_{i}"] = _normalize_date8(dr)
            # BPI effective/expiry
            if str(code) == "BPI":
                eff = str(out.get("effective_raw",""))
                exp = str(out.get("expiry_raw",""))
                eff_n = _normalize_date8(eff) or _first_date_in(body)
                exp_n = _normalize_date8(exp)
                if eff_n:
                    out["effective"] = eff_n
                if exp_n:
                    out["expiry"] = exp_n
            # SAV vehicle heuristics
            if str(code) == "SAV":
                vin = find_vin(body) or ""
                yr = find_year(body) or ""
                mk, mdl = find_make_model(body)
                if vin: out["vin"] = vin
                if yr: out["year"] = yr
                if mk: out["make"] = mk
                if mdl: out["model"] = mdl
        out["provenance"] = prov
        # Meaning and hints
        out["meaning"] = CODE_MEANING.get(str(code), "(unknown code)") if 'CODE_MEANING' in globals() else None
        # Late import avoidance: reuse heuristics in this module
        h_dates = find_dates(body)
        h_amts = find_amounts(body)
        h_vin = find_vin(body)
        h_year = find_year(body)
        h_postal = find_postal(body)
        h_cov = guess_coverage_code(body)
        hints: Dict[str, str] = {}
        if h_dates: hints["dates"] = h_dates
        if h_amts: hints["amounts"] = h_amts
        if h_vin: hints["vin"] = h_vin
        if h_year: hints["year"] = h_year
        if h_postal: hints["postal"] = h_postal
        if h_cov and not h_cov.isdigit(): hints["coverage_guess"] = h_cov
        out["hints"] = hints if hints else None
        rows.append(out)
    all_df = pd.DataFrame(rows)
    out_dict: Dict[str, pd.DataFrame] = {"ALL": all_df, "ALL_RECORDS": all_df}
    for code, grp in all_df.groupby("record_code"):
        out_dict[str(code)] = grp.reset_index(drop=True)
    return out_dict


# -----------------------------
# I/O utilities (no CLI wiring here)
# -----------------------------

def load_text(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [ln.rstrip("\n\r") for ln in f]


def save_outputs(dfs: dict[str, pd.DataFrame], outdir: Path, excel_path: Path | None, per_code_csv: bool):
    outdir.mkdir(parents=True, exist_ok=True)

    all_df = dfs.get("ALL_RECORDS")
    if all_df is not None:
        all_df.to_csv(outdir / "all_records.csv", index=False)

    if per_code_csv:
        for name, dfx in dfs.items():
            if name == "ALL_RECORDS":
                continue
            safe = (name[:31] or name).replace("/", "_")
            dfx.to_csv(outdir / f"{safe}.csv", index=False)

    if excel_path:
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            for name, dfx in dfs.items():
                safe_name = (name[:31] or name).replace("/", "_")
                dfx.to_excel(writer, index=False, sheet_name=safe_name)


def print_summary(df: pd.DataFrame):
    print("\n=== Record counts by code ===")
    cnt = df.groupby("record_code").size().sort_values(ascending=False)
    print(cnt.to_string())
    print("\nExample rows:")
    for code in cnt.index[:5]:
        subset = df[df["record_code"] == code].head(3)
        print(f"\n-- {code} --")
        print(subset.to_string(index=False))
