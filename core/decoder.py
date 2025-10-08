#!/usr/bin/env python3
"""
csio_explorer.core.decoder
--------------------------
Schema-driven decoding and enrichment logic for CSIO records.
Relies on core.tokenizer.parse_lines and utilities from core.csio_utils.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, List
import pandas as pd

from .csio_utils import (
    safe_slice,
    find_dates,
    find_amounts,
    find_vin,
    find_year,
    find_postal,
    guess_coverage_code,
    try_money_from,
    guess_deductible,
    normalize_date8,
    normalize_yymmdd,
    normalize_money,
    find_make_model,
    first_date_in,
    is_liability_coverage,
    find_bare_limit_near,
    extract_trailing_amounts,
    collapse_spaces,
    extract_dates_generic,
)
from .tokenizer import parse_lines

# Meaning mapping (shared)
from .constants import CODE_MEANING


def sac_decode_from_body(body: str) -> dict:
    """Extract premium/limit/deductible from SAC body text by scanning numeric blocks.
    This is a heuristic: last money block is premium, previous is limit; deductible is a 3-5 digit number just before last money block (assumed dollars)."""
    from .csio_utils import MONEY_BLOCK, normalize_money_block
    import re

    out = {"premium": "", "limit": "", "deductible": ""}
    body = body or ""
    blocks = list(MONEY_BLOCK.finditer(body))
    if blocks:
        # premium as last block
        prem = blocks[-1].groups()
        out["premium"] = normalize_money_block(*prem)
        # limit as second last block if present
        if len(blocks) >= 2:
            lim = blocks[-2].groups()
            out["limit"] = normalize_money_block(*lim)
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
    # Streaming policy grouping state
    policy_group_id = 0
    current_policy_number = ""
    for _, r in base_df.iterrows():
        level = r.get("level")
        code = r.get("record_code")
        rlen = r.get("record_len")
        body = r.get("body") or ""
        body_raw = body
        cleaned = collapse_spaces(body)
        # Detect link_ref (first W/R/L id if present)
        import re as _re
        link_pat = _re.compile(r"([WRL]\d{5}(?:SAVR|HRUR|LAGL)\d{5})")
        m_lr = link_pat.search(cleaned)
        link_ref = m_lr.group(1) if m_lr else None
        out: Dict[str, Any] = {
            "level": level,
            "record_code": code,
            "record_len": rlen,
            "body": body,  # keep legacy 'body' for compatibility
            "body_raw": body_raw,
            "link_ref": link_ref,
            "link_key": (safe_slice(body, 0, 6) or "").strip(),
            "unknown_tail": None,
        }
        # Assign/propagate policy grouping
        if str(code) == "BPI":
            # Policy number from standard slice [22,18]
            pn = (safe_slice(body, 22, 18) or "").strip()
            current_policy_number = pn
            policy_group_id += 1
            out["current_policy_number"] = current_policy_number
            out["policy_group_id"] = policy_group_id
        else:
            out["current_policy_number"] = current_policy_number or None
            out["policy_group_id"] = policy_group_id if policy_group_id else None
        prov: Dict[str, tuple[int, int]] = {}
        explain: Dict[str, Any] = {"matched": None, "fields": {}, "amounts": [], "dates": []}

        # Generic amount extraction for every record (left->right)
        amts = extract_trailing_amounts(body)
        for idx, (raw_tok, raw_full, dec) in enumerate(amts, start=1):
            out[f"amount_{idx}_raw"] = raw_full
            out[f"amount_{idx}"] = dec
            explain["amounts"].append({"raw": raw_full, "decoded": dec})
        # Generic date tokens collection (store first two as date_1/date_2)
        dates = extract_dates_generic(body)
        for idx, (raw_d, dec_d) in enumerate(dates[:2], start=1):
            out[f"date_{idx}_raw"] = raw_d
            if dec_d:
                out[f"date_{idx}"] = dec_d
            explain["dates"].append({"raw": raw_d, "decoded": dec_d})

        def _assign(name: str, value: str | None):
            if value is None:
                return
            val = value.strip()
            if "?" in val:
                out[name + "_raw"] = val
                out[name] = None
            else:
                out[name] = val if val != "" else None

        layout = layouts.get(str(code))
        # Regex-first mappers per record code
        consumed_text = None
        consumed_from = "cleaned"
        if str(code) == "BIS":
            # Insured name from leading segment up to double-space. Use raw body to preserve spacing.
            # Skip optional leading 6-char sequence/agency token like B10001 or B10001?
            bis_m = _re.match(r"^\s*(?:[A-Z0-9]{6}\??\s+)?([A-Z0-9 ,&.'/-]+?)(?:\s{2,}|$)", body)
            if bis_m:
                cand = (bis_m.group(1) or '').strip()
                # Heuristic: if candidate looks like an address (starts with digits), don't use it
                if not _re.match(r"^\d{1,5}\b", cand):
                    explain["matched"] = "BIS"
                    _assign("insured_name", cand)
                    consumed_text = bis_m.group(0)
                    consumed_from = "body"
        elif str(code) == "9BIS":
            addr_re = _re.compile(r"^(?P<street>.+?)\s{2,}(?P<city>[A-Z .'-]+?)\s{2,}(?P<province>[A-Z]{2})\s+(?P<postal>[A-Z]\d[A-Z]\s?\d[A-Z]\d).*$")
            mm = addr_re.match(body)
            if mm:
                explain["matched"] = "9BIS"
                _assign("street", mm.group("street"))
                _assign("city", mm.group("city"))
                _assign("province", mm.group("province"))
                _assign("postal", mm.group("postal"))
                consumed_text = mm.group(0)
                consumed_from = "body"
        elif str(code) == "LAG":
            lag_re = _re.compile(r"^(?P<street>.+?)\s{2,}(?P<city>[A-Z .'-]+?)\s{2,}(?P<province>[A-Z]{2})\s+(?P<postal>[A-Z]\d[A-Z]\s?\d[A-Z]\d).*$")
            mm = lag_re.match(body)
            if mm:
                explain["matched"] = "LAG"
                _assign("street", mm.group("street"))
                _assign("city", mm.group("city"))
                _assign("province", mm.group("province"))
                _assign("postal", mm.group("postal"))
                consumed_text = mm.group(0)
                consumed_from = "body"
        elif str(code) == "SAC":
            sac_re = _re.compile(r"^(?P<link_ref>[WRL]\d{5}SAVR\d{5})\s+(?P<coverage_code>[A-Z0-9]{2,6})\s+(?P<rest>.+)$")
            mm = sac_re.match(cleaned)
            if mm:
                explain["matched"] = "SAC"
                _assign("link_ref", mm.group("link_ref"))
                _assign("coverage_code", mm.group("coverage_code"))
                rest = mm.group("rest")
                # Remove first two amount tokens from rest to build desc_tail
                amt_tokens = [t[0] for t in amts]
                tmp = rest
                removed = 0
                for tok in amt_tokens:
                    if removed >= 2:
                        break
                    pos = tmp.find(tok)
                    if pos >= 0:
                        tmp = (tmp[:pos] + tmp[pos+len(tok):]).strip()
                        removed += 1
                if tmp:
                    out["desc_tail"] = tmp
                # Map premium/limit
                if out.get("amount_1") is not None:
                    out["premium"] = out.get("amount_1")
                if out.get("amount_2") is not None:
                    out["limit"] = out.get("amount_2")
                consumed_text = mm.group(0)
        elif str(code) == "CVH":
            cvh_re = _re.compile(r"^(?P<link_ref>[WRL]\d{5}HRUR\d{5})\s+(?P<coverage_code>[A-Z0-9]{2,6})\s+(?P<rest>.+)$")
            mm = cvh_re.match(cleaned)
            if mm:
                explain["matched"] = "CVH"
                _assign("link_ref", mm.group("link_ref"))
                _assign("coverage_code", mm.group("coverage_code"))
                rest = mm.group("rest")
                if out.get("amount_1") is not None:
                    out["premium"] = out.get("amount_1")
                if out.get("amount_2") is not None:
                    out["limit"] = out.get("amount_2")
                # Try deductible short snippet from rest (2-3 uppercase letters or 3-5 digits before amounts)
                m_d = _re.search(r"\b([A-Z]{2,3}|\d{3,5})\b", rest)
                if m_d:
                    _assign("ded_raw", m_d.group(1))
                # Remaining narrative
                out["desc_tail"] = rest
                consumed_text = mm.group(0)
        elif str(code) == "SAV":
            sav_re = _re.compile(r"^(?P<link_ref>[WRL]\d{5}LAGL\d{5})\s+(?P<rest>.+)$")
            mm = sav_re.match(cleaned)
            if mm:
                explain["matched"] = "SAV"
                _assign("link_ref", mm.group("link_ref"))
                rest = mm.group("rest")
                # vehicle heuristics
                from .csio_utils import find_vin as _fvin, find_year as _fyr, find_make_model as _fmm
                vin = _fvin(rest)
                yr = _fyr(rest)
                mk, mdl = _fmm(rest)
                if vin: _assign("vin", vin)
                if yr: _assign("year", yr)
                if mk: _assign("make", mk)
                if mdl: _assign("model", mdl)
                consumed_text = mm.group(0)
        elif str(code) == "AOI":
            aoi_re = _re.compile(r"^(?P<link_ref>[WRL]\d{5}(?:SAVR|HRUR)\d{5})\s+(?P<interest_code>\d{2,3}[A-Z]{0,2}).*?NT(?P<name>[^?]+?)\s{2,}(?P<rest>.*)$")
            mm = aoi_re.match(cleaned)
            if mm:
                explain["matched"] = "AOI"
                _assign("link_ref", mm.group("link_ref"))
                _assign("interest_code", mm.group("interest_code"))
                _assign("name", mm.group("name"))
                if mm.group("rest").strip():
                    out["unknown_tail"] = mm.group("rest").strip()
                consumed_text = mm.group(0)
        # Fallback to schema-driven slices (do not overwrite regex-extracted fields)
        layout = layouts.get(str(code))
        if layout and layout.get("fields"):
            for f in layout["fields"]:
                name = f.get("name")
                if name in out and out.get(name) not in (None, ""):
                    continue
                start = int(f.get("start", 0))
                length = int(f.get("length", 0))
                val = safe_slice(body, start, length)
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
                    # Use strict 8-digit date normalization; invalid tokens yield empty and won't overwrite
                    nd = normalize_date8(raw_src)
                    if nd is not None and nd != "":
                        out[base] = nd
                # Money amounts
                if base and any(tok in base for tok in ["amount", "amt", "premium", "limit", "ded", "deduct"]):
                    na = normalize_money(raw_src)
                    if na is not None:
                        target = base
                        if base in ("ded", "deduct"):
                            target = "deductible"
                        out[target] = na
            # legacy light normalizations to _norm for older schemas
            for k in list(out.keys()):
                if k.endswith("date") or k.endswith("_date"):
                    nd2 = normalize_yymmdd(str(out.get(k, "")).strip())
                    if nd2 and not out.get(k + "_norm"):
                        out[k + "_norm"] = nd2
                if any(tok in k for tok in ["amount", "premium", "limit", "deduct"]):
                    na2 = normalize_money(str(out.get(k, "")).strip())
                    if na2 and not out.get(k + "_norm"):
                        out[k + "_norm"] = na2
            # Additional heuristics for SAC/CVH/PAY/BPI/SAV
            if str(code) == "SAC":
                # Liability bare-limit fallback when limit is missing/zero
                cov_code = (out.get("coverage_code") or "").strip()
                lim_cur = (out.get("limit") or "").strip()
                if (not lim_cur or lim_cur in ("0", "0.00", "0.0")) and is_liability_coverage(cov_code):
                    fallback = find_bare_limit_near(body, cov_code)
                    if fallback:
                        out["limit"] = fallback
            if str(code) == "CVH":
                prem = try_money_from(body, -1)
                lim = try_money_from(body, -2)
                ded_guess = guess_deductible(body)
                if prem and not out.get("premium"):
                    out["premium"] = prem
                if lim and not out.get("limit"):
                    out["limit"] = lim
                if ded_guess and not out.get("deductible"):
                    out["deductible"] = ded_guess
                # fallback cents/dollars for raw fields
                if not out.get("premium"):
                    out["premium"] = normalize_money(str(out.get("premium_raw","")) or "") or ""
                if not out.get("limit"):
                    out["limit"] = normalize_money(str(out.get("limit_raw","")) or "") or ""
                if not out.get("deductible"):
                    dr = str(out.get("ded_raw",""))
                    if dr and dr.strip().isdigit():
                        try:
                            out["deductible"] = f"{int(dr):.2f}"
                        except Exception:
                            pass
                cov_code = (out.get("coverage_code") or "").strip()
                lim_cur = (out.get("limit") or "").strip()
                if (not lim_cur or lim_cur in ("0", "0.00", "0.0")) and is_liability_coverage(cov_code):
                    fallback = find_bare_limit_near(body, cov_code)
                    if fallback:
                        out["limit"] = fallback
            if str(code) == "PAY":
                for i in (1,2):
                    ar = str(out.get(f"amount_{i}_raw",""))
                    if ar and not out.get(f"amount_{i}"):
                        out[f"amount_{i}"] = normalize_money(ar) or ""
                    dr = str(out.get(f"date_{i}_raw",""))
                    if dr and not out.get(f"date_{i}"):
                        out[f"date_{i}"] = normalize_date8(dr)
            if str(code) == "BPI":
                eff = str(out.get("effective_raw",""))
                exp = str(out.get("expiry_raw",""))
                eff_n = normalize_date8(eff) or first_date_in(body)
                # Prefer explicit expiry_raw; if invalid/empty, try the second date token in body
                exp_n = normalize_date8(exp)
                if not exp_n:
                    dates_in = extract_dates_generic(body)
                    if len(dates_in) >= 2:
                        cand = dates_in[1][1]  # decoded form of second token
                        if cand:
                            exp_n = cand
                if eff_n:
                    out["effective"] = eff_n
                if exp_n:
                    out["expiry"] = exp_n
            if str(code) == "SAV":
                from .csio_utils import find_vin as _fvin, find_year as _fyr, find_make_model as _fmm
                if not out.get("vin"):
                    vin = _fvin(body) or ""
                    if vin:
                        out["vin"] = vin
                if not out.get("year"):
                    yr = _fyr(body) or ""
                    if yr:
                        out["year"] = yr
                if not out.get("make") or not out.get("model"):
                    mk, mdl = _fmm(body)
                    if mk: out["make"] = mk
                    if mdl: out["model"] = mdl
        # unknown_tail if not set and we matched
        if out.get("unknown_tail") is None and consumed_text:
            src_txt = body if consumed_from == "body" else cleaned
            tail = src_txt[len(consumed_text):].strip()
            out["unknown_tail"] = tail or None
        out["provenance"] = prov
        out["explain"] = explain
        # Meaning and hints
        out["meaning"] = CODE_MEANING.get(str(code), "(unknown code)") if 'CODE_MEANING' in globals() else None
        # Hints using utilities
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
