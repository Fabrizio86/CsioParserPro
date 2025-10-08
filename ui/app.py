#!/usr/bin/env python3
"""
csio_explorer.ui.app
--------------------
Streamlit UI for exploring CSIO EDI flat files.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path
import io
import yaml
from typing import List, Dict, Any

# Safely set page config only when running under Streamlit (avoid ScriptRunContext warning in bare mode)
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
except Exception:
    get_script_run_ctx = None  # type: ignore

if 'get_script_run_ctx' in globals():
    try:
        ctx = get_script_run_ctx() if get_script_run_ctx else None
    except Exception:
        ctx = None
    if ctx is not None:
        try:
            st.set_page_config(page_title="CSIO Parser Pro", layout="wide")
        except Exception:
            pass

# -----------------------------
# Shared constants and helpers
# -----------------------------
try:
    from ..core.constants import CODE_MEANING, PRIORITY_TABS  # type: ignore
    from .app_helpers import render_policy_summary, render_per_code_tabs, render_exports_sidebar  # type: ignore
    from ..core.csio_utils import (  # type: ignore
        find_dates,
        find_amounts,
        find_vin,
        find_year,
        find_postal,
        guess_coverage_code,
    )
    from ..core.schema_io import load_schema, save_schema  # type: ignore
    from ..core.decoder import parse_with_schema  # type: ignore
except Exception:
    # Fallback for when this file is executed as a standalone script via Streamlit,
    # where relative imports are not available (no package context).
    import importlib, sys
    from pathlib import Path as _Path
    _pkg_dir = _Path(__file__).resolve().parents[1]
    _pkg_parent = str(_pkg_dir.parent)
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    _pkg_name = _pkg_dir.name

    _constants = importlib.import_module(f"{_pkg_name}.core.constants")
    CODE_MEANING = getattr(_constants, "CODE_MEANING")
    PRIORITY_TABS = getattr(_constants, "PRIORITY_TABS")

    _helpers = importlib.import_module(f"{_pkg_name}.ui.app_helpers")
    render_policy_summary = getattr(_helpers, "render_policy_summary")
    render_per_code_tabs = getattr(_helpers, "render_per_code_tabs")
    render_exports_sidebar = getattr(_helpers, "render_exports_sidebar")

    _csio = importlib.import_module(f"{_pkg_name}.core.csio_utils")
    find_dates = getattr(_csio, "find_dates")
    find_amounts = getattr(_csio, "find_amounts")
    find_vin = getattr(_csio, "find_vin")
    find_year = getattr(_csio, "find_year")
    find_postal = getattr(_csio, "find_postal")
    guess_coverage_code = getattr(_csio, "guess_coverage_code")

    _schema_io = importlib.import_module(f"{_pkg_name}.core.schema_io")
    load_schema = getattr(_schema_io, "load_schema")
    save_schema = getattr(_schema_io, "save_schema")

    _decoder = importlib.import_module(f"{_pkg_name}.core.decoder")
    parse_with_schema = getattr(_decoder, "parse_with_schema")

# -----------------------------
# Heuristics/hints
# -----------------------------


# -----------------------------
# Hints builder
# -----------------------------

def build_hints(body: str, record_code: str | None = None) -> str | None:
    """Create a JSON-ish string with lightweight hints detected in the body.
    Example: {"dates": "2023-06-19", "amounts": "292.03", ...}
    """
    body = body or ""
    hints: Dict[str, str] = {}

    # These apply broadly
    dates = find_dates(body)
    if dates:
        hints["dates"] = dates
    amts = find_amounts(body)
    if amts:
        hints["amounts"] = amts
    vin = find_vin(body)
    if vin:
        hints["vin"] = vin
    yr = find_year(body)
    if yr:
        hints["year"] = yr
    postal = find_postal(body)
    if postal:
        hints["postal"] = postal

    cov = guess_coverage_code(body)
    if cov and not cov.isdigit():
        hints["coverage_guess"] = cov

    if not hints:
        return None

    # Return a simple JSON-ish dict string
    import json
    try:
        return json.dumps(hints, ensure_ascii=False)
    except Exception:
        parts = [f"{k}={v}" for k, v in hints.items()]
        return ", ".join(parts)


# -----------------------------
# Raw CSIO record splitter (handles blobs without newlines)
# -----------------------------

def split_raw_csio_records(raw: str) -> List[str]:
    """Split a raw CSIO blob into pseudo-lines at record boundaries.
    A record boundary looks like: <level digit><3-letter code><3-digit length>, e.g., 5BIS240.
    If the input already contains many newlines, prefer those instead.
    Returns a list of strings representing individual records (not including trailing newlines).
    """
    raw = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    # If the content already has many lines, keep them
    lines = [ln for ln in raw.split("\n") if ln.strip() != ""]
    if len(lines) > 5:
        return lines

    import re
    pat = re.compile(r"\d[A-Z]{3}\d{3}")
    starts = [m.start() for m in pat.finditer(raw)]
    if not starts:
        # No recognizable records; return the whole text as a single line
        return [raw]

    segments: List[str] = []
    # Include any leading garbage as its own segment so it shows up as UNK
    if starts[0] > 0:
        lead = raw[: starts[0]].strip()
        if lead:
            segments.append(lead)

    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(raw)
        seg = raw[s:e].strip("\n")
        if seg:
            segments.append(seg)
    return segments


# -----------------------------
# UI
# -----------------------------

def make_app():
    """Main Streamlit application entry point.
    This function orchestrates input handling, schema loading, parsing,
    policy summary rendering, per-record-code tabs, and exports.
    Keep this as a thin coordinator; heavy lifting is factored into helpers.
    """
    st.title("CSIO Parser Pro")

    # Sidebar: Inputs and Schema
    st.sidebar.header("Inputs")
    input_mode = st.sidebar.radio("Provide data via", ["Upload file", "Paste text"], horizontal=False)
    pasted_text = None
    uploaded = None
    if input_mode == "Upload file":
        uploaded = st.sidebar.file_uploader("Upload CSIO flat file", type=["txt", "dat", "csv", "log"]) 
    else:
        pasted_text = st.sidebar.text_area("Paste CSIO data", height=200, help="You can paste a blob without newlines.")

    # Schema controls
    st.sidebar.header("Schema")
    default_schema_path = Path(__file__).with_name("csio_layout.yaml")
    schema_bytes = None
    schema_upl = st.sidebar.file_uploader("Load schema (YAML/JSON)", type=["yaml", "yml", "json"], key="schema_uploader")
    if schema_upl is not None:
        schema_bytes = schema_upl.getvalue()
        schema = load_schema(schema_bytes)
    else:
        # Load bundled default
        try:
            schema = load_schema(default_schema_path)
        except Exception:
            schema = {"version": "csio-240-v1", "layouts": {}}

    # View settings
    show_prov = st.sidebar.toggle("Show provenance overlays", value=True)

    # Reference dictionary
    with st.sidebar.expander("Record code meanings"):
        ref_df = pd.DataFrame({"code": list(CODE_MEANING.keys()), "meaning": list(CODE_MEANING.values())})
        st.dataframe(ref_df, hide_index=True, use_container_width=True)

    # Parse input lines
    lines: List[str] = []
    if uploaded is not None:
        content = uploaded.getvalue().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        if len(lines) <= 2:
            lines = split_raw_csio_records(content)
    elif pasted_text:
        lines = split_raw_csio_records(pasted_text)

    if not lines:
        st.info("Upload or paste CSIO data to begin.")
        return

    parsed = parse_with_schema(lines, schema)
    # Avoid ambiguous truth value for DataFrame by checking for None explicitly
    all_df = parsed.get("ALL")
    if all_df is None:
        all_df = parsed.get("ALL_RECORDS")
    if all_df is None or getattr(all_df, "empty", True):
        st.warning("No records parsed.")
        return

    # Overview
    st.subheader("Overview")
    counts = all_df.groupby("record_code").size().reset_index(name="count").sort_values("count", ascending=False)
    counts["meaning"] = counts["record_code"].map(CODE_MEANING).fillna("(unknown)")
    st.dataframe(counts[["record_code", "meaning", "count"]], hide_index=True, use_container_width=True)

    # Unknown codes alert
    unknown_codes = [c for c in counts["record_code"].tolist() if str(c) not in (schema.get("layouts") or {})]
    if unknown_codes:
        st.warning(f"Unknown/Unmapped codes (no layout): {', '.join(unknown_codes)}")
        with st.expander("Pending layout", expanded=False):
            st.write("These codes do not have a layout yet. Generate YAML stubs to start documenting them.")
            if st.button("Generate layout skeleton", key="gen_stub_all"):
                try:
                    layouts = schema.get("layouts") or {}
                    for uc in unknown_codes:
                        sc = str(uc)
                        if sc not in layouts:
                            layouts[sc] = {"description": "Pending layout", "fields": []}
                    new_schema = {"version": schema.get("version", "csio-240-v1"), "layouts": layouts}
                    save_schema(new_schema, default_schema_path)
                    st.success(f"Layout stubs saved to {default_schema_path}. Reload to apply.")
                except Exception as ex:
                    st.error(f"Failed to write stubs: {ex}")

    # Policy Summary (roll-up by policy)
    render_policy_summary(parsed, schema, all_df)

    # Per-code tabs
    render_per_code_tabs(parsed, schema, all_df, show_prov)

    # Schema Preview
    st.subheader("Schema Preview: CSIO Field Reference")
    dict_rows = []
    for c, spec in (schema.get("layouts") or {}).items():
        for f in spec.get("fields") or []:
            dict_rows.append({
                "code": c,
                "field": f.get("name"),
                "meaning": f.get("meaning"),
                "start": f.get("start"),
                "length": f.get("length"),
            })
    dd_df = pd.DataFrame(dict_rows)
    if not dd_df.empty:
        st.dataframe(dd_df, hide_index=True, use_container_width=True)
        st.download_button("Download data_dictionary.csv", dd_df.to_csv(index=False).encode("utf-8"), file_name="data_dictionary.csv", mime="text/csv")
        st.download_button("Download csio_layout.yaml", (default_schema_path.read_text(encoding='utf-8') if default_schema_path.exists() else yaml.safe_dump(schema, sort_keys=False)).encode("utf-8"), file_name="csio_layout.yaml", mime="text/yaml")

    # Schema Editor
    st.subheader("Schema Editor")
    layouts = schema.get("layouts") or {}
    codes_sorted = sorted(layouts.keys())
    sel_code = st.selectbox("Select code to edit", options=codes_sorted or ["BIS"], index=0 if codes_sorted else 0)
    current = layouts.get(sel_code, {"description": "", "fields": []})
    st.text_input("Description", value=current.get("description", ""), key="desc_"+sel_code)

    # Editable grid
    fields_df = pd.DataFrame(current.get("fields") or [])
    fields_df = st.data_editor(fields_df, num_rows="dynamic", use_container_width=True, key=f"editor_{sel_code}")

    # Save schema
    if st.button("Save Schema", type="primary"):
        # Apply edited fields back
        layouts[sel_code] = {"description": st.session_state.get("desc_"+sel_code, current.get("description","")), "fields": fields_df.to_dict(orient='records')}
        new_schema = {"version": schema.get("version","csio-240-v1"), "layouts": layouts}
        try:
            save_schema(new_schema, default_schema_path)
            st.success(f"Schema saved to {default_schema_path}")
        except Exception as ex:
            st.error(f"Failed to save schema: {ex}")

    # Exports
    render_exports_sidebar(parsed, all_df, schema, default_schema_path)


def _has_streamlit_ctx() -> bool:
    """Return True when running under Streamlit runtime (ScriptRunContext exists)."""
    try:
        return bool(get_script_run_ctx and get_script_run_ctx())
    except Exception:
        return False


if __name__ == "__main__":
    # If not running under Streamlit, relaunch via `streamlit run` so the UI shows up.
    if not _has_streamlit_ctx():
        import os
        import sys
        import subprocess

        if os.environ.get("CSIO_LAUNCHED") != "1":
            os.environ["CSIO_LAUNCHED"] = "1"
            script_path = Path(__file__).resolve()
            cmd = [sys.executable, "-m", "streamlit", "run", str(script_path)]
            subprocess.run(cmd)
        else:
            make_app()
    else:
        make_app()
