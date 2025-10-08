#!/usr/bin/env python3
"""
csio_explorer.ui.app_helpers
---------------------------
UI helper functions extracted from app.py to keep the main orchestrator thin and readable.
Behavior is preserved; only code organization is improved.
"""
from __future__ import annotations

import io
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

from csio_explorer.core.constants import CODE_MEANING, BASE_COLUMNS, COMPUTED_COLUMNS
from csio_explorer.core.csio_utils import canonicalize_display_columns


def compute_display_columns(dfc: pd.DataFrame, layout_fields: List[Dict[str, Any]], show_prov_overlays: bool) -> List[str]:
    """Compute column order for decoded view.
    Keys first (level, record_code, record_len, link_ref), then schema fields (favor decoded over *_raw),
    common computed columns, any *_norm, diagnostics (hints), and pin unknown_tail/body_raw and provenance at the right.
    Then canonicalize to remove duplicates and drop non-existent placeholders.
    """
    base_cols: list[str] = list(BASE_COLUMNS)
    # Insert link_ref after base if present
    key_cols: list[str] = base_cols + (["link_ref"] if "link_ref" in dfc.columns else [])

    schema_cols = [f.get("name") for f in (layout_fields or []) if f.get("name")]
    preferred_decoded: list[str] = []
    for nm in schema_cols:
        if nm.endswith("_raw"):
            base = nm[:-4]
            preferred_decoded.append(base if base in dfc.columns else nm)
        else:
            preferred_decoded.append(nm)
    for extra in COMPUTED_COLUMNS:
        if extra in dfc.columns and extra not in preferred_decoded:
            preferred_decoded.append(extra)
    norm_cols = [c for c in dfc.columns if c.endswith("_norm")]
    display_cols: list[str] = key_cols + [c for c in preferred_decoded if c in dfc.columns] + norm_cols
    if "hints" in dfc.columns:
        display_cols.append("hints")
    # Pin diagnostics/raw at the far right
    if "unknown_tail" in dfc.columns:
        display_cols.append("unknown_tail")
    if "body_raw" in dfc.columns:
        display_cols.append("body_raw")
    elif "body" in dfc.columns:
        display_cols.append("body")
    if show_prov_overlays and "provenance" in dfc.columns:
        display_cols.append("provenance")
    # Canonicalize to strictly unique and existing columns only
    return canonicalize_display_columns(display_cols, available=list(dfc.columns))


def render_policy_summary(parsed: Dict[str, pd.DataFrame], schema: Dict[str, Any], all_df: pd.DataFrame) -> None:
    """Render the Policy Summary section anchored by BPI.policy_number.
    Uses decoded frames from the parsed dictionary.
    """
    st.subheader("Policy Summary")
    bpi_df = parsed.get("BPI")
    if bpi_df is None or getattr(bpi_df, "empty", True):
        st.info("No BPI records found to build policy summary.")
        return

    bpi_df = bpi_df.copy()
    # Robustly ensure a policy_number column exists. If schema-derived field is missing,
    # derive from the raw body using the standard BPI slice [22, length=18].
    try:
        has_policy = "policy_number" in bpi_df.columns
    except Exception:
        has_policy = False
    if not has_policy:
        from csio_explorer.core.csio_utils import safe_slice as _safe_slice
        src_col = "body_raw" if "body_raw" in bpi_df.columns else ("body" if "body" in bpi_df.columns else None)
        if src_col:
            bpi_df["policy_number"] = bpi_df[src_col].astype(str).apply(lambda s: _safe_slice(s, 22, 18).strip())
        else:
            # Create empty column to avoid KeyError downstream
            bpi_df["policy_number"] = ""

    bpi_df["policy_number_str"] = bpi_df["policy_number"].astype(str).str.strip()
    policies = sorted([p for p in bpi_df["policy_number_str"].unique().tolist() if p])
    if not policies:
        st.info("No policy numbers found in BPI.")
        return

    sel_pol = st.selectbox("Select policy", options=policies, index=0)
    pol_rows = bpi_df[bpi_df["policy_number_str"] == sel_pol]
    # Resolve policy_group_id from selected BPI row
    group_id = None
    if "policy_group_id" in pol_rows.columns and not pol_rows.empty:
        try:
            gid_val = pol_rows["policy_group_id"].iloc[0]
            group_id = int(gid_val) if pd.notna(gid_val) and str(gid_val).strip() != "" else None
        except Exception:
            group_id = None

    def by_code(code: str) -> pd.DataFrame:
        dfx = parsed.get(code)
        if dfx is None or getattr(dfx, "empty", True):
            return pd.DataFrame()
        if group_id is not None and "policy_group_id" in dfx.columns:
            try:
                return dfx[dfx.get("policy_group_id") == group_id].copy()
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    bis_df = by_code("BIS")
    n9bis_df = by_code("9BIS")
    lag_df = by_code("LAG")
    pay_df = by_code("PAY")
    cvh_df = by_code("CVH")
    sac_df = by_code("SAC")
    aoi_df = by_code("AOI")
    n9aoi_df = by_code("9AOI")

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Policy:** {sel_pol}")
        if not pol_rows.empty:
            eff = pol_rows.get("effective")
            exp = pol_rows.get("expiry")
            eff_val = eff.iloc[0] if eff is not None and not eff.empty else ""
            exp_val = exp.iloc[0] if exp is not None and not exp.empty else ""
            st.caption(f"Effective: {eff_val}  |  Expiry: {exp_val}")
    with right:
        st.caption(f"Policy group: {group_id or ''}")

    with st.expander("Insureds", expanded=True):
        if not bis_df.empty:
            st.dataframe(bis_df[[c for c in ["insured_name"] if c in bis_df.columns]], hide_index=True, use_container_width=True)
        else:
            st.info("No insureds found.")

    with st.expander("Mailing Address (9BIS)", expanded=True):
        if not n9bis_df.empty:
            cols = [c for c in ["street","city","province","postal"] if c in n9bis_df.columns]
            st.dataframe(n9bis_df[cols], hide_index=True, use_container_width=True)
        else:
            st.info("No mailing address found.")

    with st.expander("Risk Location (LAG)", expanded=True):
        if not lag_df.empty:
            cols = [c for c in ["street","city","province","postal"] if c in lag_df.columns]
            st.dataframe(lag_df[cols], hide_index=True, use_container_width=True)
        else:
            st.info("No risk location found.")

    with st.expander("Payments (PAY)", expanded=True):
        if not pay_df.empty:
            sched_rows = []
            for _, r in pay_df.iterrows():
                for i in (1, 2):
                    a = r.get(f"amount_{i}") or r.get(f"amt_{i}") or r.get(f"amount_{i}_norm")
                    d = r.get(f"date_{i}") or r.get(f"date_{i}_norm")
                    if (a and str(a).strip()) or (d and str(d).strip()):
                        sched_rows.append({"amount": a, "date": d})
            sched_df = pd.DataFrame(sched_rows)
            if not sched_df.empty:
                st.dataframe(sched_df, hide_index=True, use_container_width=True)
            else:
                st.info("No payment schedule found.")
        else:
            st.info("No payment records found.")

    with st.expander("Home Coverages (CVH)", expanded=True):
        if not cvh_df.empty:
            cols = [c for c in ["coverage_code","premium","limit","deductible","desc_tail"] if c in cvh_df.columns]
            st.dataframe(cvh_df[cols], hide_index=True, use_container_width=True)
        else:
            st.info("No home coverages found.")

    with st.expander("Auto Coverages (SAC)", expanded=True):
        if not sac_df.empty:
            cols = [c for c in ["coverage_code","premium","limit","deductible"] if c in sac_df.columns]
            st.dataframe(sac_df[cols], hide_index=True, use_container_width=True)
        else:
            st.info("No auto coverages found.")

    with st.expander("Additional Interests (AOI / 9AOI)", expanded=True):
        if not aoi_df.empty:
            st.dataframe(aoi_df[[c for c in ["interest_type","name"] if c in aoi_df.columns]], hide_index=True, use_container_width=True)
        if not n9aoi_df.empty:
            st.dataframe(n9aoi_df[[c for c in ["street","city","province","postal"] if c in n9aoi_df.columns]], hide_index=True, use_container_width=True)
        if aoi_df.empty and n9aoi_df.empty:
            st.info("No additional interests found.")

    # Export Policy to Excel
    raw_df = all_df[all_df.get("policy_group_id") == group_id] if group_id is not None and "policy_group_id" in all_df.columns else pd.DataFrame()
    exp_buf = io.BytesIO()
    with pd.ExcelWriter(exp_buf, engine="xlsxwriter") as writer:
        # Summary sheet
        summ_rows = [{
            "policy_number": sel_pol,
            "effective": (pol_rows.get("effective").iloc[0] if "effective" in pol_rows.columns and not pol_rows["effective"].empty else ""),
            "expiry": (pol_rows.get("expiry").iloc[0] if "expiry" in pol_rows.columns and not pol_rows["expiry"].empty else ""),
        }]
        pd.DataFrame(summ_rows).to_excel(writer, index=False, sheet_name="Summary")
        for nm, dfx in [
            ("BIS", bis_df),
            ("9BIS", n9bis_df),
            ("LAG", lag_df),
            ("PAY", pay_df),
            ("CVH", cvh_df),
            ("AOI_9AOI", pd.concat([aoi_df, n9aoi_df], ignore_index=True, sort=False)),
            ("raw", raw_df),
        ]:
            safe = nm[:31]
            (dfx if dfx is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name=safe)
    exp_buf.seek(0)
    st.download_button(
        "Export Policy to Excel",
        exp_buf.getvalue(),
        file_name=f"policy_{sel_pol}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def render_per_code_tabs(parsed: Dict[str, pd.DataFrame], schema: Dict[str, Any], all_df: pd.DataFrame, show_prov: bool) -> None:
    """Render record-code tabs showing decoded values for each code present."""
    counts = all_df.groupby("record_code").size().reset_index(name="count").sort_values("count", ascending=False)
    codes = [c for c in counts["record_code"].tolist()]
    tabs = st.tabs(codes)

    for tab, code in zip(tabs, codes):
        with tab:
            layout = (schema.get("layouts") or {}).get(str(code), {})
            fields = layout.get("fields") or []
            dfc = parsed.get(str(code), all_df[all_df["record_code"] == code]).copy()
            # Search filter across all visible columns
            query = st.text_input(f"Search {code}", key=f"search_{code}")
            if query:
                q = query.lower()
                dfc = dfc[dfc.apply(lambda r: any(str(v).lower().find(q) >= 0 for v in r.values), axis=1)]

            # Arrange columns based on Decoded view preferences using helper
            display_cols = compute_display_columns(dfc, fields, show_prov)
            st.dataframe(dfc[display_cols], hide_index=True, use_container_width=True)

            # Explain row inspector (first 3)
            if "explain" in dfc.columns:
                with st.expander("Explain row (first 3)", expanded=False):
                    try:
                        show = min(3, len(dfc))
                        for i in range(show):
                            row = dfc.iloc[i]
                            st.json({
                                "row_index": i,
                                "record_code": row.get("record_code"),
                                "link_ref": row.get("link_ref"),
                                "matched": (row.get("explain") or {}).get("matched") if isinstance(row.get("explain"), dict) else None,
                                "amount_tokens": (row.get("explain") or {}).get("amounts") if isinstance(row.get("explain"), dict) else None,
                                "date_tokens": (row.get("explain") or {}).get("dates") if isinstance(row.get("explain"), dict) else None,
                            })
                    except Exception as _:
                        st.caption("No explain data available for this tab.")

            # Per-tab export CSV
            safe = (str(code)[:31] or str(code)).replace("/", "_")
            st.download_button(
                f"Export {code} CSV",
                dfc.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe}.csv",
                mime="text/csv",
                key=f"csv_{code}",
            )


def render_exports_sidebar(parsed: Dict[str, pd.DataFrame], all_df: pd.DataFrame, schema: Dict[str, Any], default_schema_path) -> None:
    """Render sidebar exports: per-code CSV zip, full Excel, and JSON Lines."""
    st.sidebar.header("Exports")

    # CSV per code (zip)
    import zipfile
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # metadata sheet as text
        meta = {"schema_version": schema.get("version"), "codes": list(parsed.keys()), "code_meaning": CODE_MEANING}
        zf.writestr("_metadata.json", __import__('json').dumps(meta, ensure_ascii=False, indent=2))
        # ALL
        zf.writestr("ALL_RECORDS.csv", all_df.to_csv(index=False))
        for c in [k for k in parsed.keys() if k not in ("ALL","ALL_RECORDS")]:
            zf.writestr(f"{str(c)[:31].replace('/', '_')}.csv", parsed[c].to_csv(index=False))
    zip_buf.seek(0)
    st.sidebar.download_button("Export all CSVs (zip)", zip_buf.getvalue(), file_name="csio_per_code.zip", mime="application/zip")

    # Excel workbook
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        all_df.to_excel(writer, index=False, sheet_name="ALL")
        for c in [k for k in parsed.keys() if k not in ("ALL","ALL_RECORDS")]:
            parsed[c].to_excel(writer, index=False, sheet_name=str(c)[:31])
        # Try to embed some metadata via a META sheet
        meta_df = pd.DataFrame([{ "schema_version": schema.get("version"), "code_meaning": __import__('json').dumps(CODE_MEANING) }])
        meta_df.to_excel(writer, index=False, sheet_name="META")
    excel_buf.seek(0)
    st.sidebar.download_button(
        "Export Excel (all)",
        excel_buf.getvalue(),
        file_name="csio_all.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # JSON Lines
    import json
    jsonl = io.StringIO()
    for _, row in all_df.iterrows():
        jsonl.write(json.dumps({k: (v if not isinstance(v, dict) else v) for k, v in row.to_dict().items()}, ensure_ascii=False) + "\n")
    st.sidebar.download_button("Export JSON Lines", jsonl.getvalue().encode("utf-8"), file_name="csio.jsonl", mime="application/json")
