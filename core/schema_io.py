#!/usr/bin/env python3
"""
csio_explorer.core.schema_io
----------------------------
Schema load/save helpers for CSIO layouts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json


def load_schema(src: Any) -> Dict[str, Any]:
    """Load schema from a Path, str path, bytes/str YAML/JSON, or a dict that already looks like schema.
    Returns a dict with keys: version (str), layouts (dict[code]-> {description, fields}).
    Each field is a dict: {name, start, length, meaning}.
    YAML is preferred; JSON is supported as a fallback when PyYAML is not available.
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
        # Try YAML first (if available), else JSON. If both fail, fall back to empty schema.
        text_stripped = text.strip()
        if not text_stripped:
            schema = {}
        else:
            try:
                import yaml  # type: ignore
                schema = yaml.safe_load(text) or {}
            except Exception:
                try:
                    schema = json.loads(text)
                except Exception:
                    schema = {}
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
    """Save schema to path. Prefer YAML; fall back to JSON if PyYAML isn't available."""
    try:
        import yaml  # type: ignore
        text = yaml.safe_dump(schema, sort_keys=False, allow_unicode=True)
    except Exception:
        text = json.dumps(schema, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


def dump_schema(schema: Dict[str, Any]) -> str:
    """Return schema as a string (YAML if possible, otherwise JSON)."""
    try:
        import yaml  # type: ignore
        return yaml.safe_dump(schema, sort_keys=False, allow_unicode=True)
    except Exception:
        return json.dumps(schema, ensure_ascii=False, indent=2)
