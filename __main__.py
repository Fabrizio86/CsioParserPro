#!/usr/bin/env python3
"""
Module entry point for `python -m csio_explorer`.
Lauches the Streamlit UI, even when invoked directly from Python.
"""
from __future__ import annotations

from pathlib import Path


def main():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
    except Exception:
        get_script_run_ctx = None  # type: ignore

    def _has_streamlit_ctx() -> bool:
        try:
            return bool(get_script_run_ctx and get_script_run_ctx())
        except Exception:
            return False

    # Only import the app module when needed; support both package and script execution.
    app_mod = None

    if not _has_streamlit_ctx():
        import os
        import sys
        import subprocess

        if os.environ.get("CSIO_LAUNCHED") != "1":
            os.environ["CSIO_LAUNCHED"] = "1"
            # Compute the path to ui/app.py without importing it (works even when run as a bare script)
            script_path = (Path(__file__).parent / "ui" / "app.py").resolve()
            # Ensure the parent of the package dir is on PYTHONPATH so absolute imports work under Streamlit
            pkg_parent = str(Path(__file__).resolve().parent.parent)
            existing_pp = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = (pkg_parent + (os.pathsep + existing_pp if existing_pp else ""))
            cmd = [sys.executable, "-m", "streamlit", "run", str(script_path)]
            subprocess.run(cmd)
            return

    # We are in a Streamlit context already (or relaunch recursion): import the app module now
    if app_mod is None:
        try:
            # Try relative import when running as a package (python -m csio_explorer)
            from .ui import app as app_mod  # type: ignore
        except Exception:
            # Fallback for direct script execution: add package parent to sys.path and import absolutely
            import sys
            pkg_parent = str(Path(__file__).resolve().parent.parent)
            if pkg_parent not in sys.path:
                sys.path.insert(0, pkg_parent)
            from csio_explorer.ui import app as app_mod  # type: ignore

    app_mod.make_app()


if __name__ == "__main__":
    main()
