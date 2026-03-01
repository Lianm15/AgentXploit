"""CSS loading utility."""

from pathlib import Path

import streamlit as st

# Resolve paths relative to the project root (parent of this file's directory).
_ROOT = Path(__file__).parent.parent


def load_css(*relative_paths: str) -> None:
    """
    Read one or more CSS files and inject them into the Streamlit page.

    Paths are resolved relative to the project root, so callers write:
        load_css("styles/common.css", "styles/home.css")
    """
    combined = "\n".join(
        (_ROOT / path).read_text(encoding="utf-8") for path in relative_paths
    )
    st.markdown(f"<style>{combined}</style>", unsafe_allow_html=True)
