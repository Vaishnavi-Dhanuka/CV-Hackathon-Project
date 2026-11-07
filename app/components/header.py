"""Header component for the LUMINA dashboard."""

import streamlit as st


def render_header() -> None:
    """Render the application name, subtitle, and project summary."""
    st.markdown(
        """
        <div class="lumina-eyebrow">Computer vision lab&nbsp; / &nbsp;AI-assisted restoration</div>
        <div class="lumina-brand">
            <div class="lumina-brand-mark" aria-hidden="true">✦</div>
            <div class="lumina-wordmark">LUMINA</div>
        </div>
        <h1>Low-light image intelligence</h1>
        <div class="lumina-subtitle">
            Inspect image quality and compare classical, deep learning, and hybrid
            restoration approaches in one focused workspace.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
