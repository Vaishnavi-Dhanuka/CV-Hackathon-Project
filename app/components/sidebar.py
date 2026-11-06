"""Sidebar navigation and project context."""

import streamlit as st


def render_sidebar(has_image: bool, analysis: dict | None = None) -> str:
    """Render project details and return the selected enhancement mode."""
    with st.sidebar:
        st.markdown(
            '<div class="lumina-eyebrow">Computer vision lab</div>',
            unsafe_allow_html=True,
        )
        st.markdown("## LUMINA")
        st.caption("Low-light restoration workspace")

        st.markdown("### Project overview")
        st.write(
            "Analyze low-light images and prepare side-by-side evaluation "
            "across restoration approaches."
        )

        st.markdown("### Pipeline")
        selected_mode = st.radio(
            "Choose a pipeline",
            ["Classical CV", "Deep Learning", "Hybrid Intelligence"],
            key="enhancement_mode",
            label_visibility="collapsed",
        )

        st.markdown("### Input analysis")
        if has_image and analysis is not None:
            st.caption("Input image analyzed")
            st.write(f"Brightness: **{analysis['brightness']:.1f} / 255**")
            st.write(f"Contrast: **{analysis['contrast']:.1f}**")
            st.write(f"Size: **{analysis['dimensions'][0]} × {analysis['dimensions'][1]}**")
        else:
            st.caption("Upload an image to see its analysis here.")

        st.markdown("### About")
        st.caption("Hackathon 2026 · Robust Low-Light Image Restoration")

    return selected_mode
