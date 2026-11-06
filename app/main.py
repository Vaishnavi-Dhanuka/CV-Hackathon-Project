"""Streamlit entry point for the LUMINA low-light restoration dashboard."""

from io import BytesIO

import streamlit as st

from app.components.header import render_header
from app.components.immersive_experience import render_immersive_experience
from app.components.image_display import (
    render_comparison_results,
    render_intermediate_visualizations,
    render_upload_details,
)
from app.components.metrics_display import render_image_analysis, render_metrics
from app.components.sidebar import render_sidebar
from app.utils.image_utils import load_and_analyze_image


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --lumina-background: #0b1020;
            --lumina-sidebar: #111a2e;
            --lumina-card: #151f35;
            --lumina-accent: #6366f1;
            --lumina-cyan: #38bdf8;
            --lumina-text: #f8fafc;
            --lumina-muted: #94a3b8;
            --lumina-border: #26334d;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 12% 0%, rgba(99, 102, 241, .14), transparent 34rem),
                var(--lumina-background);
            color: var(--lumina-text);
        }
        [data-testid="stAppViewContainer"] {
            background: transparent;
        }
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            background: transparent;
        }
        [data-testid="stMain"] {
            color: var(--lumina-text);
        }
        [data-testid="stSidebar"] {
            background: var(--lumina-sidebar);
            border-right: 1px solid var(--lumina-border);
            color: var(--lumina-text);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--lumina-muted);
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--lumina-text);
        }
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] {
            gap: .5rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] {
            border: 1px solid var(--lumina-border);
            border-radius: 10px;
            background: rgba(21, 31, 53, .62);
            padding: .55rem .7rem;
            transition: border-color .15s ease, background .15s ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) {
            border-color: var(--lumina-accent);
            background: rgba(99, 102, 241, .18);
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
            border-color: var(--lumina-cyan);
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] input {
            accent-color: var(--lumina-accent);
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] p {
            color: var(--lumina-text);
        }
        [data-testid="stMetric"] {
            min-width: 0;
            height: 100%;
            background: var(--lumina-card);
            border: 1px solid var(--lumina-border);
            border-radius: 14px;
            padding: 16px;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--lumina-muted);
            font-size: .82rem;
        }
        [data-testid="stMetricValue"] {
            color: var(--lumina-text);
            overflow-wrap: anywhere;
        }
        [data-testid="stVerticalBlock"][class*="st-key-upload-panel"],
        [data-testid="stVerticalBlock"][class*="st-key-upload-details"],
        [data-testid="stVerticalBlock"][class*="st-key-comparison-card-"] {
            background: rgba(21, 31, 53, .72);
            border-color: var(--lumina-border);
            border-radius: 16px;
        }
        [class*="st-key-upload-panel"] [data-testid="stFileUploader"] section {
            background: rgba(11, 16, 32, .62);
            border: 1px dashed rgba(99, 102, 241, .72);
            border-radius: 14px;
        }
        [class*="st-key-upload-panel"] [data-testid="stFileUploader"] section:hover {
            border-color: var(--lumina-cyan);
        }
        [class*="st-key-upload-panel"] [data-testid="stFileUploader"] small,
        [class*="st-key-upload-panel"] [data-testid="stFileUploader"] label {
            color: var(--lumina-muted);
        }
        [class*="st-key-upload-panel"] [data-testid="stFileUploader"] button {
            border: 1px solid var(--lumina-border);
            border-radius: 8px;
            background: #1d2942;
            color: var(--lumina-text);
        }
        [class*="st-key-upload-panel"] [data-testid="stFileUploader"] button:hover {
            border-color: var(--lumina-cyan);
            color: var(--lumina-text);
        }
        [class*="st-key-comparison-card-"] [data-testid="stImage"] img {
            width: 100%;
            height: 220px;
            object-fit: contain;
            background: #0b1020;
            border: 1px solid var(--lumina-border);
            border-radius: 10px;
        }
        .lumina-eyebrow {
            color: var(--lumina-cyan);
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
        }
        .lumina-brand {
            display: flex;
            align-items: center;
            gap: .8rem;
            margin: .2rem 0 .45rem;
        }
        .lumina-brand-mark {
            display: grid;
            width: 2.7rem;
            height: 2.7rem;
            place-items: center;
            border: 1px solid rgba(129, 140, 248, .48);
            border-radius: 13px;
            background: linear-gradient(145deg, rgba(99, 102, 241, .3), rgba(56, 189, 248, .12));
            color: #c7d2fe;
            font-size: 1.35rem;
        }
        .lumina-wordmark {
            color: var(--lumina-text);
            font-size: clamp(1.7rem, 3vw, 2.25rem);
            font-weight: 750;
            letter-spacing: .12em;
            line-height: 1.1;
        }
        .lumina-subtitle {
            max-width: 48rem;
            color: var(--lumina-muted);
            font-size: 1rem;
            line-height: 1.7;
        }
        .lumina-empty-state {
            margin-top: .75rem;
            border: 1px solid rgba(56, 189, 248, .2);
            border-radius: 12px;
            background: rgba(56, 189, 248, .06);
            padding: .9rem 1rem;
            color: var(--lumina-muted);
            line-height: 1.6;
        }
        .lumina-empty-state strong {
            color: var(--lumina-text);
        }
        .lumina-card-kicker {
            color: var(--lumina-muted);
            font-size: .7rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .lumina-section-note {
            color: var(--lumina-muted);
            font-size: .88rem;
        }
        [data-testid="stExpander"] {
            background-color: var(--lumina-card);
            border-color: var(--lumina-border);
            border-radius: 12px;
            color: var(--lumina-text);
            overflow: hidden;
            transition: background-color .18s ease, border-color .18s ease;
        }
        [data-testid="stExpander"] details {
            background-color: var(--lumina-card);
            color: var(--lumina-text);
        }
        [data-testid="stExpander"] details > summary {
            background-color: var(--lumina-card);
            color: var(--lumina-text);
        }
        [data-testid="stExpander"] details > summary * {
            color: var(--lumina-text);
        }
        [data-testid="stExpander"] details > summary {
            border-radius: 11px;
            transition: background-color .18s ease, color .18s ease;
        }
        [data-testid="stExpander"] details > summary:hover {
            background-color: #1b2942;
            color: var(--lumina-text);
        }
        [data-testid="stExpander"] details[open],
        [data-testid="stExpander"] details[open] > summary,
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background-color: var(--lumina-card);
            color: var(--lumina-text);
        }
        [data-testid="stExpander"] details > summary:focus-visible {
            outline: 2px solid var(--lumina-cyan);
            outline-offset: -2px;
        }
        [data-testid="stExpander"] details > summary svg {
            fill: currentColor;
            color: var(--lumina-text);
        }
        [data-testid="stDownloadButton"] button {
            border: 1px solid rgba(99, 102, 241, .65);
            border-radius: 10px;
            background: rgba(99, 102, 241, .16);
            color: var(--lumina-text);
        }
        [data-testid="stDownloadButton"] button:hover:not(:disabled) {
            border-color: var(--lumina-cyan);
            background: rgba(99, 102, 241, .28);
            color: var(--lumina-text);
        }
        [data-testid="stMarkdownContainer"] a {
            color: var(--lumina-cyan);
        }
        .stApp p,
        .stApp label,
        .stApp [data-testid="stCaptionContainer"] {
            line-height: 1.55;
        }
        @media (max-width: 900px) {
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: .75rem;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 calc(50% - .75rem) !important;
                min-width: min(100%, 15rem) !important;
                max-width: calc(50% - .4rem) !important;
            }
        }
        @media (max-width: 600px) {
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                flex: 1 1 100% !important;
                max-width: 100% !important;
            }
            [data-testid="stMetric"] {
                padding: 12px;
            }
            [class*="st-key-comparison-card-"] [data-testid="stImage"] img {
                height: 190px;
            }
        }
        h1, h2, h3 {
            letter-spacing: -.025em;
            color: var(--lumina-text);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="LUMINA | Low-Light Restoration",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_styles()
    render_immersive_experience()
    st.markdown('<div id="restoration-workspace"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lumina-eyebrow">Restoration workspace / Image analysis</div>',
        unsafe_allow_html=True,
    )
    render_header()

    with st.container(border=True, key="upload-panel"):
        st.markdown(
            '<div class="lumina-eyebrow">01 / Image input</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Bring your image into focus")
        st.caption("Upload a JPG, JPEG, or PNG to inspect its image quality profile.")
        uploaded_file = st.file_uploader(
            "Choose a low-light image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="uploaded_image",
        )
        if uploaded_file is None:
            st.markdown(
                '<div class="lumina-empty-state"><strong>Ready when you are.</strong> '
                "Drop a low-light image here or browse files. Your image is analyzed "
                "locally; enhancement outputs appear when their modules are connected.</div>",
                unsafe_allow_html=True,
            )

    if uploaded_file is None:
        render_sidebar(has_image=False)
        render_comparison_results()
        render_metrics()
        render_intermediate_visualizations()
        _render_download()
        return

    try:
        image, analysis = load_and_analyze_image(
            uploaded_file.getvalue(),
            uploaded_file.name,
        )
    except (ValueError, OSError) as error:
        render_sidebar(has_image=False)
        st.error(f"Could not read this image: {error}")
        return

    selected_mode = render_sidebar(has_image=True, analysis=analysis)
    render_upload_details(uploaded_file.name, uploaded_file.size, analysis["dimensions"])
    render_image_analysis(image, analysis)

    render_comparison_results(image=image, selected_mode=selected_mode)
    render_metrics()
    render_intermediate_visualizations()
    _render_download()


def _render_download() -> None:
    st.subheader("Download")
    final_output = st.session_state.get("final_enhanced_image")
    st.download_button(
        "Download final enhanced image",
        data=final_output if final_output is not None else BytesIO(),
        file_name="lumina_enhanced.png",
        mime="image/png",
        disabled=final_output is None,
        help=(
            "Available after an enhancement module produces an output."
            if final_output is None
            else None
        ),
    )
    if final_output is None:
        st.caption("Download will be enabled when an enhancement output is available.")


if __name__ == "__main__":
    main()
