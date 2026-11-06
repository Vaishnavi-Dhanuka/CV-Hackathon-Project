"""Image upload details, output comparisons, and intermediate placeholders."""

import streamlit as st
from PIL import Image


def render_upload_details(
    filename: str,
    file_size: int,
    dimensions: tuple[int, int],
) -> None:
    """Show basic information about the validated input image."""
    with st.container(border=True, key="upload-details"):
        st.markdown(
            '<div class="lumina-eyebrow">Image received</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Input file")
        first, second, third = st.columns(3)
        first.markdown("**Filename**")
        first.caption(filename)
        second.markdown("**Dimensions**")
        second.caption(f"{dimensions[0]} × {dimensions[1]} px")
        third.markdown("**File size**")
        if file_size < 1024 * 1024:
            third.caption(f"{file_size / 1024:.1f} KB")
        else:
            third.caption(f"{file_size / (1024 * 1024):.2f} MB")


def render_comparison_results(
    image: Image.Image | None = None,
    selected_mode: str = "Classical CV",
) -> None:
    """Render the original input and explicit placeholders for future outputs."""
    st.divider()
    st.markdown(
        '<div class="lumina-eyebrow">02 / Side-by-side review</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Image comparison")
    st.markdown(
        f'<div class="lumina-section-note">Selected pipeline: <strong>{selected_mode}</strong>'
        " &nbsp;·&nbsp; Outputs are shown when available.</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(4, gap="medium")
    titles = [
        ("Original", "INPUT"),
        ("Classical CV", "CLASSICAL"),
        ("Deep Learning", "MODEL"),
        ("Hybrid", "FUSION"),
    ]

    for index, (column, (title, kicker)) in enumerate(zip(columns, titles)):
        with column:
            with st.container(border=True, key=f"comparison-card-{index}"):
                st.markdown(
                    f'<div class="lumina-card-kicker">{kicker}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"#### {title}")
                if index == 0 and image is not None:
                    st.image(image, use_container_width=True)
                elif index == 0:
                    st.info("Upload an image to preview the original.")
                else:
                    st.info("Enhancement module is not connected yet.")


def render_intermediate_visualizations() -> None:
    """Render placeholders for pipeline artifacts that are not integrated yet."""
    st.divider()
    st.markdown(
        '<div class="lumina-eyebrow">04 / Pipeline artifacts</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Intermediate results")
    st.caption("Expand a stage to check its current availability.")
    labels = [
        "Illumination map",
        "Retinex output",
        "CLAHE output",
        "Denoised output",
        "Deep learning output",
        "Hybrid fusion output",
    ]
    columns = st.columns(2, gap="medium")
    for index, label in enumerate(labels):
        with columns[index % len(columns)]:
            with st.expander(label):
                st.caption(
                    "Not available yet — the corresponding pipeline module is not connected."
                )
