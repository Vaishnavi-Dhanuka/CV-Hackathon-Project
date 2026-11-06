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
    classical_image: Image.Image | None = None,
    deep_learning_image: Image.Image | None = None,
    hybrid_image: Image.Image | None = None,
    classical_error: str | None = None,
    deep_learning_error: str | None = None,
    hybrid_error: str | None = None,
) -> None:
    """Render available pipeline outputs and explicit errors or placeholders."""
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
        ("Official RetinexFormer", "MODEL"),
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
                elif title == "Classical CV" and classical_image is not None:
                    st.image(classical_image, use_container_width=True)
                elif title == "Classical CV" and classical_error is not None:
                    st.error(classical_error)
                elif title == "Official RetinexFormer" and deep_learning_image is not None:
                    st.image(deep_learning_image, use_container_width=True)
                elif title == "Official RetinexFormer" and deep_learning_error is not None:
                    st.error(deep_learning_error)
                elif title == "Hybrid" and hybrid_image is not None:
                    st.image(hybrid_image, use_container_width=True)
                elif title == "Hybrid" and hybrid_error is not None:
                    st.error(hybrid_error)
                else:
                    st.info("Select this pipeline to generate its comparison output.")


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
