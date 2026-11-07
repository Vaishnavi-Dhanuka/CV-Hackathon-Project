"""Image upload details, output comparisons, and pipeline artifacts."""

from collections.abc import Mapping
from typing import Any

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


def render_intermediate_visualizations(
    selected_mode: str = "Original Image",
    classical_image: Image.Image | None = None,
    classical_artifacts: Mapping[str, Image.Image] | None = None,
    classical_metadata: Mapping[str, Any] | None = None,
    deep_learning_image: Image.Image | None = None,
    hybrid_image: Image.Image | None = None,
    fusion_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Render actual outputs returned by the selected restoration pipelines."""
    st.divider()
    st.markdown(
        '<div class="lumina-eyebrow">04 / Pipeline artifacts</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Intermediate results")
    st.caption(f"Pipeline artifacts for {selected_mode}.")

    classical_selected = selected_mode in ("Classical CV", "Hybrid Intelligence")
    deep_learning_selected = selected_mode in (
        "Official RetinexFormer",
        "Deep Learning",
        "Hybrid Intelligence",
    )
    classical_stages = (
        (
            ("Illumination map", "illumination"),
            ("Reflectance", "reflectance"),
            ("Retinex output", "retinex"),
            ("CLAHE output", "clahe"),
            ("Denoised output", "denoised"),
            ("Classical CV final output", "final"),
        )
        if classical_artifacts is not None
        else ()
    )
    stages: list[tuple[str, Image.Image | None, str]] = []
    for label, key in classical_stages:
        output = classical_artifacts.get(key)
        if key == "final" and output is None:
            output = classical_image
        stages.append((label, output, "Classical CV stage was not returned."))
    if not classical_selected:
        stages.extend(
            (label, None, "Not run for the selected pipeline.")
            for label, _ in (
                ("Illumination map", "illumination"),
                ("Reflectance", "reflectance"),
                ("Retinex output", "retinex"),
                ("CLAHE output", "clahe"),
                ("Denoised output", "denoised"),
                ("Classical CV final output", "final"),
            )
        )
    elif classical_artifacts is None:
        stages.extend(
            (label, None, "Classical CV pipeline artifacts are unavailable.")
            for label, _ in (
                ("Illumination map", "illumination"),
                ("Reflectance", "reflectance"),
                ("Retinex output", "retinex"),
                ("CLAHE output", "clahe"),
                ("Denoised output", "denoised"),
                ("Classical CV final output", "final"),
            )
        )

    stages.append(
        (
            "Official RetinexFormer output",
            deep_learning_image if deep_learning_selected else None,
            (
                "The selected pipeline has not run the Official RetinexFormer API."
                if not deep_learning_selected
                else "The API did not return an enhanced image."
            ),
        )
    )
    stages.append(
        (
            "Hybrid fusion output",
            hybrid_image if selected_mode == "Hybrid Intelligence" else None,
            (
                "Hybrid fusion has not run for the selected pipeline."
                if selected_mode != "Hybrid Intelligence"
                else "Hybrid fusion did not return an output image."
            ),
        )
    )

    columns = st.columns(2, gap="medium")
    for index, (label, output, unavailable_message) in enumerate(stages):
        with columns[index % len(columns)]:
            with st.expander(label):
                if output is not None:
                    st.image(output, use_container_width=True)
                else:
                    st.info(unavailable_message)
    if classical_image is not None and selected_mode == "Hybrid Intelligence":
        with st.expander("Classical CV branch used for fusion"):
            st.image(classical_image, use_container_width=True)
    if (
        classical_selected
        and isinstance(classical_metadata, Mapping)
        and isinstance(classical_metadata.get("latency_ms"), Mapping)
    ):
        with st.expander("Classical CV pipeline metadata"):
            stage_latency = classical_metadata["latency_ms"]
            latency_summary = "; ".join(
                f"{stage}: {latency:.2f} ms"
                for stage, latency in stage_latency.items()
                if isinstance(latency, (int, float))
            )
            if latency_summary:
                st.write(f"Actual M1 stage timings — {latency_summary}.")
            config = classical_metadata.get("config")
            adaptive_rule = (
                config.get("adaptive_rule") if isinstance(config, Mapping) else None
            )
            if adaptive_rule is not None:
                st.caption(f"Adaptive Retinex selection: {adaptive_rule}")
    if (
        selected_mode == "Hybrid Intelligence"
        and isinstance(fusion_metadata, Mapping)
    ):
        baseline = fusion_metadata.get("global_baseline_weights")
        if isinstance(baseline, Mapping):
            classical_weight = baseline.get("classical_weight")
            dl_weight = baseline.get("dl_weight")
            with st.expander("Hybrid fusion metadata"):
                st.caption(
                    "Spatial contribution maps are shown in the fusion analysis above."
                )
                if isinstance(classical_weight, (int, float)) and isinstance(
                    dl_weight, (int, float)
                ):
                    st.write(
                        f"Global baseline weights — Classical CV: {classical_weight:.3f}; "
                        f"Official RetinexFormer: {dl_weight:.3f}."
                    )
