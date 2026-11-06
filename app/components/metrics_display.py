"""Image analysis and evaluation metric UI components."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from app.utils.image_utils import ImageAnalysis


def _style_histogram(figure: go.Figure, title: str) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        title=title,
        title_font=dict(color="#f8fafc", size=16),
        xaxis_title="Intensity (0–255)",
        yaxis_title="Pixel count",
        bargap=0.03,
        height=300,
        margin=dict(l=12, r=12, t=48, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,16,32,0.65)",
        font=dict(color="#f8fafc"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#f8fafc"),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(38,51,77,0)",
        ),
    )
    figure.update_xaxes(
        range=[0, 255],
        gridcolor="rgba(148, 163, 184, .18)",
        zerolinecolor="rgba(148, 163, 184, .28)",
        linecolor="#26334d",
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#cbd5e1"),
    )
    figure.update_yaxes(
        gridcolor="rgba(148, 163, 184, .18)",
        zerolinecolor="rgba(148, 163, 184, .28)",
        linecolor="#26334d",
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#cbd5e1"),
    )
    return figure


def _brightness_histogram(grayscale: np.ndarray) -> go.Figure:
    counts = np.bincount(grayscale.ravel(), minlength=256)
    figure = go.Figure(
        go.Bar(
            x=np.arange(256),
            y=counts,
            marker_color="#818cf8",
            hovertemplate="Intensity: %{x}<br>Pixels: %{y}<extra></extra>",
        )
    )
    return _style_histogram(figure, "Grayscale brightness distribution")


def _rgb_histogram(rgb: np.ndarray) -> go.Figure:
    figure = go.Figure()
    colors = {
        "Red": ("#f87171", 0),
        "Green": ("#4ade80", 1),
        "Blue": ("#38bdf8", 2),
    }
    for label, (color, channel) in colors.items():
        counts = np.bincount(rgb[:, :, channel].ravel(), minlength=256)
        figure.add_trace(
            go.Scatter(
                x=np.arange(256),
                y=counts,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=f"{label}: %{{x}}<br>Pixels: %{{y}}<extra></extra>",
            )
        )
    return _style_histogram(figure, "RGB channel distribution")


def render_image_analysis(image: Image.Image, analysis: ImageAnalysis) -> None:
    """Display pixel-derived measurements and expandable input visualizations."""
    st.divider()
    st.markdown(
        '<div class="lumina-eyebrow">Image diagnostics / Input only</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Image analysis")
    brightness, contrast, sharpness = st.columns(3)
    brightness.metric(
        "Average brightness",
        f"{analysis['brightness']:.1f} / 255",
    )
    contrast.metric("Grayscale contrast", f"{analysis['contrast']:.2f}")
    sharpness.metric(
        "Sharpness (Laplacian variance)",
        f"{analysis['sharpness']:.2f}",
    )

    dimensions, noise, channels = st.columns([1, 1, 1.4])
    dimensions.metric("Image dimensions", f"{analysis['width']} x {analysis['height']} px")
    noise.metric("Noise estimate (MAD)", f"{analysis['noise_proxy']:.6f}")
    means = analysis["channel_means"]
    with channels.container(border=True, key="rgb-means"):
        st.markdown("**RGB channel means**")
        red, green, blue = st.columns(3)
        red.markdown(f"R<br><strong>{means['red']:.2f}</strong>", unsafe_allow_html=True)
        green.markdown(f"G<br><strong>{means['green']:.2f}</strong>", unsafe_allow_html=True)
        blue.markdown(f"B<br><strong>{means['blue']:.2f}</strong>", unsafe_allow_html=True)
    st.caption(
        f"Preliminary brightness assessment: **{analysis['brightness_classification']}**. "
        "Labels use grayscale-mean thresholds (<64, <128, <192, and >=192) and "
        "are heuristics, not ground truth. Sharpness is variance of the grayscale "
        "Laplacian. Noise is estimated as 1.4826 × MAD of the grayscale residual "
        "after 3 × 3 median filtering. This robust estimate can include real "
        "texture and edge detail, so it is not a noise-only measurement."
    )

    rgb = np.asarray(image)
    grayscale = np.asarray(image.convert("L"))
    with st.expander("Grayscale image"):
        original_column, grayscale_column = st.columns(2, gap="medium")
        display_width = min(image.width, 420)
        with original_column:
            st.markdown("**Original**")
            st.image(image, width=display_width)
        with grayscale_column:
            st.markdown("**Grayscale**")
            st.image(
                Image.fromarray(grayscale),
                width=display_width,
            )
    with st.expander("Brightness histogram"):
        st.plotly_chart(
            _brightness_histogram(grayscale),
            use_container_width=True,
            config={"displayModeBar": False},
            theme=None,
        )
    with st.expander("RGB histogram"):
        st.plotly_chart(
            _rgb_histogram(rgb),
            use_container_width=True,
            config={"displayModeBar": False},
            theme=None,
        )


def render_metrics() -> None:
    """Show evaluation metric slots without inventing results."""
    st.divider()
    st.markdown(
        '<div class="lumina-eyebrow">03 / Evaluation</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Evaluation metrics")
    st.caption("Values remain unavailable until the evaluation pipeline is integrated.")
    columns = st.columns(5)
    for column, label in zip(columns, ["PSNR", "SSIM", "LPIPS", "NIQE", "Latency"]):
        column.metric(label, "N/A")
