"""Visualization of the spatial heuristic maps used by Hybrid fusion."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

_MAX_DISPLAY_SIDE = 512


def validate_fusion_metadata(
    metadata: Mapping[str, Any] | None,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Validate fusion maps and baseline weights without rescaling them.

    The maps are fractional weights, expected in [0, 1] and complementary at
    each pixel. Values are retained exactly because min-max normalization
    would misrepresent the actual fusion contribution.

    Raises:
        ValueError: If metadata is absent, malformed, non-finite, out of range,
            non-complementary, or dimensionally inconsistent.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("Hybrid fusion metadata is unavailable.")

    classical_map = metadata.get("classical_weight_map")
    dl_map = metadata.get("dl_weight_map")
    baseline = metadata.get("global_baseline_weights")
    if not isinstance(classical_map, np.ndarray) or not isinstance(dl_map, np.ndarray):
        raise ValueError("Fusion metadata must contain both NumPy weight maps.")
    if classical_map.ndim != 2 or dl_map.ndim != 2:
        raise ValueError("Fusion weight maps must be two-dimensional.")
    if classical_map.shape != dl_map.shape:
        raise ValueError("Classical and DL fusion weight maps must have equal shapes.")
    if classical_map.size == 0:
        raise ValueError("Fusion weight maps must be non-empty.")
    if expected_shape is not None and classical_map.shape != expected_shape:
        raise ValueError(
            "Fusion weight-map dimensions must match the original image "
            f"({classical_map.shape} != {expected_shape})."
        )
    if not (
        np.issubdtype(classical_map.dtype, np.integer)
        or np.issubdtype(classical_map.dtype, np.floating)
    ) or not (
        np.issubdtype(dl_map.dtype, np.integer)
        or np.issubdtype(dl_map.dtype, np.floating)
    ):
        raise ValueError("Fusion weight maps must contain numeric values.")
    if not np.all(np.isfinite(classical_map)) or not np.all(np.isfinite(dl_map)):
        raise ValueError("Fusion weight maps must contain only finite values.")
    if (
        np.any(classical_map < 0)
        or np.any(classical_map > 1)
        or np.any(dl_map < 0)
        or np.any(dl_map > 1)
    ):
        raise ValueError("Fusion weight maps must be normalized to [0, 1].")
    if not np.allclose(classical_map + dl_map, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("Classical and DL weight maps must sum to one per pixel.")
    if not isinstance(baseline, Mapping):
        raise ValueError("Fusion metadata is missing global baseline weights.")

    classical_weight = baseline.get("classical_weight")
    dl_weight = baseline.get("dl_weight")
    for name, value in (
        ("classical_weight", classical_weight),
        ("dl_weight", dl_weight),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Global baseline {name} must be finite and in [0, 1].")
    if not np.isclose(classical_weight + dl_weight, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("Global baseline weights must sum to one.")

    return (
        np.asarray(classical_map, dtype=np.float32),
        np.asarray(dl_map, dtype=np.float32),
        float(classical_weight),
        float(dl_weight),
    )


def _maps_for_display(
    classical_map: np.ndarray,
    dl_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample large maps for rendering while preserving fractional values."""
    height, width = classical_map.shape
    longest_side = max(height, width)
    if longest_side <= _MAX_DISPLAY_SIDE:
        return classical_map, dl_map

    scale = _MAX_DISPLAY_SIDE / longest_side
    display_size = (max(1, round(width * scale)), max(1, round(height * scale)))

    def resize(weight_map: np.ndarray) -> np.ndarray:
        return np.asarray(
            Image.fromarray(weight_map.astype(np.float32, copy=False)).resize(
                display_size,
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )

    return resize(classical_map), resize(dl_map)


def average_spatial_contributions(
    classical_map: np.ndarray,
    dl_map: np.ndarray,
) -> tuple[float, float]:
    """Return image-wide mean weights, preserving their normalized sum."""
    if (
        not isinstance(classical_map, np.ndarray)
        or not isinstance(dl_map, np.ndarray)
        or classical_map.ndim != 2
        or dl_map.ndim != 2
        or classical_map.shape != dl_map.shape
        or classical_map.size == 0
    ):
        raise ValueError("Spatial contribution maps must be matching non-empty arrays.")
    if (
        not np.all(np.isfinite(classical_map))
        or not np.all(np.isfinite(dl_map))
        or not np.allclose(classical_map + dl_map, 1.0, rtol=0.0, atol=1e-5)
    ):
        raise ValueError("Spatial contribution maps must be finite and sum to one.")
    return float(np.mean(classical_map)), float(np.mean(dl_map))


def build_weight_map_figure(
    classical_map: np.ndarray,
    dl_map: np.ndarray,
) -> go.Figure:
    """Build side-by-side fractional contribution maps with fixed [0, 1] scale."""
    classical_map, dl_map = _maps_for_display(classical_map, dl_map)
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Classical CV weight", "Deep Learning weight"),
        horizontal_spacing=0.12,
    )
    figure.add_trace(
        go.Heatmap(
            z=classical_map,
            zmin=0,
            zmax=1,
            colorscale="Oranges",
            colorbar=dict(title="Weight", x=0.44, len=0.85),
            hovertemplate="x=%{x}<br>y=%{y}<br>Classical weight=%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            z=dl_map,
            zmin=0,
            zmax=1,
            colorscale="Blues",
            colorbar=dict(title="Weight", x=1.02, len=0.85),
            hovertemplate="x=%{x}<br>y=%{y}<br>DL weight=%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        title="Spatial heuristic fusion contribution weights (display preview)",
        template="plotly_dark",
        height=420,
        margin=dict(l=12, r=80, t=70, b=35),
    )
    figure.update_xaxes(title_text="Map column")
    figure.update_yaxes(title_text="Map row", autorange="reversed")
    return figure


def build_contribution_figure(
    classical_map: np.ndarray,
    dl_map: np.ndarray,
) -> go.Figure:
    """Build a centered map where positive means more Classical contribution."""
    classical_map, dl_map = _maps_for_display(classical_map, dl_map)
    contribution = classical_map - dl_map
    figure = go.Figure(
        go.Heatmap(
            z=contribution,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu",
            colorbar=dict(
                title="Classical − DL<br>weight",
                tickvals=[-1, 0, 1],
                ticktext=["More DL", "Equal", "More Classical"],
            ),
            hovertemplate="x=%{x}<br>y=%{y}<br>Classical − DL=%{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Relative heuristic contribution (not model attention)",
        template="plotly_dark",
        height=420,
        margin=dict(l=12, r=80, t=70, b=35),
    )
    figure.update_xaxes(title_text="Map column")
    figure.update_yaxes(title_text="Map row", autorange="reversed")
    return figure


def build_overlay_image(
    image: Image.Image,
    classical_map: np.ndarray,
    dl_map: np.ndarray,
    opacity: float = 0.45,
) -> np.ndarray:
    """Blend a blue-to-red contribution tint over the original RGB image."""
    if not isinstance(image, Image.Image):
        raise ValueError("Overlay source must be a PIL image.")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("Overlay opacity must be in [0, 1].")
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    if original.shape[:2] != classical_map.shape or classical_map.shape != dl_map.shape:
        raise ValueError("Overlay image and contribution maps must have matching dimensions.")

    contribution = np.clip((classical_map - dl_map + 1.0) / 2.0, 0.0, 1.0)
    tint = np.empty_like(original)
    tint[..., 0] = 255.0 * contribution
    tint[..., 1] = 48.0
    tint[..., 2] = 255.0 * (1.0 - contribution)
    blend_alpha = (opacity * np.abs(classical_map - dl_map))[..., None]
    return np.ascontiguousarray(
        np.clip(original * (1.0 - blend_alpha) + tint * blend_alpha, 0, 255),
        dtype=np.uint8,
    )


def render_fusion_explainability(
    metadata: Mapping[str, Any] | None,
    image: Image.Image,
) -> None:
    """Render validated Hybrid weights, baseline, legend, and optional overlay."""
    st.divider()
    st.markdown(
        '<div class="lumina-eyebrow">Hybrid fusion / Contribution maps</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Fusion explainability")
    try:
        classical_map, dl_map, classical_weight, dl_weight = validate_fusion_metadata(
            metadata,
            expected_shape=(image.height, image.width),
        )
    except ValueError as error:
        st.info(f"Fusion contribution maps unavailable: {error}")
        return

    mean_classical, mean_dl = average_spatial_contributions(classical_map, dl_map)
    contribution_columns = st.columns(4)
    contribution_columns[0].metric(
        "Mean spatial · Classical CV",
        f"{mean_classical:.1%}",
    )
    contribution_columns[1].metric(
        "Mean spatial · Deep Learning",
        f"{mean_dl:.1%}",
    )
    contribution_columns[2].metric(
        "Global baseline · Classical CV",
        f"{classical_weight:.1%}",
    )
    contribution_columns[3].metric(
        "Global baseline · Deep Learning",
        f"{dl_weight:.1%}",
    )
    st.caption(
        "Mean spatial contributions average the actual per-pixel fusion weights; "
        "each pair sums to 100%. These proportions describe the blend, not output quality."
    )
    statistics = metadata.get("input_statistics")
    if isinstance(statistics, Mapping):
        mean_luminance = statistics.get("mean_luminance")
        contrast = statistics.get("luminance_contrast")
        underexposed_fraction = statistics.get("underexposed_fraction")
        values = (mean_luminance, contrast, underexposed_fraction)
        if all(
            isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(value)
            for value in values
        ):
            st.caption(
                "Weight-selection context: the global prior uses input mean "
                f"luminance ({float(mean_luminance):.1f}/255), contrast "
                f"({float(contrast):.1f}), and underexposed-pixel fraction "
                f"({float(underexposed_fraction):.1%}); local weights then "
                "adjust from smoothed input illumination. No branch confidence "
                "or ground truth is used."
            )
    st.plotly_chart(
        build_weight_map_figure(classical_map, dl_map),
        use_container_width=True,
        config={"displayModeBar": False},
        theme=None,
    )
    st.plotly_chart(
        build_contribution_figure(classical_map, dl_map),
        use_container_width=True,
        config={"displayModeBar": False},
        theme=None,
    )
    show_overlay = st.checkbox(
        "Overlay contribution map on original image",
        value=False,
        key="show_fusion_weight_overlay",
    )
    if show_overlay:
        st.image(
            build_overlay_image(image, classical_map, dl_map),
            caption="Red: greater Classical CV contribution · Blue: greater Deep Learning contribution",
            use_container_width=True,
        )
    st.info(
        "Higher Classical CV weight means greater Classical CV contribution. "
        "Higher DL weight means greater Deep Learning contribution. These maps "
        "are heuristic fusion contribution weights, not model attention or "
        "learned explainability."
    )
