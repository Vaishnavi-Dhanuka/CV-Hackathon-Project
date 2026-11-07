"""Streamlit entry point for the LUMINA low-light restoration dashboard."""

import hashlib
from io import BytesIO
from typing import Any, MutableMapping, TypedDict

import numpy as np
import streamlit as st
from PIL import Image

from app.components.header import render_header
from app.components.fusion_explainability import render_fusion_explainability
from app.components.immersive_experience import render_immersive_experience
from app.components.image_display import (
    render_comparison_results,
    render_intermediate_visualizations,
    render_upload_details,
)
from app.components.metrics_display import render_image_analysis, render_metrics
from app.components.sidebar import render_sidebar
from app.utils.image_utils import load_and_analyze_image
from evaluation.analysis_metrics import calculate_noise_reduction
from evaluation.metrics import calculate_niqe
from hybrid.classical_client import enhance_with_classical_artifacts
from hybrid.dl_client import API_BASE_URL, enhance_with_api
from hybrid.fusion import spatial_adaptive_fusion

_DL_CACHE_KEY = "deep_learning_result_cache"
_DL_CACHE_LIMIT = 3


class _DeepLearningResult(TypedDict):
    """Cached result and diagnostic metrics for one input/mode pair."""

    key: str
    enhanced_png: bytes | None
    latency_seconds: float | None
    niqe: float | None
    niqe_error: str | None
    noise_reduction: float | None
    noise_reduction_error: str | None
    error: str | None


class _ClassicalResult(TypedDict):
    """Cached Classical CV output and its diagnostic metrics."""

    key: str
    enhanced_png: bytes | None
    artifacts_png: dict[str, bytes] | None
    pipeline_metadata: dict[str, Any] | None
    latency_seconds: float | None
    niqe: float | None
    niqe_error: str | None
    noise_reduction: float | None
    noise_reduction_error: str | None
    error: str | None


class _HybridResult(TypedDict):
    """Cached branch and fusion outputs for one Hybrid Intelligence input."""

    key: str
    classical_png: bytes | None
    classical_artifacts_png: dict[str, bytes] | None
    classical_pipeline_metadata: dict[str, Any] | None
    dl_png: bytes | None
    hybrid_png: bytes | None
    fusion_metadata: dict[str, object] | None
    classical_latency_seconds: float | None
    dl_latency_seconds: float | None
    niqe: float | None
    niqe_error: str | None
    noise_reduction: float | None
    noise_reduction_error: str | None
    classical_error: str | None
    dl_error: str | None
    error: str | None


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
        st.session_state.pop("final_enhanced_image", None)
        render_sidebar(has_image=False)
        render_comparison_results()
        render_metrics()
        render_intermediate_visualizations()
        _render_download()
        return

    uploaded_bytes = uploaded_file.getvalue()
    try:
        image, analysis = load_and_analyze_image(
            uploaded_bytes,
            uploaded_file.name,
        )
    except (ValueError, OSError) as error:
        render_sidebar(has_image=False)
        st.error(f"Could not read this image: {error}")
        return

    selected_mode = render_sidebar(has_image=True, analysis=analysis)
    render_upload_details(uploaded_file.name, uploaded_file.size, analysis["dimensions"])
    render_image_analysis(image, analysis)

    classical_result: _ClassicalResult | None = None
    dl_result: _DeepLearningResult | None = None
    hybrid_result: _HybridResult | None = None
    if selected_mode == "Classical CV":
        classical_result = _get_classical_result(
            image,
            uploaded_bytes,
            selected_mode,
            st.session_state,
        )
        st.session_state["final_enhanced_image"] = classical_result["enhanced_png"]
    elif selected_mode in ("Official RetinexFormer", "Deep Learning"):
        dl_result = _get_deep_learning_result(
            image,
            uploaded_bytes,
            selected_mode,
            st.session_state,
        )
        st.session_state["final_enhanced_image"] = dl_result["enhanced_png"]
    elif selected_mode == "Hybrid Intelligence":
        hybrid_result = _get_hybrid_result(
            image,
            uploaded_bytes,
            selected_mode,
            st.session_state,
        )
        st.session_state["final_enhanced_image"] = hybrid_result["hybrid_png"]
    else:
        st.session_state["final_enhanced_image"] = _encode_png(
            np.asarray(image.convert("RGB"), dtype=np.uint8)
        )

    deep_learning_png = None
    if dl_result is not None:
        deep_learning_png = dl_result["enhanced_png"]
    elif hybrid_result is not None:
        deep_learning_png = hybrid_result["dl_png"]
    deep_learning_image = _decode_png(deep_learning_png)
    classical_png = (
        classical_result["enhanced_png"]
        if classical_result is not None
        else hybrid_result["classical_png"]
        if hybrid_result is not None
        else None
    )
    classical_image = _decode_png(classical_png)
    hybrid_image = (
        _decode_png(hybrid_result["hybrid_png"])
        if hybrid_result is not None
        else None
    )

    render_comparison_results(
        image=image,
        selected_mode=selected_mode,
        classical_image=classical_image,
        deep_learning_image=deep_learning_image,
        hybrid_image=hybrid_image,
        classical_error=(
            classical_result["error"]
            if classical_result is not None
            else hybrid_result["classical_error"]
            if hybrid_result is not None
            else None
        ),
        deep_learning_error=(
            dl_result["error"]
            if dl_result is not None
            else hybrid_result["dl_error"] if hybrid_result is not None else None
        ),
        hybrid_error=hybrid_result["error"] if hybrid_result is not None else None,
    )
    if (
        hybrid_result is not None
        and hybrid_result["hybrid_png"] is not None
    ):
        render_fusion_explainability(hybrid_result["fusion_metadata"], image)
    if dl_result is not None and dl_result["error"] is not None:
        if st.button(
            "Retry Official RetinexFormer enhancement",
            key=f"retry-{dl_result['key']}",
        ):
            _clear_image_cache(st.session_state, uploaded_bytes, "dl")
            st.rerun()
    if classical_result is not None and classical_result["error"] is not None:
        if st.button(
            "Retry Classical CV enhancement",
            key=f"retry-{classical_result['key']}",
        ):
            _clear_image_cache(st.session_state, uploaded_bytes)
            st.rerun()
    if hybrid_result is not None and hybrid_result["error"] is not None:
        if st.button(
            "Retry Hybrid Intelligence",
            key=f"retry-{hybrid_result['key']}",
        ):
            _clear_image_cache(st.session_state, uploaded_bytes)
            st.rerun()

    if (
        hybrid_result is None
        and classical_result is not None
        and classical_result["latency_seconds"] is not None
    ):
        st.caption(
            f"Classical CV processing latency: "
            f"{classical_result['latency_seconds']:.2f} s. "
            "Latency is not a measure of restoration quality."
        )
    if (
        hybrid_result is not None
        and hybrid_result["classical_latency_seconds"] is not None
        and hybrid_result["dl_latency_seconds"] is not None
    ):
        st.caption(
            "Branch processing latency — "
            f"Classical CV: {hybrid_result['classical_latency_seconds']:.2f} s; "
            f"Official RetinexFormer API: {hybrid_result['dl_latency_seconds']:.2f} s. "
            "Fusion time is not included; latency is not a measure of restoration quality."
        )

    if classical_result is not None:
        niqe = classical_result["niqe"]
        niqe_error = classical_result["niqe_error"]
        noise_reduction = classical_result["noise_reduction"]
        noise_reduction_error = classical_result["noise_reduction_error"]
    elif dl_result is not None:
        niqe = dl_result["niqe"]
        niqe_error = dl_result["niqe_error"]
        noise_reduction = dl_result["noise_reduction"]
        noise_reduction_error = dl_result["noise_reduction_error"]
    elif hybrid_result is not None:
        niqe = hybrid_result["niqe"]
        niqe_error = hybrid_result["niqe_error"]
        noise_reduction = hybrid_result["noise_reduction"]
        noise_reduction_error = hybrid_result["noise_reduction_error"]
    else:
        niqe = None
        niqe_error = None
        noise_reduction = None
        noise_reduction_error = None

    if classical_result is not None:
        total_latency = classical_result["latency_seconds"]
        latency_label = "Classical CV latency"
    elif dl_result is not None:
        total_latency = dl_result["latency_seconds"]
        latency_label = "RetinexFormer latency"
    elif hybrid_result is not None:
        total_latency = (
            hybrid_result["classical_latency_seconds"]
            + hybrid_result["dl_latency_seconds"]
            if hybrid_result["classical_latency_seconds"] is not None
            and hybrid_result["dl_latency_seconds"] is not None
            else None
        )
        latency_label = "Combined enhancement time"
    else:
        try:
            niqe = calculate_niqe(np.asarray(image.convert("RGB"), dtype=np.uint8))
            niqe_error = None
        except Exception as error:
            niqe = None
            niqe_error = f"{type(error).__name__}: {error}"
        noise_reduction = None
        noise_reduction_error = None
        total_latency = None
        latency_label = "Latency"
    render_metrics(
        niqe=niqe,
        niqe_error=niqe_error,
        noise_reduction=noise_reduction,
        noise_reduction_error=noise_reduction_error,
        latency_seconds=total_latency,
        latency_label=latency_label,
    )
    artifact_pngs = (
        classical_result["artifacts_png"]
        if classical_result is not None
        else hybrid_result["classical_artifacts_png"]
        if hybrid_result is not None
        else None
    )
    render_intermediate_visualizations(
        selected_mode=selected_mode,
        classical_image=classical_image,
        classical_artifacts=_decode_png_artifacts(artifact_pngs),
        classical_metadata=(
            classical_result["pipeline_metadata"]
            if classical_result is not None
            else hybrid_result["classical_pipeline_metadata"]
            if hybrid_result is not None
            else None
        ),
        deep_learning_image=deep_learning_image,
        hybrid_image=hybrid_image,
        fusion_metadata=(
            hybrid_result["fusion_metadata"]
            if hybrid_result is not None
            else None
        ),
    )
    _render_download()


def _cached_branch_result(
    cache: dict[str, Any],
    image_digest: str,
    branch: str,
) -> tuple[bytes | None, float | None, str | None] | None:
    """Find a same-image branch result stored by either standalone or hybrid mode."""
    for cache_key, cached in reversed(tuple(cache.items())):
        if not cache_key.endswith(f":{image_digest}") or not isinstance(cached, dict):
            continue
        if branch == "classical":
            if "classical_png" in cached:
                png = cached["classical_png"]
                latency = cached.get("classical_latency_seconds")
                error = cached.get("classical_error")
            elif cached.get("key", "").split(":", maxsplit=1)[0] == "Classical CV":
                png = cached.get("enhanced_png")
                latency = cached.get("latency_seconds")
                error = cached.get("error")
            else:
                continue
        else:
            if "dl_png" in cached:
                png = cached["dl_png"]
                latency = cached.get("dl_latency_seconds")
                error = cached.get("dl_error")
            elif cached.get("key", "").split(":", maxsplit=1)[0] in (
                "Deep Learning",
                "Official RetinexFormer",
            ):
                png = cached.get("enhanced_png")
                latency = cached.get("latency_seconds")
                error = cached.get("error")
            else:
                continue
        if png is not None or error is not None:
            return png, latency, error
    return None


def _cached_classical_artifacts(
    cache: dict[str, Any],
    image_digest: str,
) -> tuple[dict[str, bytes] | None, dict[str, Any] | None]:
    """Find actual M1 stage images and metadata for a same-image branch."""
    for cache_key, cached in reversed(tuple(cache.items())):
        if not cache_key.endswith(f":{image_digest}") or not isinstance(cached, dict):
            continue
        mode = cached.get("key", "").split(":", maxsplit=1)[0]
        if "classical_png" in cached:
            artifacts = cached.get("classical_artifacts_png")
            metadata = cached.get("classical_pipeline_metadata")
        elif mode == "Classical CV":
            artifacts = cached.get("artifacts_png")
            metadata = cached.get("pipeline_metadata")
        else:
            continue
        if isinstance(artifacts, dict):
            return artifacts, metadata if isinstance(metadata, dict) else None
    return None, None


def _clear_image_cache(
    session_state: MutableMapping[str, Any],
    uploaded_bytes: bytes,
    branch: str | None = None,
) -> None:
    """Clear cached results for an image, optionally including one branch."""
    cache = session_state.get(_DL_CACHE_KEY)
    if not isinstance(cache, dict):
        return
    digest = hashlib.sha256(uploaded_bytes).hexdigest()
    for cache_key, cached in tuple(cache.items()):
        if not cache_key.endswith(f":{digest}") or not isinstance(cached, dict):
            continue
        if branch == "dl":
            is_dl_result = "dl_png" in cached or (
                "enhanced_png" in cached
                and cached.get("key", "").split(":", maxsplit=1)[0]
                in ("Deep Learning", "Official RetinexFormer")
            )
            if not is_dl_result:
                continue
        cache.pop(cache_key, None)


def _get_classical_result(
    image: Image.Image,
    uploaded_bytes: bytes,
    selected_mode: str,
    session_state: MutableMapping[str, Any],
) -> _ClassicalResult:
    """Return a cached M1 result or run the Classical CV adapter once."""
    digest = hashlib.sha256(uploaded_bytes).hexdigest()
    cache_key = f"{selected_mode}:{digest}"
    cache = session_state.get(_DL_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        session_state[_DL_CACHE_KEY] = cache
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and "artifacts_png" in cached:
        return cached
    cache.pop(cache_key, None)

    result: _ClassicalResult = {
        "key": cache_key,
        "enhanced_png": None,
        "artifacts_png": None,
        "pipeline_metadata": None,
        "latency_seconds": None,
        "niqe": None,
        "niqe_error": None,
        "noise_reduction": None,
        "noise_reduction_error": None,
        "error": None,
    }
    branch = _cached_branch_result(cache, digest, "classical")
    artifacts_png, pipeline_metadata = _cached_classical_artifacts(cache, digest)
    try:
        if branch is not None and artifacts_png is not None:
            png, latency, error_message = branch
            result["enhanced_png"] = png
            result["artifacts_png"] = artifacts_png
            result["pipeline_metadata"] = pipeline_metadata
            result["latency_seconds"] = latency
            result["error"] = error_message
            if png is None:
                raise RuntimeError(error_message or "Cached Classical CV output is missing.")
            enhanced_array = np.asarray(_decode_png(png), dtype=np.uint8)
        else:
            with st.spinner("Enhancing image with Classical CV..."):
                (
                    enhanced_array,
                    latency_seconds,
                    artifacts,
                    pipeline_metadata,
                ) = enhance_with_classical_artifacts(image)
            result["latency_seconds"] = latency_seconds
            result["enhanced_png"] = _encode_png(enhanced_array)
            result["artifacts_png"] = _encode_png_artifacts(artifacts)
            result["pipeline_metadata"] = pipeline_metadata

        input_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        try:
            result["niqe"] = calculate_niqe(enhanced_array)
        except Exception as error:
            result["niqe_error"] = f"{type(error).__name__}: {error}"
        try:
            result["noise_reduction"] = calculate_noise_reduction(
                input_array,
                enhanced_array,
            )
        except Exception as error:
            result["noise_reduction_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        if result["error"] is None:
            result["error"] = f"Classical CV enhancement failed: {error}"

    cache[cache_key] = result
    while len(cache) > _DL_CACHE_LIMIT:
        del cache[next(iter(cache))]
    return result


def _get_deep_learning_result(
    image: Image.Image,
    uploaded_bytes: bytes,
    selected_mode: str,
    session_state: MutableMapping[str, Any],
) -> _DeepLearningResult:
    """Return a cached result or call the API once for this image and mode."""
    digest = hashlib.sha256(uploaded_bytes).hexdigest()
    cache_key = f"{selected_mode}:{digest}"
    cache = session_state.get(_DL_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        session_state[_DL_CACHE_KEY] = cache
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return cached

    result: _DeepLearningResult = {
        "key": cache_key,
        "enhanced_png": None,
        "latency_seconds": None,
        "niqe": None,
        "niqe_error": None,
        "noise_reduction": None,
        "noise_reduction_error": None,
        "error": None,
    }

    branch = _cached_branch_result(cache, digest, "dl")
    try:
        if branch is not None:
            png, latency_seconds, error_message = branch
            result["enhanced_png"] = png
            result["latency_seconds"] = latency_seconds
            result["error"] = error_message
            if png is None:
                raise RuntimeError(error_message or "Cached RetinexFormer output is missing.")
            enhanced_array = np.asarray(_decode_png(png), dtype=np.uint8)
        else:
            with st.spinner("Enhancing image with the Official RetinexFormer API..."):
                enhanced_array, latency_seconds = enhance_with_api(image)
            result["latency_seconds"] = latency_seconds
            result["enhanced_png"] = _encode_png(enhanced_array)

        input_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        try:
            result["niqe"] = calculate_niqe(enhanced_array)
        except Exception as error:
            result["niqe_error"] = f"{type(error).__name__}: {error}"
        try:
            result["noise_reduction"] = calculate_noise_reduction(
                input_array,
                enhanced_array,
            )
        except Exception as error:
            result["noise_reduction_error"] = f"{type(error).__name__}: {error}"
    except (ConnectionError, TimeoutError) as error:
        result["error"] = (
            "The official RetinexFormer API is unavailable or did not respond. "
            f"Confirm the M2 server is running at {API_BASE_URL}. Details: {error}"
        )
    except Exception as error:
        if result["error"] is None:
            result["error"] = f"RetinexFormer enhancement failed: {error}"

    cache[cache_key] = result
    while len(cache) > _DL_CACHE_LIMIT:
        del cache[next(iter(cache))]
    return result


def _decode_png(image_bytes: bytes | None) -> Image.Image | None:
    """Decode cached PNG bytes as an independent RGB Pillow image."""
    if image_bytes is None:
        return None
    with Image.open(BytesIO(image_bytes)) as result_image:
        return result_image.convert("RGB")


def _get_hybrid_result(
    image: Image.Image,
    uploaded_bytes: bytes,
    selected_mode: str,
    session_state: MutableMapping[str, Any],
) -> _HybridResult:
    """Enhance through both branches, then fuse; cache by mode and input bytes."""
    digest = hashlib.sha256(uploaded_bytes).hexdigest()
    cache_key = f"{selected_mode}:{digest}"
    cache = session_state.get(_DL_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        session_state[_DL_CACHE_KEY] = cache
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and "classical_artifacts_png" in cached:
        return cached
    cache.pop(cache_key, None)

    result: _HybridResult = {
        "key": cache_key,
        "classical_png": None,
        "classical_artifacts_png": None,
        "classical_pipeline_metadata": None,
        "dl_png": None,
        "hybrid_png": None,
        "fusion_metadata": None,
        "classical_latency_seconds": None,
        "dl_latency_seconds": None,
        "niqe": None,
        "niqe_error": None,
        "noise_reduction": None,
        "noise_reduction_error": None,
        "classical_error": None,
        "dl_error": None,
        "error": None,
    }

    classical_array: np.ndarray | None = None
    dl_array: np.ndarray | None = None
    with st.spinner("Running Classical CV and Deep Learning enhancement..."):
        cached_classical = _cached_branch_result(cache, digest, "classical")
        classical_artifacts_png, classical_pipeline_metadata = (
            _cached_classical_artifacts(cache, digest)
        )
        if cached_classical is not None and classical_artifacts_png is not None:
            png, result["classical_latency_seconds"], error_message = cached_classical
            result["classical_png"] = png
            result["classical_artifacts_png"] = classical_artifacts_png
            result["classical_pipeline_metadata"] = classical_pipeline_metadata
            result["classical_error"] = error_message
            if png is not None:
                try:
                    classical_array = np.asarray(_decode_png(png), dtype=np.uint8)
                except Exception as error:
                    result["classical_error"] = (
                        f"Cached Classical CV image is invalid: {error}"
                    )
        else:
            try:
                (
                    classical_array,
                    result["classical_latency_seconds"],
                    classical_artifacts,
                    result["classical_pipeline_metadata"],
                ) = enhance_with_classical_artifacts(image)
                result["classical_png"] = _encode_png(classical_array)
                result["classical_artifacts_png"] = _encode_png_artifacts(
                    classical_artifacts
                )
            except Exception as error:
                result["classical_error"] = (
                    f"Classical CV enhancement failed: {type(error).__name__}: {error}"
                )

        cached_dl = _cached_branch_result(cache, digest, "dl")
        if cached_dl is not None:
            png, result["dl_latency_seconds"], error_message = cached_dl
            result["dl_png"] = png
            result["dl_error"] = error_message
            if png is not None:
                try:
                    dl_array = np.asarray(_decode_png(png), dtype=np.uint8)
                except Exception as error:
                    result["dl_error"] = f"Cached RetinexFormer image is invalid: {error}"
        else:
            try:
                dl_array, result["dl_latency_seconds"] = enhance_with_api(image)
                result["dl_png"] = _encode_png(dl_array)
            except (ConnectionError, TimeoutError) as error:
                result["dl_error"] = (
                    "Official RetinexFormer API unavailable at "
                    f"{API_BASE_URL}: {type(error).__name__}: {error}"
                )
            except Exception as error:
                result["dl_error"] = (
                    f"Official RetinexFormer enhancement failed: "
                    f"{type(error).__name__}: {error}"
                )

        if result["classical_error"] is None and result["dl_error"] is None:
            if classical_array is None or dl_array is None:
                result["error"] = (
                    "Hybrid output unavailable because an enhancement branch "
                    "returned no image."
                )
            else:
                result["error"] = _fuse_hybrid_output(
                    image,
                    classical_array,
                    dl_array,
                    result,
                )
        else:
            failures = [
                message
                for message in (
                    result["classical_error"],
                    result["dl_error"],
                )
                if message is not None
            ]
            result["error"] = (
                "Hybrid output unavailable because an enhancement branch failed: "
                + " ".join(failures)
            )

    cache[cache_key] = result
    while len(cache) > _DL_CACHE_LIMIT:
        del cache[next(iter(cache))]
    return result


def _fuse_hybrid_output(
    image: Image.Image,
    classical_array: np.ndarray,
    dl_array: np.ndarray,
    result: _HybridResult,
) -> str | None:
    """Fuse successful outputs and update cached image-quality statistics."""
    try:
        input_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        fusion_result = spatial_adaptive_fusion(
            classical_array,
            dl_array,
            input_array,
            return_weights=True,
        )
        if (
            not isinstance(fusion_result, tuple)
            or len(fusion_result) != 2
        ):
            raise TypeError(
                "Fusion with return_weights=True must return an image and metadata."
            )
        fused_array, fusion_metadata = fusion_result
        if not isinstance(fused_array, np.ndarray):
            raise TypeError("Fusion must return a NumPy array.")
        if fused_array.dtype != np.uint8:
            raise TypeError("Fusion output must have dtype uint8.")
        if fused_array.shape != input_array.shape:
            raise ValueError("Fusion output must match the input HxWx3 dimensions.")

        result["hybrid_png"] = _encode_png(fused_array)
        if isinstance(fusion_metadata, dict):
            result["fusion_metadata"] = fusion_metadata
    except Exception as error:
        return f"Hybrid fusion failed: {type(error).__name__}: {error}"

    try:
        result["niqe"] = calculate_niqe(fused_array)
    except Exception as error:
        result["niqe_error"] = f"{type(error).__name__}: {error}"
    try:
        result["noise_reduction"] = calculate_noise_reduction(
            input_array,
            fused_array,
        )
    except Exception as error:
        result["noise_reduction_error"] = f"{type(error).__name__}: {error}"
    return None


def _encode_png(image: np.ndarray) -> bytes:
    """Encode an RGB NumPy image as PNG bytes for display and download."""
    output = BytesIO()
    Image.fromarray(image).convert("RGB").save(output, format="PNG")
    return output.getvalue()


def _encode_png_artifacts(artifacts: dict[str, np.ndarray]) -> dict[str, bytes]:
    """Encode actual M1 stage images for safe session-state caching."""
    return {
        name: _encode_png(image)
        for name, image in artifacts.items()
        if name != "final"
    }


def _decode_png_artifacts(
    artifacts: dict[str, bytes] | None,
) -> dict[str, Image.Image] | None:
    """Decode cached M1 stage images as independent RGB Pillow images."""
    if artifacts is None:
        return None
    return {
        name: image
        for name, png in artifacts.items()
        if (image := _decode_png(png)) is not None
    }


def _render_download() -> None:
    st.subheader("Download")
    final_output = st.session_state.get("final_enhanced_image")
    is_original = st.session_state.get("enhancement_mode") == "Original Image"
    st.download_button(
        "Download selected output as PNG",
        data=final_output if final_output is not None else BytesIO(),
        file_name="lumina_original.png" if is_original else "lumina_enhanced.png",
        mime="image/png",
        disabled=final_output is None,
        help=(
            "Available after an image or enhancement output is selected."
            if final_output is None
            else None
        ),
    )
    if final_output is None:
        st.caption("Download will be enabled when an enhancement output is available.")


if __name__ == "__main__":
    main()
