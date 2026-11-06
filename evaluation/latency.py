"""Inference latency measurement helpers."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any


def _is_cuda_device(device: Any) -> bool:
    if device is None:
        return False
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return device_type == "cuda"
    return str(device).lower().startswith("cuda")


def _synchronize_if_cuda(device: Any) -> None:
    if not _is_cuda_device(device):
        return

    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "CUDA latency synchronization requires PyTorch. "
            "Install torch or measure on CPU."
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!s} was requested, but CUDA is unavailable."
        )
    torch.cuda.synchronize(device)


def measure_restoration_latency(
    image: Any,
    restoration_fn: Callable[[Any], Any],
    *,
    warmup_runs: int = 1,
    runs: int = 10,
    device: Any = None,
) -> dict[str, float | int]:
    """Measure restoration inference time and return summary statistics.

    The restoration callable receives ``image`` once per warm-up and measured
    run; its return value is ignored and not modified. Model construction and
    weight loading should happen before calling this function. Warm-up calls
    are excluded from the reported inference times, which also excludes any
    lazy initialization performed during warm-up. Setting ``warmup_runs=0``
    explicitly includes first-call initialization in the measurements.

    When ``device`` identifies a CUDA device (for example ``"cuda"`` or
    ``"cuda:0"``), CUDA is synchronized immediately before and after each
    timed call so asynchronous kernel execution is included. Pass the actual
    device used by the restoration callable; CPU timing does not require
    PyTorch.

    Args:
        image: Input passed unchanged to ``restoration_fn``.
        restoration_fn: Callable performing one restoration inference.
        warmup_runs: Number of untimed calls before measurement; must be >= 0.
        runs: Number of timed inferences; must be >= 1.
        device: Device used by the callable. CUDA is synchronized when selected.

    Returns:
        A dictionary containing ``mean_ms``, ``median_ms``, ``min_ms``,
        ``max_ms``, and ``runs``.

    Raises:
        TypeError: If the callable or run counts have invalid types.
        ValueError: If the image is None or run counts are out of range.
        RuntimeError: If CUDA is requested but unavailable.
        Exception: Any exception raised by ``restoration_fn`` is propagated.
    """
    if image is None:
        raise ValueError("image must not be None.")
    if not callable(restoration_fn):
        raise TypeError("restoration_fn must be callable.")
    if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int):
        raise TypeError("warmup_runs must be a non-negative integer.")
    if isinstance(runs, bool) or not isinstance(runs, int):
        raise TypeError("runs must be a positive integer.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0.")
    if runs < 1:
        raise ValueError("runs must be >= 1.")

    for _ in range(warmup_runs):
        _synchronize_if_cuda(device)
        restoration_fn(image)
        _synchronize_if_cuda(device)

    durations_ms: list[float] = []
    for _ in range(runs):
        _synchronize_if_cuda(device)
        start_ns = time.perf_counter_ns()
        restoration_fn(image)
        _synchronize_if_cuda(device)
        durations_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)

    return {
        "mean_ms": statistics.fmean(durations_ms),
        "median_ms": statistics.median(durations_ms),
        "min_ms": min(durations_ms),
        "max_ms": max(durations_ms),
        "runs": runs,
    }
