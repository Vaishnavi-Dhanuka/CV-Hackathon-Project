"""Paired-image evaluation runner for M3 restoration workflows.

Run with ``python -m evaluation.runner --help`` for command-line usage.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from evaluation.metrics import (
    calculate_lpips,
    calculate_niqe,
    calculate_psnr,
    calculate_ssim,
)
from hybrid.classical_client import enhance_with_classical
from hybrid.dl_client import enhance_with_api
from hybrid.fusion import spatial_adaptive_fusion

CSV_COLUMNS = (
    "image_name",
    "method",
    "psnr",
    "ssim",
    "lpips",
    "niqe",
    "latency_ms",
    "fusion_latency_ms",
    "output_height",
    "output_width",
)
METHODS = (
    "Classical CV",
    "Pretrained Retinexformer",
    "Hybrid Intelligence",
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
_ASPECT_RATIO_TOLERANCE = 1e-3


def _read_rgb(path: Path) -> tuple[Image.Image, np.ndarray]:
    """Load an image as RGB PIL and an independent uint8 NumPy array."""
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        ValueError,
    ) as error:
        raise ValueError(f"Could not read image {path}: {error}") from error
    array = np.array(image, dtype=np.uint8, copy=True)
    if array.size == 0:
        raise ValueError(f"Image {path} is empty.")
    return image, array


def _image_files(directory: Path, *, allow_missing: bool = False) -> list[Path]:
    if not directory.is_dir():
        if allow_missing:
            return []
        raise ValueError(f"Image directory does not exist: {directory}")
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(directory).as_posix().casefold(),
    )


def _validate_enhanced_image(image: Any, method: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{method} must return a NumPy array.")
    if image.dtype != np.uint8:
        raise TypeError(f"{method} must return a uint8 image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{method} must return an HxWx3 RGB image.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{method} returned an empty image.")
    return np.ascontiguousarray(image)


def _latency_ms(seconds: Any, method: str) -> float:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise TypeError(f"{method} latency must be a numeric number of seconds.")
    latency = float(seconds)
    if not math.isfinite(latency) or latency < 0:
        raise ValueError(f"{method} latency must be finite and non-negative.")
    return latency * 1000.0


def _resize_reference(
    reference: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Resize a same-aspect-ratio reference to output dimensions with Lanczos."""
    output_width, output_height = output_size
    reference_height, reference_width = reference.shape[:2]
    reference_ratio = reference_width / reference_height
    output_ratio = output_width / output_height
    if not math.isclose(
        reference_ratio,
        output_ratio,
        rel_tol=_ASPECT_RATIO_TOLERANCE,
        abs_tol=_ASPECT_RATIO_TOLERANCE,
    ):
        raise ValueError(
            "Ground-truth and enhanced image aspect ratios differ "
            f"({reference_width}x{reference_height} vs "
            f"{output_width}x{output_height}); refusing to distort the reference."
        )
    if (reference_width, reference_height) == output_size:
        return reference
    resized = Image.fromarray(reference).resize(
        output_size,
        resample=Image.Resampling.LANCZOS,
    )
    return np.array(resized, dtype=np.uint8, copy=True)


def _infer(
    method: str,
    image: Image.Image,
    input_array: np.ndarray,
) -> tuple[np.ndarray, float, float | None]:
    """Run one method; return output, total enhancement ms, and fusion ms."""
    if method == "Classical CV":
        enhanced, latency_seconds = enhance_with_classical(image)
        return (
            _validate_enhanced_image(enhanced, method),
            _latency_ms(latency_seconds, method),
            None,
        )
    if method == "Pretrained Retinexformer":
        enhanced, latency_seconds = enhance_with_api(image)
        return (
            _validate_enhanced_image(enhanced, method),
            _latency_ms(latency_seconds, method),
            None,
        )

    classical_output = None
    classical_seconds = None
    dl_output = None
    dl_seconds = None
    branch_failures: list[str] = []
    try:
        classical_output, classical_seconds = enhance_with_classical(image)
    except Exception as error:
        branch_failures.append(
            f"Classical CV failed: {type(error).__name__}: {error}"
        )
    try:
        dl_output, dl_seconds = enhance_with_api(image)
    except Exception as error:
        branch_failures.append(
            f"Deep Learning API failed: {type(error).__name__}: {error}"
        )
    if branch_failures:
        raise RuntimeError("; ".join(branch_failures))
    if classical_output is None or dl_output is None:
        raise RuntimeError("An enhancement branch returned no output.")
    if classical_seconds is None or dl_seconds is None:
        raise RuntimeError("An enhancement branch returned no latency.")

    classical_output = _validate_enhanced_image(classical_output, "Classical CV")
    dl_output = _validate_enhanced_image(dl_output, "Deep Learning API")
    if classical_output.shape != input_array.shape:
        raise ValueError("Classical output dimensions do not match the input image.")
    if dl_output.shape != input_array.shape:
        raise ValueError("Deep Learning output dimensions do not match the input image.")

    started_at = time.perf_counter()
    fused = spatial_adaptive_fusion(classical_output, dl_output, input_array)
    fusion_latency_ms = (time.perf_counter() - started_at) * 1000.0
    return (
        _validate_enhanced_image(fused, method),
        _latency_ms(classical_seconds, "Classical CV")
        + _latency_ms(dl_seconds, "Deep Learning API"),
        fusion_latency_ms,
    )


def _new_row(image_name: str, method: str) -> dict[str, str | int | float]:
    return {
        "image_name": image_name,
        "method": method,
        "psnr": "",
        "ssim": "",
        "lpips": "",
        "niqe": "",
        "latency_ms": "",
        "fusion_latency_ms": "",
        "output_height": "",
        "output_width": "",
    }


def _record_metrics(
    row: dict[str, str | int | float],
    enhanced: np.ndarray,
    reference: np.ndarray | None,
    reference_issue: str | None,
    image_name: str,
    method: str,
    failures: list[str],
) -> None:
    height, width = enhanced.shape[:2]
    row["output_height"] = height
    row["output_width"] = width

    try:
        row["niqe"] = calculate_niqe(enhanced)
    except Exception as error:
        failures.append(f"{image_name} [{method}] NIQE failed: {error}")

    if reference is None:
        failures.append(
            f"{image_name} [{method}] paired metrics unavailable: "
            f"{reference_issue or 'no matching ground-truth reference.'}"
        )
        return

    try:
        aligned_reference = _resize_reference(reference, (width, height))
    except ValueError as error:
        failures.append(f"{image_name} [{method}] paired metrics unavailable: {error}")
        return

    for metric_name, metric in (
        ("psnr", calculate_psnr),
        ("ssim", calculate_ssim),
        ("lpips", calculate_lpips),
    ):
        try:
            value = float(metric(aligned_reference, enhanced))
            if math.isnan(value):
                raise ValueError("metric returned NaN")
            row[metric_name] = value
        except Exception as error:
            failures.append(
                f"{image_name} [{method}] {metric_name.upper()} failed: {error}"
            )


def run_evaluation(
    input_dir: Path,
    reference_dir: Path,
    output_csv: Path,
    *,
    template_csv: Path | None = None,
    output_images_dir: Path | None = None,
) -> list[str]:
    """Evaluate every input with all methods and write CSV rows and failures.

    Inputs and references are paired by their relative path and filename.
    Enhanced outputs are compared at their own dimensions. A reference with a
    matching aspect ratio is resized to those dimensions using Pillow Lanczos;
    a differing aspect ratio is reported rather than distorted.

    Returns:
        A list of human-readable inference, reference, or metric failures.
        Failed values remain blank in the CSV; no metric values are fabricated.
    """
    input_dir = Path(input_dir)
    reference_dir = Path(reference_dir)
    output_csv = Path(output_csv)
    template_csv = template_csv or Path(__file__).with_name("results_template.csv")
    if output_images_dir is not None:
        output_images_dir = Path(output_images_dir)
        if output_images_dir.exists() and any(output_images_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite existing enhanced-image outputs in "
                f"{output_images_dir}."
            )
    with template_csv.open("r", newline="", encoding="utf-8") as template_file:
        template_columns = next(csv.reader(template_file), [])
    if tuple(template_columns) != CSV_COLUMNS:
        raise ValueError(
            f"Results template {template_csv} must have columns: "
            + ", ".join(CSV_COLUMNS)
        )

    inputs = _image_files(input_dir)
    reference_directory_exists = reference_dir.is_dir()
    references = {
        path.relative_to(reference_dir).as_posix().casefold(): path
        for path in _image_files(reference_dir, allow_missing=True)
    }
    if not inputs:
        raise ValueError(f"No supported input images found in {input_dir}.")

    rows: list[dict[str, str | int | float]] = []
    failures: list[str] = []
    for input_path in inputs:
        relative_name = input_path.relative_to(input_dir).as_posix()
        try:
            image, input_array = _read_rgb(input_path)
        except ValueError as error:
            for method in METHODS:
                rows.append(_new_row(relative_name, method))
                failures.append(f"{relative_name} [{method}] inference unavailable: {error}")
            continue

        reference_path = references.get(relative_name.casefold())
        reference_array: np.ndarray | None = None
        reference_issue = None
        if reference_path is not None:
            try:
                _, reference_array = _read_rgb(reference_path)
            except ValueError as error:
                reference_issue = f"ground-truth read failed: {error}"
        else:
            reference_issue = (
                f"reference directory does not exist: {reference_dir}"
                if not reference_directory_exists
                else "no matching ground-truth reference."
            )

        for method in METHODS:
            row = _new_row(relative_name, method)
            try:
                enhanced, latency_ms, fusion_latency_ms = _infer(
                    method,
                    image,
                    input_array,
                )
                if output_images_dir is not None:
                    safe_method = method.casefold().replace(" ", "_")
                    enhanced_path = (
                        output_images_dir
                        / safe_method
                        / f"{relative_name}.png"
                    )
                    enhanced_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(enhanced, mode="RGB").save(
                        enhanced_path,
                        format="PNG",
                    )
                row["latency_ms"] = latency_ms
                if fusion_latency_ms is not None:
                    row["fusion_latency_ms"] = fusion_latency_ms
                _record_metrics(
                    row,
                    enhanced,
                    reference_array,
                    reference_issue,
                    relative_name,
                    method,
                    failures,
                )
            except Exception as error:
                failures.append(
                    f"{relative_name} [{method}] inference failed: "
                    f"{type(error).__name__}: {error}"
                )
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    errors_file = output_csv.with_suffix(output_csv.suffix + ".errors.txt")
    with errors_file.open("w", encoding="utf-8") as error_file:
        if failures:
            error_file.write("\n".join(failures) + "\n")
        else:
            error_file.write("No evaluation failures.\n")
    return failures


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate M3 enhancement workflows on paired images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of low-light inputs (recursively scanned).",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Directory of ground-truth images with matching relative paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--save-outputs",
        type=Path,
        help="Optional directory for enhanced PNGs organized by method.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI, returning a nonzero status if any evaluation failed."""
    args = _build_parser().parse_args(argv)
    try:
        failures = run_evaluation(
            args.input_dir,
            args.reference_dir,
            args.output,
            output_images_dir=args.save_outputs,
        )
    except (OSError, ValueError) as error:
        print(f"Evaluation could not start: {error}", file=sys.stderr)
        return 2

    errors_file = args.output.with_suffix(args.output.suffix + ".errors.txt")
    print(f"Evaluation results written to {args.output}")
    print(f"Failure details written to {errors_file}")
    if failures:
        print(f"Completed with {len(failures)} reported issue(s):", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("All images and metrics completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
