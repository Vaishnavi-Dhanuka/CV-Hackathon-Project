"""Reproducible, paired validation experiments for M3 fusion strategies.

Run with ``python -m evaluation.fusion_experiments --help``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import UnidentifiedImageError
from skimage.color import deltaE_ciede2000, rgb2lab

from evaluation.metrics import (
    calculate_lpips,
    calculate_niqe,
    calculate_psnr,
    calculate_ssim,
)
from evaluation.runner import (
    CSV_COLUMNS,
    IMAGE_SUFFIXES,
    _read_rgb,
    _resize_reference,
    _validate_enhanced_image,
)
from hybrid.classical_client import enhance_with_classical
from hybrid.dl_client import enhance_with_api
from hybrid.fusion import adaptive_fusion, spatial_adaptive_fusion, weighted_fusion

SEED = 2026
VALIDATION_FRACTION = 0.2
METHODS = (
    "Classical CV",
    "Pretrained Retinexformer",
    "Weighted 50:50",
    "Adaptive fusion",
    "Spatial-adaptive fusion",
    "DL-dominant 90:10",
    "DL-dominant 95:5",
    "DL-dominant 80:20",
)
_PAIRED_QUALITY_METRICS = ("psnr", "ssim", "lpips")


def create_split_manifest(
    input_dir: Path,
    reference_dir: Path,
    manifest_path: Path,
    *,
    seed: int = SEED,
    validation_fraction: float = VALIDATION_FRACTION,
) -> dict[str, Any]:
    """Write a seeded, filename-paired train/validation manifest.

    Both paths are relative to their supplied dataset directories. The split
    uses only ``our485``; callers must not pass the held-out ``eval15`` split.
    """
    input_dir = Path(input_dir)
    reference_dir = Path(reference_dir)
    manifest_path = Path(manifest_path)
    if "eval15" in {part.casefold() for part in input_dir.parts}:
        raise ValueError("The validation split must not be created from eval15.")
    if not input_dir.is_dir() or not reference_dir.is_dir():
        raise ValueError("Both input and reference directories must exist.")

    def files_by_name(directory: Path) -> dict[str, Path]:
        return {
            path.relative_to(directory).as_posix().casefold(): path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        }

    inputs = files_by_name(input_dir)
    references = files_by_name(reference_dir)
    if not inputs:
        raise ValueError(f"No supported input images found in {input_dir}.")
    if inputs.keys() != references.keys():
        missing = sorted(inputs.keys() - references.keys())
        extra = sorted(references.keys() - inputs.keys())
        raise ValueError(
            f"Input/reference pairs do not match; missing references={missing[:5]}, "
            f"extra references={extra[:5]}."
        )
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 < validation_fraction < 1
    ):
        raise ValueError("Use an integer seed and validation_fraction between 0 and 1.")

    names = sorted(
        (path.relative_to(input_dir).as_posix() for path in inputs.values()),
        key=str.casefold,
    )
    random.Random(seed).shuffle(names)
    validation_count = max(1, round(len(names) * validation_fraction))
    validation_names = sorted(names[:validation_count], key=str.casefold)
    training_names = sorted(names[validation_count:], key=str.casefold)
    manifest: dict[str, Any] = {
        "dataset": "LOLv2 our485",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "input_directory": str(input_dir),
        "reference_directory": str(reference_dir),
        "training_images": training_names,
        "validation_images": validation_names,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing split manifest: {manifest_path}"
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


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


def _fusion_candidates(
    classical: np.ndarray,
    dl: np.ndarray,
    low_light: np.ndarray,
) -> dict[str, tuple[np.ndarray, float]]:
    """Return all fixed candidate images with their fusion-only time in ms."""
    candidates: dict[str, tuple[np.ndarray, float]] = {}
    for label, weight in (
        ("Weighted 50:50", 0.5),
        ("DL-dominant 90:10", 0.1),
        ("DL-dominant 95:5", 0.05),
        ("DL-dominant 80:20", 0.2),
    ):
        started = time.perf_counter()
        output = weighted_fusion(classical, dl, classical_weight=weight)
        candidates[label] = (output, (time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    adaptive = adaptive_fusion(classical, dl, low_light)
    candidates["Adaptive fusion"] = (
        adaptive,
        (time.perf_counter() - started) * 1000,
    )
    started = time.perf_counter()
    spatial = spatial_adaptive_fusion(classical, dl, low_light)
    candidates["Spatial-adaptive fusion"] = (
        spatial,
        (time.perf_counter() - started) * 1000,
    )
    return candidates


def _record_metrics(
    row: dict[str, str | int | float],
    enhanced: np.ndarray,
    reference: np.ndarray,
    failures: list[str],
    color_rows: list[dict[str, Any]],
    image_name: str,
    method: str,
) -> None:
    height, width = enhanced.shape[:2]
    row["output_height"] = height
    row["output_width"] = width
    try:
        aligned_reference = _resize_reference(reference, (width, height))
    except ValueError as error:
        failures.append(f"{image_name} [{method}] reference alignment failed: {error}")
        return

    try:
        reference_lab = rgb2lab(aligned_reference.astype(np.float32) / 255.0)
        enhanced_lab = rgb2lab(enhanced.astype(np.float32) / 255.0)
        color_rows.append(
            {
                "image_name": image_name,
                "method": method,
                "mean_delta_e2000": float(
                    np.mean(deltaE_ciede2000(reference_lab, enhanced_lab))
                ),
            }
        )
    except Exception as error:
        color_rows.append(
            {"image_name": image_name, "method": method, "mean_delta_e2000": ""}
        )
        failures.append(
            f"{image_name} [{method}] CIEDE2000 color-fidelity metric failed: "
            f"{type(error).__name__}: {error}"
        )

    paired_metrics = {
        "psnr": calculate_psnr,
        "ssim": calculate_ssim,
        "lpips": calculate_lpips,
    }
    for metric_name, metric in paired_metrics.items():
        try:
            value = float(metric(aligned_reference, enhanced))
            if math.isnan(value):
                raise ValueError("metric returned NaN")
            row[metric_name] = value
        except Exception as error:
            failures.append(
                f"{image_name} [{method}] {metric_name.upper()} failed: "
                f"{type(error).__name__}: {error}"
            )
    try:
        value = float(calculate_niqe(enhanced))
        if math.isnan(value):
            raise ValueError("metric returned NaN")
        row["niqe"] = value
    except Exception as error:
        failures.append(
            f"{image_name} [{method}] NIQE failed: {type(error).__name__}: {error}"
        )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_validation_experiment(
    input_dir: Path,
    reference_dir: Path,
    manifest_path: Path,
    output_csv: Path,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Evaluate every fusion candidate on the fixed validation split.

    Enhancement clients run once per image; all fusion candidates reuse those
    paired outputs. No eval15 image is read by this function.
    """
    input_dir = Path(input_dir)
    reference_dir = Path(reference_dir)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("dataset") != "LOLv2 our485":
        raise ValueError("The split manifest must identify LOLv2 our485.")
    validation_names = manifest.get("validation_images")
    if not isinstance(validation_names, list) or not validation_names:
        raise ValueError("The split manifest has no validation images.")

    output_csv = Path(output_csv)
    errors_path = output_csv.with_suffix(output_csv.suffix + ".errors.txt")
    summary_path = output_csv.with_name(output_csv.stem + "_summary.csv")
    decision_path = output_csv.with_name(output_csv.stem + "_decision.json")
    color_path = output_csv.with_name(output_csv.stem + "_color_fidelity.csv")
    for path in (output_csv, errors_path, summary_path, decision_path, color_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    rows: list[dict[str, Any]] = []
    color_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for image_name in validation_names:
        input_path = input_dir / Path(image_name)
        reference_path = reference_dir / Path(image_name)
        try:
            image, low_light = _read_rgb(input_path)
            _, reference = _read_rgb(reference_path)
        except (OSError, ValueError, UnidentifiedImageError) as error:
            for method in METHODS:
                rows.append(_new_row(image_name, method))
                failures.append(
                    f"{image_name} [{method}] paired image read failed: {error}"
                )
            continue

        outputs: dict[str, tuple[np.ndarray, float, float | None]] = {}
        branch_errors: dict[str, str] = {}
        for method, client in (
            ("Classical CV", enhance_with_classical),
            ("Pretrained Retinexformer", enhance_with_api),
        ):
            try:
                output, seconds = client(image)
                outputs[method] = (
                    _validate_enhanced_image(output, method),
                    float(seconds),
                    None,
                )
            except Exception as error:
                branch_errors[method] = f"{type(error).__name__}: {error}"

        if not branch_errors:
            classical, classical_seconds, _ = outputs["Classical CV"]
            dl, dl_seconds, _ = outputs["Pretrained Retinexformer"]
            if classical.shape != low_light.shape or dl.shape != low_light.shape:
                branch_errors["fusion"] = "Enhancement output dimensions differ from input."
            else:
                try:
                    for method, (enhanced, fusion_ms) in _fusion_candidates(
                        classical, dl, low_light
                    ).items():
                        outputs[method] = (
                            _validate_enhanced_image(enhanced, method),
                            classical_seconds + dl_seconds,
                            fusion_ms,
                        )
                except Exception as error:
                    branch_errors["fusion"] = f"{type(error).__name__}: {error}"

        for method in METHODS:
            row = _new_row(image_name, method)
            if method in branch_errors:
                failures.append(
                    f"{image_name} [{method}] inference failed: {branch_errors[method]}"
                )
            elif "fusion" in branch_errors and method not in outputs:
                failures.append(
                    f"{image_name} [{method}] inference failed: {branch_errors['fusion']}"
                )
            elif method not in outputs:
                failures.append(f"{image_name} [{method}] output is unavailable.")
            else:
                enhanced, inference_seconds, fusion_ms = outputs[method]
                row["latency_ms"] = inference_seconds * 1000
                if fusion_ms is not None:
                    row["fusion_latency_ms"] = fusion_ms
                _record_metrics(
                    row,
                    enhanced,
                    reference,
                    failures,
                    color_rows,
                    image_name,
                    method,
                )
            rows.append(row)

    _write_rows(output_csv, rows)
    errors_path.write_text(
        "\n".join(failures) + "\n" if failures else "No validation failures.\n",
        encoding="utf-8",
    )
    _write_color_fidelity(color_path, color_rows)
    summary = summarize_candidates(rows, color_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        columns = list(summary[0]) if summary else ["method"]
        writer = csv.DictWriter(summary_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)
    decision = select_validation_strategy(
        rows,
        color_rows,
        seed=int(manifest.get("seed", SEED)),
    )
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    return rows, failures, decision


def _write_color_fidelity(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as color_file:
        writer = csv.DictWriter(
            color_file,
            fieldnames=("image_name", "method", "mean_delta_e2000"),
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_candidates(
    rows: list[dict[str, Any]],
    color_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Summarize valid measurements while preserving missing values."""
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        result: dict[str, Any] = {"method": method, "samples": len(selected)}
        for metric in ("psnr", "ssim", "lpips", "niqe", "latency_ms", "fusion_latency_ms"):
            values = [
                float(row[metric])
                for row in selected
                if row.get(metric) not in ("", None)
            ]
            result[f"{metric}_mean"] = (
                float(np.mean(values)) if values else ""
            )
            result[f"{metric}_median"] = (
                float(np.median(values)) if values else ""
            )
            result[f"{metric}_valid_samples"] = len(values)
        color_values = [
            float(row["mean_delta_e2000"])
            for row in (color_rows or [])
            if row["method"] == method
            and row.get("mean_delta_e2000") not in ("", None)
        ]
        result["mean_delta_e2000"] = (
            float(np.mean(color_values)) if color_values else ""
        )
        result["color_fidelity_valid_samples"] = len(color_values)
        summary.append(result)
    return summary


def _paired_deltas(
    rows: list[dict[str, Any]],
    method: str,
    metric: str,
    *,
    lower_is_better: bool = False,
) -> list[float]:
    candidates: dict[str, dict[str, Any]] = {}
    baselines: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["method"] == method:
            candidates[row["image_name"]] = row
        elif row["method"] == "Pretrained Retinexformer":
            baselines[row["image_name"]] = row
    deltas: list[float] = []
    for image_name in sorted(candidates.keys() & baselines.keys()):
        candidate = candidates[image_name]
        baseline = baselines[image_name]
        candidate_value = candidate.get(metric)
        baseline_value = baseline.get(metric)
        if candidate_value in ("", None) or baseline_value in ("", None):
            continue
        candidate_number = float(candidate_value)
        baseline_number = float(baseline_value)
        if math.isinf(candidate_number) or math.isinf(baseline_number):
            delta = (
                0.0
                if candidate_number == baseline_number
                else math.inf if candidate_number > baseline_number else -math.inf
            )
        else:
            delta = candidate_number - baseline_number
        deltas.append(-delta if lower_is_better else delta)
    return deltas


def _bootstrap_lower_bound(
    values: list[float],
    *,
    seed: int = SEED,
    iterations: int = 10_000,
) -> float | None:
    """Return the lower endpoint of a reproducible percentile 95% CI."""
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        return None
    rng = np.random.default_rng(seed)
    observations = np.asarray(values, dtype=np.float64)
    samples = rng.choice(
        observations,
        size=(iterations, observations.size),
        replace=True,
    ).mean(axis=1)
    return float(np.quantile(samples, 0.025))


def select_validation_strategy(
    rows: list[dict[str, Any]],
    color_rows: list[dict[str, Any]] | None = None,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    """Select a fusion only when paired metrics show consistent improvement.

    A candidate must have positive paired bootstrap lower bounds for PSNR,
    SSIM, and the negative LPIPS difference, and must not worsen mean NIQE.
    NIQE is a secondary naturalness metric, not a reference-fidelity score.
    """
    candidates = ("Weighted 50:50", "Adaptive fusion", "Spatial-adaptive fusion",
                  "DL-dominant 90:10", "DL-dominant 95:5", "DL-dominant 80:20")
    evaluated: list[dict[str, Any]] = []
    for index, method in enumerate(candidates):
        intervals = {
            metric: _bootstrap_lower_bound(
                _paired_deltas(
                    rows,
                    method,
                    metric,
                    lower_is_better=metric == "lpips",
                ),
                seed=seed + index * 3 + offset,
            )
            for offset, metric in enumerate(_PAIRED_QUALITY_METRICS)
        }
        niqe_deltas = _paired_deltas(
            rows,
            method,
            "niqe",
            lower_is_better=False,
        )
        niqe_mean_delta = (
            float(np.mean(niqe_deltas)) if niqe_deltas else None
        )
        color_deltas = _paired_deltas(
            [
                {
                    "image_name": row["image_name"],
                    "method": row["method"],
                    "mean_delta_e2000": row.get("mean_delta_e2000"),
                }
                for row in (color_rows or [])
            ],
            method,
            "mean_delta_e2000",
        )
        mean_color_delta = float(np.mean(color_deltas)) if color_deltas else None
        reliable = (
            all(value is not None and value > 0 for value in intervals.values())
            and niqe_mean_delta is not None
            and niqe_mean_delta <= 0
            and mean_color_delta is not None
            and mean_color_delta <= 0
        )
        summaries = {
            row["method"]: row for row in summarize_candidates(rows)
        }
        evaluated.append(
            {
                "method": method,
                "paired_bootstrap_lower_95_ci": intervals,
                "mean_niqe_delta_vs_retinexformer": niqe_mean_delta,
                "mean_delta_e2000_delta_vs_retinexformer": mean_color_delta,
                "reliably_improves": reliable,
                "mean_fusion_latency_ms": (
                    float(summaries[method]["fusion_latency_ms_mean"])
                    if summaries[method]["fusion_latency_ms_mean"] != ""
                    else None
                ),
            }
        )

    qualifying = [row for row in evaluated if row["reliably_improves"]]
    selected = (
        min(
            qualifying,
            key=lambda row: row["mean_fusion_latency_ms"]
            if row["mean_fusion_latency_ms"] is not None
            else math.inf,
        )["method"]
        if qualifying
        else "Pretrained Retinexformer"
    )
    return {
        "validation_seed": SEED,
        "validation_images": len(
            {row["image_name"] for row in rows}
        ),
        "selection_rule": (
            "Require positive paired-bootstrap 95% CI lower bounds for PSNR, "
            "SSIM, and -LPIPS, plus mean NIQE and mean CIEDE2000 no worse than "
            "RetinexFormer. NIQE is descriptive, not a reference-fidelity measure."
        ),
        "selected_strategy": selected,
        "recommended_production_method": (
            "RetinexFormer" if selected == "Pretrained Retinexformer" else selected
        ),
        "fusion_candidates": evaluated,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a reproducible our485 split and compare fusion candidates."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--create-split-only",
        action="store_true",
        help="Create the manifest without running model inference.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.create_split_only:
            manifest = create_split_manifest(
                args.input_dir,
                args.reference_dir,
                args.manifest,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
            )
            print(
                f"Created paired split with {len(manifest['training_images'])} train "
                f"and {len(manifest['validation_images'])} validation images: "
                f"{args.manifest}"
            )
            return 0

        _, failures, decision = run_validation_experiment(
            args.input_dir,
            args.reference_dir,
            args.manifest,
            args.output,
        )
    except (OSError, ValueError, TypeError) as error:
        print(f"Validation experiment could not start: {error}", file=sys.stderr)
        return 2
    print(f"Validation results written to {args.output}")
    print(f"Selected strategy: {decision['selected_strategy']}")
    if failures:
        print(
            f"Completed with {len(failures)} issue(s); see "
            f"{args.output.with_suffix(args.output.suffix + '.errors.txt')}.",
            file=sys.stderr,
        )
        return 1
    print("All validation images and metrics completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
