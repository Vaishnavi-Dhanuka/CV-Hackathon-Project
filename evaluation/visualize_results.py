"""Aggregate paired-image evaluation results and render comparison charts."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = (
    "Classical CV",
    "Pretrained Retinexformer",
    "Hybrid Intelligence",
)

METRICS: dict[str, tuple[str, str, bool]] = {
    "psnr": ("PSNR", "PSNR (dB)", True),
    "ssim": ("SSIM", "SSIM (unitless)", True),
    "lpips": ("LPIPS", "LPIPS distance", False),
    "niqe": ("NIQE", "NIQE score", False),
    "latency_ms": ("Inference latency", "Inference latency (ms)", False),
    "fusion_latency_ms": (
        "Hybrid fusion latency",
        "Fusion latency (ms)",
        False,
    ),
}
SUMMARY_COLUMNS = ("metric", "method", "mean", "median", "std", "valid_samples")
_MISSING_VALUES = {"", "na", "n/a", "nan", "null", "none"}


def _read_results(csv_path: Path) -> list[dict[str, str]]:
    """Read and validate the columns and method names in a results CSV."""
    required_columns = {"image_name", "method", *METRICS}
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as results_file:
            reader = csv.DictReader(results_file)
            columns = set(reader.fieldnames or ())
            missing = required_columns - columns
            if missing:
                raise ValueError(
                    f"Results CSV is missing required columns: {', '.join(sorted(missing))}."
                )
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"Could not read results CSV {csv_path}: {error}") from error

    for line_number, row in enumerate(rows, start=2):
        method = (row.get("method") or "").strip()
        if method not in METHODS:
            raise ValueError(
                f"Unknown or missing method on CSV line {line_number}: "
                f"{method or '<blank>'}."
            )
    return rows


def _parse_metric_value(
    raw_value: str | None,
    *,
    metric: str,
    line_number: int,
) -> float | None:
    """Parse a CSV value; blank, NaN, and non-finite non-PSNR values are missing."""
    text = (raw_value or "").strip()
    if text.casefold() in _MISSING_VALUES:
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(
            f"Invalid {metric} value on CSV line {line_number}: {text!r}."
        ) from error

    if math.isnan(value):
        return None
    if metric == "psnr" and value == math.inf:
        return value
    if not math.isfinite(value):
        return None
    return value


def _group_measurements(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[float]]:
    groups = {
        (metric, method): []
        for metric in METRICS
        for method in METHODS
    }
    for line_number, row in enumerate(rows, start=2):
        method = row["method"].strip()
        for metric in METRICS:
            value = _parse_metric_value(
                row.get(metric),
                metric=metric,
                line_number=line_number,
            )
            if value is not None:
                groups[(metric, method)].append(value)
    return groups


def _aggregate(values: list[float]) -> tuple[float | None, float | None, float | None]:
    """Return mean, median, and population standard deviation for values."""
    if not values:
        return None, None, None
    if math.inf in values:
        mean = math.inf
        median = float(statistics.median(values))
        std = 0.0 if all(value == math.inf for value in values) else math.inf
        return mean, median, std
    return (
        statistics.fmean(values),
        float(statistics.median(values)),
        statistics.pstdev(values),
    )


def aggregate_results(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    """Aggregate metric values for each method without imputing missing data."""
    groups = _group_measurements(rows)
    summary: list[dict[str, Any]] = []
    for metric in METRICS:
        for method in METHODS:
            values = groups[(metric, method)]
            mean, median, std = _aggregate(values)
            summary.append(
                {
                    "metric": metric,
                    "method": method,
                    "mean": "" if mean is None else mean,
                    "median": "" if median is None else median,
                    "std": "" if std is None else std,
                    "valid_samples": len(values),
                }
            )
    return summary, groups


def _write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary)


def _render_chart(
    metric: str,
    groups: dict[tuple[str, str], list[float]],
    output_path: Path,
) -> bool:
    """Save a mean-by-method chart; return false if there are no observations."""
    values_by_method = {
        method: groups[(metric, method)]
        for method in METHODS
        if groups[(metric, method)]
    }
    if not values_by_method:
        output_path.unlink(missing_ok=True)
        return False

    title, axis_label, higher_is_better = METRICS[metric]
    means = {
        method: _aggregate(values)[0]
        for method, values in values_by_method.items()
    }
    finite_means = [value for value in means.values() if value is not None and math.isfinite(value)]
    axis_max = max(finite_means, default=1.0)
    if axis_max <= 0:
        axis_max = 1.0
    display_top = axis_max * 1.22

    positions = np.arange(len(values_by_method))
    labels = list(values_by_method)
    heights: list[float] = []
    for method in labels:
        mean = means[method]
        if mean is None or not math.isfinite(mean):
            heights.append(display_top * 0.88)
        else:
            heights.append(mean)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = axis.bar(positions, heights, color=["#6477d9", "#38a3a5", "#e08e45"][:len(labels)])
    for bar, method in zip(bars, labels):
        values = values_by_method[method]
        mean, _, std = _aggregate(values)
        sample_count = len(values)
        if mean is not None and math.isinf(mean):
            annotation = f"∞  (n={sample_count})"
        else:
            annotation = f"{mean:.4g}  (n={sample_count})"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            annotation,
            ha="center",
            va="bottom",
            fontsize=9,
        )
        if std is not None and math.isfinite(std) and math.isfinite(bar.get_height()):
            axis.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                yerr=std,
                fmt="none",
                ecolor="#333333",
                capsize=4,
            )

    direction = "higher is better" if higher_is_better else "lower is better"
    axis.set_title(f"{title} by enhancement method ({direction})")
    axis.set_ylabel(axis_label)
    axis.set_xlabel("Enhancement method")
    axis.set_xticks(positions, labels, rotation=10, ha="right")
    axis.set_ylim(0, display_top)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return True


def _format_value(value: Any) -> str:
    if value == "":
        return "N/A"
    if isinstance(value, (int, float)):
        if math.isinf(value):
            return "∞"
        return f"{value:.4g}"
    return str(value)


def _write_text_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "Evaluation results summary",
        "Aggregates are computed from valid CSV values only; missing values are excluded.",
        "Standard deviation is population standard deviation. Latency is not a quality score.",
        "",
    ]
    for metric, (title, _, higher_is_better) in METRICS.items():
        direction = "higher is better" if higher_is_better else "lower is better"
        lines.append(f"{title} ({direction}):")
        for row in summary:
            if row["metric"] == metric:
                lines.append(
                    f"  {row['method']}: mean={_format_value(row['mean'])}, "
                    f"median={_format_value(row['median'])}, "
                    f"std={_format_value(row['std'])}, "
                    f"n={row['valid_samples']}"
                )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def visualize_results(csv_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Aggregate a runner CSV and save available charts, summary CSV, and text.

    Metric blanks, NaNs, and non-finite values other than positive-infinite
    PSNR are excluded from aggregates. Positive-infinite PSNR is retained as a
    valid perfect-match value and marked explicitly in its chart.

    Returns:
        Summary rows for every metric and method, including zero-count groups.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    rows = _read_results(csv_path)
    summary, groups = aggregate_results(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        _render_chart(metric, groups, output_dir / f"{metric}.png")
    _write_summary_csv(output_dir / "summary.csv", summary)
    _write_text_summary(output_dir / "summary.txt", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation CSV results and create comparison charts."
    )
    parser.add_argument("--input", type=Path, required=True, help="Evaluation results CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for charts, summary CSV, and text summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = visualize_results(args.input, args.output_dir)
    except (OSError, ValueError) as error:
        print(f"Could not visualize results: {error}", file=sys.stderr)
        return 2
    chart_count = sum(
        1 for metric in METRICS if (args.output_dir / f"{metric}.png").exists()
    )
    print(f"Saved {chart_count} chart(s), summary.csv, and summary.txt to {args.output_dir}")
    print(f"Aggregated {len(summary)} metric/method combinations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
