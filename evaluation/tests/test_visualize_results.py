"""Tests for evaluation CSV parsing, aggregation, and missing-value handling."""

import csv
import math
from pathlib import Path

import pytest

from evaluation import visualize_results as visualization

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


def _write_results(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    method: str,
    *,
    psnr: str = "",
    ssim: str = "",
    lpips: str = "",
    niqe: str = "",
    latency_ms: str = "",
    fusion_latency_ms: str = "",
) -> dict[str, str]:
    return {
        "image_name": "sample.png",
        "method": method,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "niqe": niqe,
        "latency_ms": latency_ms,
        "fusion_latency_ms": fusion_latency_ms,
        "output_height": "96",
        "output_width": "96",
    }


def test_aggregates_mean_median_population_std_and_valid_count() -> None:
    rows = [
        _row(
            "Classical CV",
            psnr="20",
            ssim="0.7",
            lpips="0.3",
            niqe="5",
            latency_ms="100",
        ),
        _row(
            "Classical CV",
            psnr="40",
            ssim="0.9",
            lpips="0.1",
            niqe="3",
            latency_ms="200",
        ),
        _row("Pretrained Retinexformer", psnr="30", latency_ms="150"),
    ]

    summary, groups = visualization.aggregate_results(rows)
    psnr = next(
        row
        for row in summary
        if row["metric"] == "psnr" and row["method"] == "Classical CV"
    )
    assert psnr == {
        "metric": "psnr",
        "method": "Classical CV",
        "mean": 30.0,
        "median": 30.0,
        "std": 10.0,
        "valid_samples": 2,
    }
    assert groups[("fusion_latency_ms", "Classical CV")] == []


def test_missing_values_are_excluded_not_replaced_with_zero() -> None:
    rows = [
        _row("Classical CV", psnr="", ssim="NaN", lpips="N/A", niqe="null"),
        _row("Classical CV", psnr="25", latency_ms=""),
        _row("Hybrid Intelligence", psnr="inf", fusion_latency_ms="12"),
    ]

    summary, _ = visualization.aggregate_results(rows)
    classical_psnr = next(
        row
        for row in summary
        if row["metric"] == "psnr" and row["method"] == "Classical CV"
    )
    missing_niqe = next(
        row
        for row in summary
        if row["metric"] == "niqe" and row["method"] == "Classical CV"
    )
    hybrid_psnr = next(
        row
        for row in summary
        if row["metric"] == "psnr" and row["method"] == "Hybrid Intelligence"
    )
    assert classical_psnr["mean"] == 25.0
    assert classical_psnr["valid_samples"] == 1
    assert missing_niqe["mean"] == ""
    assert missing_niqe["median"] == ""
    assert missing_niqe["std"] == ""
    assert missing_niqe["valid_samples"] == 0
    assert math.isinf(hybrid_psnr["mean"])
    assert math.isinf(hybrid_psnr["median"])
    assert hybrid_psnr["std"] == 0.0


def test_visualizer_writes_only_charts_with_valid_samples(tmp_path) -> None:
    results_csv = tmp_path / "results.csv"
    output_dir = tmp_path / "visuals"
    _write_results(
        results_csv,
        [
            _row("Classical CV", psnr="20", latency_ms="100"),
            _row("Pretrained Retinexformer", psnr="25", latency_ms="NaN"),
            _row("Hybrid Intelligence", psnr="inf", fusion_latency_ms="5"),
        ],
    )

    summary = visualization.visualize_results(results_csv, output_dir)

    assert len(summary) == len(visualization.METRICS) * len(visualization.METHODS)
    assert (output_dir / "psnr.png").is_file()
    assert (output_dir / "latency_ms.png").is_file()
    assert (output_dir / "fusion_latency_ms.png").is_file()
    assert not (output_dir / "ssim.png").exists()
    assert not (output_dir / "lpips.png").exists()
    assert not (output_dir / "niqe.png").exists()
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "summary.txt").is_file()
    assert "n=0" in (output_dir / "summary.txt").read_text(encoding="utf-8")


def test_missing_required_csv_columns_are_reported(tmp_path) -> None:
    results_csv = tmp_path / "incomplete.csv"
    results_csv.write_text("method,psnr\nClassical CV,20\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        visualization.visualize_results(results_csv, tmp_path / "out")


def test_removes_stale_chart_when_metric_has_no_valid_samples(tmp_path) -> None:
    results_csv = tmp_path / "results.csv"
    output_dir = tmp_path / "visuals"
    output_dir.mkdir()
    stale_chart = output_dir / "ssim.png"
    stale_chart.write_bytes(b"old chart")
    _write_results(results_csv, [_row("Classical CV", ssim="")])

    visualization.visualize_results(results_csv, output_dir)

    assert not stale_chart.exists()


def test_nonnumeric_metric_is_not_silently_discarded(tmp_path) -> None:
    results_csv = tmp_path / "invalid.csv"
    _write_results(results_csv, [_row("Classical CV", ssim="unknown")])

    with pytest.raises(ValueError, match="Invalid ssim value"):
        visualization.visualize_results(results_csv, tmp_path / "out")
