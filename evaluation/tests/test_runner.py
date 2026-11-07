"""Tests for the paired-image evaluation runner."""

import csv
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from evaluation import runner


def _write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _mock_metrics(monkeypatch) -> None:
    monkeypatch.setattr(runner, "calculate_psnr", Mock(return_value=30.0))
    monkeypatch.setattr(runner, "calculate_ssim", Mock(return_value=0.9))
    monkeypatch.setattr(runner, "calculate_lpips", Mock(return_value=0.1))
    monkeypatch.setattr(runner, "calculate_niqe", Mock(return_value=4.0))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_writes_one_row_per_method_and_separates_hybrid_fusion_latency(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    reference_dir = tmp_path / "reference"
    output = tmp_path / "results.csv"
    _write_image(input_dir / "scene.png", (96, 96), (10, 20, 30))
    _write_image(reference_dir / "scene.png", (96, 96), (100, 110, 120))
    classical_output = np.full((96, 96, 3), 40, dtype=np.uint8)
    dl_output = np.full((96, 96, 3), 60, dtype=np.uint8)
    fused_output = np.full((96, 96, 3), 50, dtype=np.uint8)
    classical = Mock(return_value=(classical_output, 0.01))
    dl = Mock(return_value=(dl_output, 0.02))
    fusion = Mock(return_value=fused_output)
    monkeypatch.setattr(runner, "enhance_with_classical", classical)
    monkeypatch.setattr(runner, "enhance_with_api", dl)
    monkeypatch.setattr(runner, "spatial_adaptive_fusion", fusion)
    _mock_metrics(monkeypatch)

    saved_images = tmp_path / "enhanced"
    failures = runner.run_evaluation(
        input_dir,
        reference_dir,
        output,
        output_images_dir=saved_images,
    )

    assert failures == []
    rows = _rows(output)
    assert [row["method"] for row in rows] == list(runner.METHODS)
    assert len(rows) == 3
    assert rows[0]["latency_ms"] == "10.0"
    assert rows[1]["latency_ms"] == "20.0"
    hybrid_row = rows[2]
    assert hybrid_row["latency_ms"] == "30.0"
    assert float(hybrid_row["fusion_latency_ms"]) >= 0
    assert hybrid_row["output_height"] == "96"
    assert hybrid_row["output_width"] == "96"
    assert fusion.call_count == 1
    assert (tmp_path / "results.csv.errors.txt").read_text(
        encoding="utf-8"
    ) == "No evaluation failures.\n"
    assert (
        saved_images
        / "hybrid_intelligence"
        / "scene.png.png"
    ).is_file()


def test_resizes_reference_without_changing_aspect_ratio(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    reference_dir = tmp_path / "reference"
    output = tmp_path / "results.csv"
    _write_image(input_dir / "nested" / "scene.png", (96, 48), (10, 20, 30))
    _write_image(reference_dir / "nested" / "scene.png", (192, 96), (100, 110, 120))
    enhanced = np.full((48, 96, 3), 60, dtype=np.uint8)
    monkeypatch.setattr(
        runner,
        "enhance_with_classical",
        Mock(return_value=(enhanced, 0.01)),
    )
    dl = Mock(return_value=(enhanced, 0.02))
    monkeypatch.setattr(
        runner,
        "enhance_with_api",
        dl,
    )
    monkeypatch.setattr(runner, "spatial_adaptive_fusion", Mock(return_value=enhanced))
    _mock_metrics(monkeypatch)

    failures = runner.run_evaluation(input_dir, reference_dir, output)

    assert failures == []
    assert len(_rows(output)) == 3
    assert runner.calculate_psnr.call_args.args[0].shape == (48, 96, 3)
    assert runner.calculate_ssim.call_args.args[0].shape == (48, 96, 3)
    assert runner.calculate_lpips.call_args.args[0].shape == (48, 96, 3)


def test_missing_reference_keeps_records_and_blanks_paired_metrics(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    reference_dir = tmp_path / "reference"
    output = tmp_path / "results.csv"
    _write_image(input_dir / "scene.png", (96, 96), (10, 20, 30))
    enhanced = np.full((96, 96, 3), 60, dtype=np.uint8)
    monkeypatch.setattr(
        runner,
        "enhance_with_classical",
        Mock(return_value=(enhanced, 0.01)),
    )
    monkeypatch.setattr(
        runner,
        "enhance_with_api",
        Mock(return_value=(enhanced, 0.02)),
    )
    monkeypatch.setattr(runner, "spatial_adaptive_fusion", Mock(return_value=enhanced))
    _mock_metrics(monkeypatch)

    failures = runner.run_evaluation(input_dir, reference_dir, output)

    rows = _rows(output)
    assert len(rows) == 3
    assert all(row["psnr"] == "" for row in rows)
    assert all(row["ssim"] == "" for row in rows)
    assert all(row["lpips"] == "" for row in rows)
    assert all(row["niqe"] == "4.0" for row in rows)
    assert len(failures) == 3
    assert all("reference directory does not exist" in failure for failure in failures)


def test_different_aspect_ratio_reports_failure_without_distortion(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    reference_dir = tmp_path / "reference"
    output = tmp_path / "results.csv"
    _write_image(input_dir / "scene.png", (96, 48), (10, 20, 30))
    _write_image(reference_dir / "scene.png", (96, 64), (100, 110, 120))
    enhanced = np.full((48, 96, 3), 60, dtype=np.uint8)
    monkeypatch.setattr(
        runner,
        "enhance_with_classical",
        Mock(return_value=(enhanced, 0.01)),
    )
    monkeypatch.setattr(
        runner,
        "enhance_with_api",
        Mock(return_value=(enhanced, 0.02)),
    )
    monkeypatch.setattr(runner, "spatial_adaptive_fusion", Mock(return_value=enhanced))
    _mock_metrics(monkeypatch)

    failures = runner.run_evaluation(input_dir, reference_dir, output)

    assert len(_rows(output)) == 3
    assert len(failures) == 3
    assert all("aspect ratios differ" in failure for failure in failures)
    assert all(row["psnr"] == "" for row in _rows(output))


def test_failed_inference_keeps_all_method_rows_and_reports_failure(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    reference_dir = tmp_path / "reference"
    output = tmp_path / "results.csv"
    _write_image(input_dir / "scene.png", (96, 96), (10, 20, 30))
    _write_image(reference_dir / "scene.png", (96, 96), (100, 110, 120))
    enhanced = np.full((96, 96, 3), 60, dtype=np.uint8)
    monkeypatch.setattr(
        runner,
        "enhance_with_classical",
        Mock(side_effect=RuntimeError("M1 unavailable")),
    )
    dl = Mock(return_value=(enhanced, 0.02))
    monkeypatch.setattr(
        runner,
        "enhance_with_api",
        dl,
    )
    fusion = Mock(return_value=enhanced)
    monkeypatch.setattr(runner, "spatial_adaptive_fusion", fusion)
    _mock_metrics(monkeypatch)

    failures = runner.run_evaluation(input_dir, reference_dir, output)

    rows = _rows(output)
    assert len(rows) == 3
    assert rows[0]["psnr"] == "" and rows[0]["latency_ms"] == ""
    assert rows[1]["psnr"] == "30.0"
    assert rows[2]["psnr"] == "" and rows[2]["latency_ms"] == ""
    assert fusion.call_count == 0
    assert dl.call_count == 2
    assert any("M1 unavailable" in failure for failure in failures)
