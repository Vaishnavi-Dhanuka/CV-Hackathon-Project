import csv
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from evaluation import finalize_benchmark
from evaluation import runner


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 96), color).save(path)


def test_finalizer_creates_results_analysis_charts_and_examples(
    tmp_path, monkeypatch
):
    low = tmp_path / "eval15" / "low"
    high = tmp_path / "eval15" / "high"
    _write_image(low / "scene.png", (20, 30, 40))
    _write_image(high / "scene.png", (100, 110, 120))
    outputs = {
        "classical": np.full((96, 96, 3), 50, dtype=np.uint8),
        "dl": np.full((96, 96, 3), 80, dtype=np.uint8),
    }
    monkeypatch.setattr(
        runner,
        "enhance_with_classical",
        Mock(return_value=(outputs["classical"], 0.01)),
    )
    monkeypatch.setattr(
        runner,
        "enhance_with_api",
        Mock(return_value=(outputs["dl"], 0.02)),
    )
    monkeypatch.setattr(runner, "spatial_adaptive_fusion", Mock(return_value=outputs["dl"]))
    monkeypatch.setattr(runner, "calculate_psnr", Mock(return_value=25.0))
    monkeypatch.setattr(runner, "calculate_ssim", Mock(return_value=0.9))
    monkeypatch.setattr(runner, "calculate_lpips", Mock(return_value=0.1))
    monkeypatch.setattr(runner, "calculate_niqe", Mock(return_value=4.0))

    output = tmp_path / "eval15.csv"
    artifacts = tmp_path / "artifacts"
    enhanced_dir = tmp_path / "enhanced"
    failures, row_count = finalize_benchmark.finalize_eval15_benchmark(
        low,
        high,
        output,
        artifacts,
        enhanced_dir,
    )

    assert failures == []
    assert row_count == 3
    with output.open(newline="", encoding="utf-8-sig") as results:
        assert len(list(csv.DictReader(results))) == row_count
    assert (artifacts / "per_image_analysis.csv").is_file()
    assert (artifacts / "color_fidelity.csv").is_file()
    assert (artifacts / "charts" / "summary.csv").is_file()
    assert (artifacts / "charts" / "psnr.png").is_file()
    assert (artifacts / "visual_comparisons" / "scene.png").is_file()
    assert (artifacts / "least_degraded_by_hybrid.png").is_file()
    assert (artifacts / "most_degraded_by_hybrid.png").is_file()
    assert "No evaluation failures." in (
        output.with_suffix(".csv.errors.txt").read_text(encoding="utf-8")
    )


def test_finalizer_refuses_to_overwrite_existing_outputs(tmp_path):
    low = tmp_path / "eval15" / "low"
    high = tmp_path / "eval15" / "high"
    _write_image(low / "scene.png", (20, 30, 40))
    _write_image(high / "scene.png", (100, 110, 120))
    output = tmp_path / "existing.csv"
    output.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        finalize_benchmark.finalize_eval15_benchmark(
            low,
            high,
            output,
            tmp_path / "artifacts",
            tmp_path / "enhanced",
        )

    assert output.read_text(encoding="utf-8") == "preserve"
