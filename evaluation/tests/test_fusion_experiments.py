import csv
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from evaluation import fusion_experiments as experiments


def _write_image(path: Path, size=(96, 96), color=(20, 30, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_split_manifest_is_seeded_and_keeps_pairs_together(tmp_path):
    low = tmp_path / "our485" / "low"
    high = tmp_path / "our485" / "high"
    for image_name in ("1.png", "2.png", "3.png", "4.png", "5.png"):
        _write_image(low / image_name)
        _write_image(high / image_name)

    manifest_path = tmp_path / "split.json"
    first = experiments.create_split_manifest(
        low, high, manifest_path, seed=21, validation_fraction=0.4
    )
    second = experiments.create_split_manifest(
        low, high, tmp_path / "second.json", seed=21, validation_fraction=0.4
    )

    assert first["validation_images"] == second["validation_images"]
    assert len(first["validation_images"]) == 2
    assert len(first["training_images"]) == 3
    assert set(first["training_images"]).isdisjoint(first["validation_images"])
    assert set(first["training_images"]) | set(first["validation_images"]) == {
        f"{number}.png" for number in range(1, 6)
    }
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == first


def test_split_manifest_rejects_eval15_and_missing_pairs(tmp_path):
    low = tmp_path / "eval15" / "low"
    high = tmp_path / "eval15" / "high"
    _write_image(low / "scene.png")
    _write_image(high / "scene.png")
    with pytest.raises(ValueError, match="must not be created from eval15"):
        experiments.create_split_manifest(low, high, tmp_path / "split.json")

    our_low = tmp_path / "our485" / "low"
    our_high = tmp_path / "our485" / "high"
    _write_image(our_low / "scene.png")
    _write_image(our_high / "other.png")
    with pytest.raises(ValueError, match="Input/reference pairs do not match"):
        experiments.create_split_manifest(
            our_low, our_high, tmp_path / "other.json"
        )


def test_validation_reuses_branch_outputs_and_records_all_strategies(
    tmp_path, monkeypatch
):
    low = tmp_path / "our485" / "low"
    high = tmp_path / "our485" / "high"
    _write_image(low / "scene.png")
    _write_image(high / "scene.png", color=(100, 110, 120))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "LOLv2 our485",
                "validation_images": ["scene.png"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "validation.csv"
    classical_output = np.full((96, 96, 3), 50, dtype=np.uint8)
    dl_output = np.full((96, 96, 3), 70, dtype=np.uint8)
    classical = Mock(return_value=(classical_output, 0.02))
    dl = Mock(return_value=(dl_output, 0.5))
    monkeypatch.setattr(experiments, "enhance_with_classical", classical)
    monkeypatch.setattr(experiments, "enhance_with_api", dl)
    monkeypatch.setattr(experiments, "calculate_psnr", Mock(return_value=20.0))
    monkeypatch.setattr(experiments, "calculate_ssim", Mock(return_value=0.8))
    monkeypatch.setattr(experiments, "calculate_lpips", Mock(return_value=0.2))
    niqe = Mock(side_effect=lambda image: 4.0)
    monkeypatch.setattr(experiments, "calculate_niqe", niqe)

    rows, failures, decision = experiments.run_validation_experiment(
        low, high, manifest_path, output
    )

    assert len(rows) == len(experiments.METHODS)
    assert {row["method"] for row in rows} == set(experiments.METHODS)
    assert failures == []
    assert decision["selected_strategy"] == "Pretrained Retinexformer"
    classical.assert_called_once()
    dl.assert_called_once()
    assert niqe.call_count == len(experiments.METHODS)
    assert all(
        len(call.args) == 1
        for call in niqe.call_args_list
    )
    assert (tmp_path / "validation_summary.csv").is_file()
    assert (tmp_path / "validation_color_fidelity.csv").is_file()
    assert (tmp_path / "validation_decision.json").is_file()
    assert (tmp_path / "validation.csv.errors.txt").read_text(
        encoding="utf-8"
    ) == "No validation failures.\n"
    with output.open(newline="", encoding="utf-8") as results:
        assert len(list(csv.DictReader(results))) == len(experiments.METHODS)


def test_validation_strategy_requires_consistent_paired_improvements():
    rows = []
    color_rows = []
    for image_name in ("one.png", "two.png", "three.png"):
        for method in experiments.METHODS:
            is_selected_candidate = method == "DL-dominant 95:5"
            rows.append(
                {
                    "image_name": image_name,
                    "method": method,
                    "psnr": 20.0 + (1.0 if is_selected_candidate else 0.0),
                    "ssim": 0.8 + (0.02 if is_selected_candidate else 0.0),
                    "lpips": 0.2 - (0.02 if is_selected_candidate else 0.0),
                    "niqe": 4.0,
                    "latency_ms": 1.0,
                    "fusion_latency_ms": 1.0,
                }
            )
            color_rows.append(
                {
                    "image_name": image_name,
                    "method": method,
                    "mean_delta_e2000": (
                        9.0 if is_selected_candidate else 10.0
                    ),
                }
            )

    decision = experiments.select_validation_strategy(rows, color_rows)

    assert decision["selected_strategy"] == "DL-dominant 95:5"
    selected = next(
        row
        for row in decision["fusion_candidates"]
        if row["method"] == "DL-dominant 95:5"
    )
    assert selected["reliably_improves"] is True
    assert all(
        lower_bound > 0
        for lower_bound in selected["paired_bootstrap_lower_95_ci"].values()
    )
    assert selected["mean_delta_e2000_delta_vs_retinexformer"] == -1.0
