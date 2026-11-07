"""Run an eval15 benchmark once and create per-image and visual diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import deltaE_ciede2000, rgb2lab

from evaluation import visualize_results
from evaluation.runner import (
    METHODS,
    _image_files,
    _read_rgb,
    _resize_reference,
    run_evaluation,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as results_file:
        return list(csv.DictReader(results_file))


def _save_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _enhanced_path(output_dir: Path, method: str, image_name: str) -> Path:
    method_directory = method.casefold().replace(" ", "_")
    return output_dir / method_directory / f"{image_name}.png"


def _color_delta_e2000(reference: np.ndarray, enhanced: np.ndarray) -> float:
    """Return mean CIEDE2000 distance after alignment; lower means closer color."""
    aligned_reference = _resize_reference(reference, (enhanced.shape[1], enhanced.shape[0]))
    reference_lab = rgb2lab(aligned_reference.astype(np.float32) / 255.0)
    enhanced_lab = rgb2lab(enhanced.astype(np.float32) / 255.0)
    return float(np.mean(deltaE_ciede2000(reference_lab, enhanced_lab)))


def _comparison_image(
    image_name: str,
    input_image: np.ndarray,
    reference_image: np.ndarray,
    enhanced_by_method: dict[str, np.ndarray],
    destination: Path,
) -> None:
    labels = (
        "Low-light input",
        "Ground truth",
        "Classical CV",
        "Official RetinexFormer",
        "Hybrid Intelligence (spatial, experimental)",
    )
    images = (
        input_image,
        reference_image,
        enhanced_by_method["Classical CV"],
        enhanced_by_method["Pretrained Retinexformer"],
        enhanced_by_method["Hybrid Intelligence"],
    )
    panel_width = 300
    panel_height = 450
    title_height = 48
    canvas = Image.new(
        "RGB",
        (panel_width * len(images), panel_height + title_height),
        (11, 16, 32),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, array) in enumerate(zip(labels, images)):
        x = index * panel_width
        draw.text((x + 8, 12), label, fill=(248, 250, 252), font=font)
        thumbnail = Image.fromarray(array, mode="RGB")
        thumbnail.thumbnail((panel_width - 12, panel_height - 12), Image.Resampling.LANCZOS)
        canvas.paste(thumbnail, (x + (panel_width - thumbnail.width) // 2, title_height))
        draw.rectangle(
            (x, title_height, x + panel_width - 1, title_height + panel_height - 1),
            outline=(38, 51, 77),
            width=1,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")


def finalize_eval15_benchmark(
    input_dir: Path,
    reference_dir: Path,
    output_csv: Path,
    artifacts_dir: Path,
    enhanced_images_dir: Path,
) -> tuple[list[str], int]:
    """Run all three methods and generate metrics, charts, analysis, and examples.

    The output CSV and artifact directories must be new paths. Existing
    official benchmark files are never replaced.
    """
    input_dir = Path(input_dir)
    reference_dir = Path(reference_dir)
    output_csv = Path(output_csv)
    artifacts_dir = Path(artifacts_dir)
    enhanced_images_dir = Path(enhanced_images_dir)
    error_path = output_csv.with_suffix(output_csv.suffix + ".errors.txt")
    for path in (output_csv, error_path, artifacts_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    if enhanced_images_dir.exists() and any(enhanced_images_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite enhanced images in {enhanced_images_dir}"
        )

    failures = run_evaluation(
        input_dir,
        reference_dir,
        output_csv,
        output_images_dir=enhanced_images_dir,
    )
    result_rows = _read_csv(output_csv)
    input_paths = _image_files(input_dir)
    expected_rows = len(input_paths) * len(METHODS)
    diagnostic_failures: list[str] = []
    if len(result_rows) != expected_rows:
        diagnostic_failures.append(
            f"Expected {expected_rows} per-image method rows but found "
            f"{len(result_rows)}."
        )

    rows_by_image: dict[str, dict[str, dict[str, str]]] = {}
    for row in result_rows:
        rows_by_image.setdefault(row["image_name"], {})[row["method"]] = row

    analysis_rows: list[dict[str, Any]] = []
    color_rows: list[dict[str, Any]] = []
    hybrid_deltas: list[tuple[float, str]] = []
    for input_path in input_paths:
        image_name = input_path.relative_to(input_dir).as_posix()
        try:
            _, low_light = _read_rgb(input_path)
            _, reference = _read_rgb(reference_dir / Path(image_name))
        except (OSError, ValueError) as error:
            diagnostic_failures.append(
                f"{image_name} diagnostic image read failed: {error}"
            )
            continue

        per_method: dict[str, dict[str, str]] = rows_by_image.get(image_name, {})
        enhanced_by_method: dict[str, np.ndarray] = {}
        for method in METHODS:
            row = per_method.get(method, {})
            enhanced_path = _enhanced_path(enhanced_images_dir, method, image_name)
            enhanced: np.ndarray | None = None
            if enhanced_path.is_file():
                try:
                    _, enhanced = _read_rgb(enhanced_path)
                    enhanced_by_method[method] = enhanced
                except ValueError as error:
                    diagnostic_failures.append(
                        f"{image_name} [{method}] saved output could not be read: {error}"
                    )

            mean_delta = ""
            if enhanced is not None:
                try:
                    mean_delta = _color_delta_e2000(reference, enhanced)
                except Exception as error:
                    diagnostic_failures.append(
                        f"{image_name} [{method}] CIEDE2000 failed: "
                        f"{type(error).__name__}: {error}"
                    )
            color_rows.append(
                {
                    "image_name": image_name,
                    "method": method,
                    "mean_delta_e2000": mean_delta,
                }
            )
            analysis = {
                "image_name": image_name,
                "method": method,
                "psnr": row.get("psnr", ""),
                "ssim": row.get("ssim", ""),
                "lpips": row.get("lpips", ""),
                "niqe": row.get("niqe", ""),
                "mean_delta_e2000": mean_delta,
                "latency_ms": row.get("latency_ms", ""),
                "fusion_latency_ms": row.get("fusion_latency_ms", ""),
                "psnr_delta_vs_retinexformer": "",
                "ssim_delta_vs_retinexformer": "",
                "lpips_delta_vs_retinexformer": "",
                "niqe_delta_vs_retinexformer": "",
            }
            baseline = per_method.get("Pretrained Retinexformer", {})
            for metric in ("psnr", "ssim", "lpips", "niqe"):
                candidate_value = row.get(metric, "")
                baseline_value = baseline.get(metric, "")
                if candidate_value and baseline_value:
                    try:
                        difference = float(candidate_value) - float(baseline_value)
                        if math.isfinite(difference):
                            analysis[f"{metric}_delta_vs_retinexformer"] = difference
                    except ValueError:
                        diagnostic_failures.append(
                            f"{image_name} [{method}] invalid {metric} in results CSV."
                        )
            analysis_rows.append(analysis)

        hybrid = per_method.get("Hybrid Intelligence", {})
        baseline = per_method.get("Pretrained Retinexformer", {})
        try:
            hybrid_psnr = float(hybrid["psnr"])
            baseline_psnr = float(baseline["psnr"])
            if math.isfinite(hybrid_psnr) and math.isfinite(baseline_psnr):
                hybrid_deltas.append((hybrid_psnr - baseline_psnr, image_name))
        except (KeyError, ValueError):
            continue

        if len(enhanced_by_method) == len(METHODS):
            reference_aligned = _resize_reference(
                reference,
                (low_light.shape[1], low_light.shape[0]),
            )
            _comparison_image(
                image_name,
                low_light,
                reference_aligned,
                enhanced_by_method,
                artifacts_dir
                / "visual_comparisons"
                / f"{Path(image_name).with_suffix('.png').as_posix().replace('/', '__')}",
            )

    analysis_columns = (
        "image_name",
        "method",
        "psnr",
        "ssim",
        "lpips",
        "niqe",
        "mean_delta_e2000",
        "latency_ms",
        "fusion_latency_ms",
        "psnr_delta_vs_retinexformer",
        "ssim_delta_vs_retinexformer",
        "lpips_delta_vs_retinexformer",
        "niqe_delta_vs_retinexformer",
    )
    _save_csv(
        artifacts_dir / "per_image_analysis.csv",
        analysis_rows,
        analysis_columns,
    )
    _save_csv(
        artifacts_dir / "color_fidelity.csv",
        color_rows,
        ("image_name", "method", "mean_delta_e2000"),
    )
    visualize_results.visualize_results(output_csv, artifacts_dir / "charts")

    examples = sorted(hybrid_deltas, key=lambda item: (item[0], item[1]))
    best_example = examples[-1] if examples else None
    worst_example = examples[0] if examples else None
    summary_lines = [
        "eval15 per-image comparison notes",
        "The existing spatial-adaptive fusion is shown as an experimental Hybrid baseline.",
        "PSNR/SSIM/LPIPS compare against paired ground truth; NIQE is no-reference and is not fidelity.",
        "Best/worst examples rank Hybrid against RetinexFormer by per-image PSNR delta only.",
    ]
    for label, item in (("Least degraded by hybrid", best_example), ("Most degraded by hybrid", worst_example)):
        if item is None:
            summary_lines.append(f"{label}: unavailable (no valid paired PSNR comparison).")
            continue
        delta, image_name = item
        summary_lines.append(f"{label}: {image_name} (Hybrid − RetinexFormer PSNR: {delta:+.4f} dB)")
        matching = next(
            (
                path
                for path in input_paths
                if path.relative_to(input_dir).as_posix() == image_name
            ),
            None,
        )
        if matching is not None:
            try:
                _, low_light = _read_rgb(matching)
                _, reference = _read_rgb(reference_dir / Path(image_name))
                outputs = {
                    method: _read_rgb(
                        _enhanced_path(enhanced_images_dir, method, image_name)
                    )[1]
                    for method in METHODS
                }
                _comparison_image(
                    image_name,
                    low_light,
                    _resize_reference(
                        reference,
                        (low_light.shape[1], low_light.shape[0]),
                    ),
                    outputs,
                    artifacts_dir / f"{label.casefold().replace(' ', '_')}.png",
                )
            except (OSError, ValueError) as error:
                diagnostic_failures.append(
                    f"{image_name} {label.casefold()} visual comparison failed: {error}"
                )
    (artifacts_dir / "example_analysis.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    existing_errors = error_path.read_text(encoding="utf-8").strip()
    error_lines = (
        []
        if existing_errors in ("", "No evaluation failures.")
        else existing_errors.splitlines()
    )
    error_lines.extend(diagnostic_failures)
    error_path.write_text(
        "\n".join(error_lines) + "\n" if error_lines else "No evaluation failures.\n",
        encoding="utf-8",
    )
    return failures + diagnostic_failures, expected_rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run eval15 once and create M3 comparison artifacts."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--enhanced-images-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        failures, row_count = finalize_eval15_benchmark(
            args.input_dir,
            args.reference_dir,
            args.output,
            args.artifacts_dir,
            args.enhanced_images_dir,
        )
    except (OSError, ValueError, TypeError) as error:
        print(f"Final benchmark could not start: {error}", file=sys.stderr)
        return 2
    print(f"Per-image results: {args.output} ({row_count} expected rows)")
    print(f"Charts and analysis: {args.artifacts_dir}")
    print(f"Enhanced PNG outputs: {args.enhanced_images_dir}")
    if failures:
        print(
            f"Completed with {len(failures)} issue(s); see "
            f"{args.output.with_suffix(args.output.suffix + '.errors.txt')}.",
            file=sys.stderr,
        )
        return 1
    print("All eval15 images and metrics completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
