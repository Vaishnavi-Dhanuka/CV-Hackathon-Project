"""Unit tests for reference-based image quality metrics."""

import math
import sys
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.metrics import (
    calculate_lpips,
    calculate_niqe,
    calculate_psnr,
    calculate_ssim,
    evaluate_paired_image,
    _get_lpips_model,
    _get_niqe_metric,
)


class ImageQualityMetricTests(unittest.TestCase):
    def test_identical_uint8_grayscale_has_perfect_scores(self) -> None:
        image = np.full((16, 16), 127, dtype=np.uint8)

        self.assertTrue(math.isinf(calculate_psnr(image, image.copy())))
        self.assertEqual(calculate_ssim(image, image.copy()), 1.0)
        self.assertEqual(
            evaluate_paired_image(image, image.copy()),
            {"psnr": math.inf, "ssim": 1.0},
        )

    def test_slightly_modified_image_has_high_finite_scores(self) -> None:
        reference = np.full((16, 16), 100, dtype=np.uint8)
        enhanced = reference.copy()
        enhanced[8, 8] = 110

        self.assertGreater(calculate_psnr(reference, enhanced), 20.0)
        self.assertTrue(0.0 < calculate_ssim(reference, enhanced) < 1.0)

    def test_completely_different_images_have_low_scores(self) -> None:
        reference = np.zeros((16, 16), dtype=np.uint8)
        enhanced = np.full((16, 16), 255, dtype=np.uint8)

        self.assertEqual(calculate_psnr(reference, enhanced), 0.0)
        self.assertLess(calculate_ssim(reference, enhanced), 0.1)

    def test_grayscale_images_are_supported(self) -> None:
        reference = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
        enhanced = reference.copy()
        enhanced[0, 0] += 1

        self.assertTrue(math.isfinite(calculate_psnr(reference, enhanced)))
        self.assertTrue(math.isfinite(calculate_ssim(reference, enhanced)))

    def test_channels_last_rgb_images_are_supported(self) -> None:
        reference = np.zeros((16, 16, 3), dtype=np.uint8)
        enhanced = reference.copy()
        enhanced[8, 8, 1] = 20

        self.assertTrue(math.isfinite(calculate_psnr(reference, enhanced)))
        self.assertTrue(math.isfinite(calculate_ssim(reference, enhanced)))

    def test_different_dimensions_are_rejected(self) -> None:
        reference = np.zeros((16, 16), dtype=np.uint8)
        enhanced = np.zeros((15, 16), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            calculate_psnr(reference, enhanced)
        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            calculate_ssim(reference, enhanced)

    def test_invalid_inputs_are_rejected(self) -> None:
        image = np.zeros((16, 16), dtype=np.uint8)
        with self.assertRaisesRegex(TypeError, "NumPy arrays"):
            calculate_psnr([[0]], image)
        with self.assertRaisesRegex(ValueError, "exactly three channels"):
            calculate_ssim(
                np.zeros((16, 16, 4), dtype=np.uint8),
                np.zeros((16, 16, 4), dtype=np.uint8),
            )
        with self.assertRaisesRegex(TypeError, "uint8"):
            calculate_psnr(
                np.zeros((16, 16), dtype=np.int16),
                np.zeros((16, 16), dtype=np.int16),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            calculate_ssim(
                np.full((16, 16), np.nan, dtype=np.float32),
                np.zeros((16, 16), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            calculate_psnr(
                np.full((16, 16), 2.0, dtype=np.float32),
                np.zeros((16, 16), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "at least 3 pixels"):
            calculate_ssim(
                np.zeros((2, 16), dtype=np.uint8),
                np.zeros((2, 16), dtype=np.uint8),
            )

    def test_normalized_float_images_use_data_range_one(self) -> None:
        reference = np.full((16, 16), 0.5, dtype=np.float32)
        enhanced = reference.copy()
        enhanced[8, 8] = 0.6

        self.assertAlmostEqual(
            calculate_psnr(reference, enhanced),
            44.082398,
            places=6,
        )
        self.assertTrue(0.0 < calculate_ssim(reference, enhanced) < 1.0)

    def test_mixed_uint8_and_float_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "Both images must use"):
            calculate_psnr(
                np.zeros((16, 16), dtype=np.uint8),
                np.zeros((16, 16), dtype=np.float32),
            )

    def test_lpips_identical_rgb_images_have_near_zero_distance(self) -> None:
        image = np.random.default_rng(10).integers(
            0, 256, size=(64, 64, 3), dtype=np.uint8
        )

        score = calculate_lpips(image, image.copy())

        self.assertAlmostEqual(score, 0.0, places=6)

    def test_lpips_slightly_modified_image_has_positive_distance(self) -> None:
        reference = np.random.default_rng(11).integers(
            0, 256, size=(64, 64, 3), dtype=np.uint8
        )
        enhanced = reference.copy()
        enhanced[24:40, 24:40] = 0

        score = calculate_lpips(reference, enhanced)

        self.assertTrue(math.isfinite(score))
        self.assertGreater(score, 0.0)

    def test_lpips_rejects_different_dimensions(self) -> None:
        reference = np.zeros((64, 64, 3), dtype=np.uint8)
        enhanced = np.zeros((64, 65, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            calculate_lpips(reference, enhanced)

    def test_lpips_rejects_invalid_inputs(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        with self.assertRaisesRegex(TypeError, "NumPy arrays"):
            calculate_lpips(image.tolist(), image)
        with self.assertRaisesRegex(ValueError, "channels-last RGB"):
            calculate_lpips(np.zeros((64, 64), dtype=np.uint8), image)
        with self.assertRaisesRegex(ValueError, "three RGB channels"):
            calculate_lpips(
                np.zeros((64, 64, 4), dtype=np.uint8),
                np.zeros((64, 64, 4), dtype=np.uint8),
            )
        with self.assertRaisesRegex(ValueError, "at least 64"):
            calculate_lpips(
                np.zeros((32, 64, 3), dtype=np.uint8),
                np.zeros((32, 64, 3), dtype=np.uint8),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            calculate_lpips(
                np.full((64, 64, 3), np.nan, dtype=np.float32),
                np.zeros((64, 64, 3), dtype=np.float32),
            )
        with self.assertRaisesRegex(TypeError, "Both images must use"):
            calculate_lpips(
                np.zeros((64, 64, 3), dtype=np.uint8),
                np.zeros((64, 64, 3), dtype=np.float32),
            )

    def test_lpips_accepts_normalized_float_rgb_images(self) -> None:
        reference = np.random.default_rng(12).random((64, 64, 3), dtype=np.float32)
        enhanced = reference.copy()

        self.assertAlmostEqual(calculate_lpips(reference, enhanced), 0.0, places=6)

    def test_lpips_model_is_loaded_only_once(self) -> None:
        self.assertIs(_get_lpips_model(), _get_lpips_model())

    def test_missing_lpips_dependencies_raise_actionable_error(self) -> None:
        with (
            patch("evaluation.metrics._LPIPS_MODEL", None),
            patch.dict(sys.modules, {"lpips": None, "torch": None}),
        ):
            with self.assertRaisesRegex(ImportError, "requires the 'lpips' and 'torch'"):
                _get_lpips_model()

    def test_niqe_accepts_rgb_image_and_returns_finite_score(self) -> None:
        image = np.random.default_rng(20).integers(
            0, 256, size=(128, 128, 3), dtype=np.uint8
        )

        score = calculate_niqe(image)

        self.assertTrue(math.isfinite(score))

    def test_niqe_rejects_invalid_input_and_channels(self) -> None:
        with self.assertRaisesRegex(TypeError, "NumPy array"):
            calculate_niqe([[0]])
        with self.assertRaisesRegex(ValueError, "three RGB channels"):
            calculate_niqe(np.zeros((128, 128, 4), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "finite"):
            calculate_niqe(np.full((128, 128, 3), np.nan, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            calculate_niqe(np.full((128, 128, 3), 2.0, dtype=np.float32))
        with self.assertRaisesRegex(TypeError, "uint8"):
            calculate_niqe(np.zeros((128, 128, 3), dtype=np.int16))

    def test_niqe_rejects_unsupported_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "3D channels-last RGB"):
            calculate_niqe(np.zeros((128, 128), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "at least 96 pixels"):
            calculate_niqe(np.zeros((95, 128, 3), dtype=np.uint8))

    def test_niqe_output_is_consistent_and_metric_is_cached(self) -> None:
        image = np.random.default_rng(21).integers(
            0, 256, size=(128, 128, 3), dtype=np.uint8
        )

        first_score = calculate_niqe(image)
        second_score = calculate_niqe(image)

        self.assertEqual(first_score, second_score)
        self.assertIs(_get_niqe_metric(), _get_niqe_metric())

    def test_missing_niqe_dependency_raises_actionable_error(self) -> None:
        with (
            patch("evaluation.metrics._NIQE_METRIC", None),
            patch.dict(sys.modules, {"pyiqa": None}),
        ):
            with self.assertRaisesRegex(
                ImportError, "requires the 'pyiqa' and 'torch' packages"
            ):
                _get_niqe_metric()


if __name__ == "__main__":
    unittest.main()
