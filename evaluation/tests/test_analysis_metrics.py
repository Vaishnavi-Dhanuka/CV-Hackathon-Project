"""Tests for noise and edge-preservation analysis metrics."""

import unittest

import numpy as np

from evaluation.analysis_metrics import (
    calculate_edge_preservation,
    calculate_noise_reduction,
    estimate_noise_level,
)


class AnalysisMetricTests(unittest.TestCase):
    def test_noise_estimation_accepts_rgb_and_grayscale(self) -> None:
        gray = np.random.default_rng(30).integers(
            0, 256, size=(128, 128), dtype=np.uint8
        )
        rgb = np.repeat(gray[:, :, None], 3, axis=2)

        self.assertAlmostEqual(estimate_noise_level(gray), estimate_noise_level(rgb))

    def test_identical_images_have_zero_noise_reduction_and_full_edge_score(self) -> None:
        image = np.zeros((128, 128), dtype=np.uint8)
        image[:, 64:] = 220

        self.assertEqual(calculate_noise_reduction(image, image.copy()), 0.0)
        self.assertEqual(calculate_edge_preservation(image, image.copy()), 1.0)

    def test_noise_reduction_is_positive_for_reduced_noise(self) -> None:
        rng = np.random.default_rng(31)
        base = np.full((128, 128), 120, dtype=np.float32)
        noisy = np.clip(base + rng.normal(0, 18, base.shape), 0, 255).astype(np.uint8)
        smoothed = np.full_like(noisy, 120)

        self.assertGreater(estimate_noise_level(noisy), estimate_noise_level(smoothed))
        self.assertGreater(calculate_noise_reduction(noisy, smoothed), 0.0)

    def test_removed_edges_reduce_edge_preservation(self) -> None:
        reference = np.zeros((128, 128), dtype=np.uint8)
        reference[:, 64:] = 255
        enhanced = np.zeros_like(reference)

        self.assertEqual(calculate_edge_preservation(reference, enhanced), 0.0)

    def test_rgb_edge_preservation_matches_grayscale(self) -> None:
        reference = np.zeros((128, 128), dtype=np.uint8)
        reference[:, 64:] = 255
        enhanced = reference.copy()
        enhanced[:, 64:] = 150
        reference_rgb = np.repeat(reference[:, :, None], 3, axis=2)
        enhanced_rgb = np.repeat(enhanced[:, :, None], 3, axis=2)

        self.assertEqual(
            calculate_edge_preservation(reference, enhanced),
            calculate_edge_preservation(reference_rgb, enhanced_rgb),
        )

    def test_constant_images_have_zero_noise_and_no_edge_score_one(self) -> None:
        constant = np.full((64, 64), 100, dtype=np.uint8)
        other_constant = np.full((64, 64), 200, dtype=np.uint8)
        edge_image = np.zeros((64, 64), dtype=np.uint8)
        edge_image[:, 32:] = 255

        self.assertEqual(estimate_noise_level(constant), 0.0)
        self.assertEqual(calculate_edge_preservation(constant, other_constant), 1.0)
        self.assertEqual(calculate_edge_preservation(constant, edge_image), 0.0)

    def test_invalid_images_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "NumPy array"):
            estimate_noise_level([[0]])
        with self.assertRaisesRegex(ValueError, "three channels"):
            estimate_noise_level(np.zeros((32, 32, 4), dtype=np.uint8))
        with self.assertRaisesRegex(TypeError, "uint8 or float"):
            estimate_noise_level(np.zeros((32, 32), dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "finite"):
            estimate_noise_level(np.full((32, 32), np.nan, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "normalized to \\[0, 1\\]"):
            estimate_noise_level(np.full((32, 32), 2.0, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "at least 3"):
            estimate_noise_level(np.zeros((2, 8), dtype=np.uint8))

    def test_pair_metrics_reject_different_dimensions(self) -> None:
        reference = np.zeros((64, 64), dtype=np.uint8)
        enhanced = np.zeros((63, 64), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "identical image dimensions"):
            calculate_noise_reduction(reference, enhanced)
        with self.assertRaisesRegex(ValueError, "identical image dimensions"):
            calculate_edge_preservation(reference, enhanced)


if __name__ == "__main__":
    unittest.main()
