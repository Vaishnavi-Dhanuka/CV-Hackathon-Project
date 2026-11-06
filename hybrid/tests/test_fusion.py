"""Tests for baseline weighted restoration fusion."""

import unittest

import numpy as np

from hybrid.fusion import (
    adaptive_fusion,
    spatial_adaptive_fusion,
    weighted_fusion,
)


class WeightedFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classical = np.full((4, 5, 3), 40, dtype=np.uint8)
        self.dl = np.full((4, 5, 3), 100, dtype=np.uint8)

    def test_equal_weight_fusion(self) -> None:
        fused = weighted_fusion(self.classical, self.dl)

        np.testing.assert_array_equal(fused, np.full((4, 5, 3), 70, dtype=np.uint8))

    def test_classical_only_fusion(self) -> None:
        fused = weighted_fusion(self.classical, self.dl, classical_weight=1.0)

        np.testing.assert_array_equal(fused, self.classical)

    def test_dl_only_fusion(self) -> None:
        fused = weighted_fusion(self.classical, self.dl, classical_weight=0.0)

        np.testing.assert_array_equal(fused, self.dl)

    def test_output_shape_dtype_and_range(self) -> None:
        classical = np.full((4, 5, 3), 255, dtype=np.uint8)
        dl = np.zeros((4, 5, 3), dtype=np.uint8)

        fused = weighted_fusion(classical, dl, classical_weight=0.75)

        self.assertEqual(fused.shape, classical.shape)
        self.assertEqual(fused.dtype, np.uint8)
        self.assertTrue(np.all((fused >= 0) & (fused <= 255)))
        np.testing.assert_array_equal(fused, np.full_like(classical, 191))

    def test_identical_inputs_are_preserved(self) -> None:
        image = np.arange(60, dtype=np.uint8).reshape((4, 5, 3))

        fused = weighted_fusion(image, image.copy(), classical_weight=0.37)

        np.testing.assert_array_equal(fused, image)

    def test_invalid_weights_are_rejected(self) -> None:
        for weight in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaisesRegex(
                ValueError, "finite and between 0 and 1"
            ):
                weighted_fusion(self.classical, self.dl, weight)

        with self.assertRaisesRegex(TypeError, "real numeric value"):
            weighted_fusion(self.classical, self.dl, "0.5")
        with self.assertRaisesRegex(TypeError, "real numeric value"):
            weighted_fusion(self.classical, self.dl, True)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "NumPy array"):
            weighted_fusion(self.classical.tolist(), self.dl)
        with self.assertRaisesRegex(ValueError, "HxWx3 RGB"):
            weighted_fusion(np.zeros((4, 5), dtype=np.uint8), self.dl)
        with self.assertRaisesRegex(ValueError, "HxWx3 RGB"):
            weighted_fusion(np.zeros((4, 5, 4), dtype=np.uint8), self.dl)
        with self.assertRaisesRegex(TypeError, "dtype uint8"):
            weighted_fusion(self.classical.astype(np.float32), self.dl)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            weighted_fusion(
                np.zeros((0, 5, 3), dtype=np.uint8),
                np.zeros((0, 5, 3), dtype=np.uint8),
            )

    def test_different_dimensions_are_rejected(self) -> None:
        different_size = np.zeros((3, 5, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            weighted_fusion(self.classical, different_size)


class AdaptiveFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classical = np.full((8, 8, 3), 20, dtype=np.uint8)
        self.dl = np.full((8, 8, 3), 220, dtype=np.uint8)

    def test_normally_illuminated_input_uses_input_statistics(self) -> None:
        input_image = np.full((8, 8, 3), 200, dtype=np.uint8)

        fused, details = adaptive_fusion(
            self.classical, self.dl, input_image, return_weights=True
        )

        self.assertLess(details["classical_weight"], 0.5)
        self.assertEqual(details["input_statistics"]["mean_luminance"], 200.0)
        self.assertEqual(details["input_statistics"]["underexposed_fraction"], 0.0)
        self.assertEqual(fused.shape, input_image.shape)

    def test_severely_underexposed_input_shifts_weight_toward_classical(self) -> None:
        input_image = np.zeros((8, 8, 3), dtype=np.uint8)

        _, details = adaptive_fusion(
            self.classical, self.dl, input_image, return_weights=True
        )

        self.assertGreater(details["classical_weight"], 0.5)
        self.assertEqual(details["input_statistics"]["mean_luminance"], 0.0)
        self.assertEqual(details["input_statistics"]["underexposed_fraction"], 1.0)

    def test_low_contrast_increases_classical_weight_relative_to_higher_contrast(
        self,
    ) -> None:
        low_contrast = np.full((8, 8, 3), 80, dtype=np.uint8)
        higher_contrast = np.empty_like(low_contrast)
        higher_contrast[::2] = 64
        higher_contrast[1::2] = 96

        _, low_details = adaptive_fusion(
            self.classical, self.dl, low_contrast, return_weights=True
        )
        _, higher_details = adaptive_fusion(
            self.classical, self.dl, higher_contrast, return_weights=True
        )

        low_stats = low_details["input_statistics"]
        higher_stats = higher_details["input_statistics"]
        self.assertEqual(low_stats["mean_luminance"], higher_stats["mean_luminance"])
        self.assertEqual(
            low_stats["underexposed_fraction"],
            higher_stats["underexposed_fraction"],
        )
        self.assertLess(low_stats["luminance_contrast"], higher_stats["luminance_contrast"])
        self.assertGreater(
            low_details["classical_weight"], higher_details["classical_weight"]
        )

    def test_weights_are_bounded_and_sum_to_one(self) -> None:
        samples = (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8, 3), 200, dtype=np.uint8),
            np.random.default_rng(42).integers(
                0, 256, size=(8, 8, 3), dtype=np.uint8
            ),
        )

        for image in samples:
            with self.subTest(mean=int(image.mean())):
                _, details = adaptive_fusion(
                    self.classical, self.dl, image, return_weights=True
                )
                classical_weight = details["classical_weight"]
                dl_weight = details["dl_weight"]
                self.assertGreaterEqual(classical_weight, 0.25)
                self.assertLessEqual(classical_weight, 0.75)
                self.assertGreaterEqual(dl_weight, 0.25)
                self.assertLessEqual(dl_weight, 0.75)
                self.assertAlmostEqual(classical_weight + dl_weight, 1.0)

    def test_invalid_shapes_and_dtypes_are_rejected(self) -> None:
        valid = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "HxWx3 RGB"):
            adaptive_fusion(np.zeros((8, 8), dtype=np.uint8), valid, valid)
        with self.assertRaisesRegex(TypeError, "dtype uint8"):
            adaptive_fusion(valid.astype(np.float32), valid, valid)
        with self.assertRaisesRegex(TypeError, "NumPy array"):
            adaptive_fusion(valid, valid, valid.tolist())
        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            adaptive_fusion(valid, valid, np.zeros((7, 8, 3), dtype=np.uint8))

    def test_output_is_deterministic(self) -> None:
        input_image = np.random.default_rng(43).integers(
            0, 256, size=(8, 8, 3), dtype=np.uint8
        )

        first_image, first_details = adaptive_fusion(
            self.classical, self.dl, input_image, return_weights=True
        )
        second_image, second_details = adaptive_fusion(
            self.classical, self.dl, input_image, return_weights=True
        )

        np.testing.assert_array_equal(first_image, second_image)
        self.assertEqual(first_details, second_details)

    def test_return_weights_controls_return_shape(self) -> None:
        input_image = np.full((8, 8, 3), 100, dtype=np.uint8)

        image_only = adaptive_fusion(self.classical, self.dl, input_image)
        image_and_details = adaptive_fusion(
            self.classical, self.dl, input_image, return_weights=True
        )

        self.assertIsInstance(image_only, np.ndarray)
        self.assertIsInstance(image_and_details, tuple)
        self.assertEqual(set(image_and_details[1]), {
            "classical_weight",
            "dl_weight",
            "input_statistics",
        })

    def test_identical_enhancement_outputs_remain_unchanged(self) -> None:
        same_output = np.arange(8 * 8 * 3, dtype=np.uint8).reshape((8, 8, 3))
        input_image = np.zeros((8, 8, 3), dtype=np.uint8)

        fused, details = adaptive_fusion(
            same_output, same_output.copy(), input_image, return_weights=True
        )

        np.testing.assert_array_equal(fused, same_output)
        self.assertEqual(details["classical_weight"], 0.7)
        self.assertAlmostEqual(details["dl_weight"], 0.3)


class SpatialAdaptiveFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classical = np.full((32, 32, 3), 30, dtype=np.uint8)
        self.dl = np.full((32, 32, 3), 210, dtype=np.uint8)
        self.input_image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.input_image[:, 16:] = 200

    def test_output_shape_and_dtype(self) -> None:
        fused = spatial_adaptive_fusion(
            self.classical, self.dl, self.input_image
        )

        self.assertEqual(fused.shape, self.classical.shape)
        self.assertEqual(fused.dtype, np.uint8)
        self.assertTrue(np.all((fused >= 0) & (fused <= 255)))

    def test_maps_have_correct_dimensions_and_sum_to_one(self) -> None:
        _, details = spatial_adaptive_fusion(
            self.classical, self.dl, self.input_image, return_weights=True
        )
        classical_map = details["classical_weight_map"]
        dl_map = details["dl_weight_map"]

        self.assertEqual(classical_map.shape, self.input_image.shape[:2])
        self.assertEqual(dl_map.shape, self.input_image.shape[:2])
        self.assertTrue(np.all((classical_map >= 0.25) & (classical_map <= 0.75)))
        self.assertTrue(np.all((dl_map >= 0.25) & (dl_map <= 0.75)))
        np.testing.assert_allclose(classical_map + dl_map, 1.0, atol=1e-7)
        self.assertIn("input_statistics", details)
        self.assertIn("global_baseline_weights", details)

    def test_dark_regions_receive_higher_classical_weight(self) -> None:
        _, details = spatial_adaptive_fusion(
            self.classical, self.dl, self.input_image, return_weights=True
        )
        classical_map = details["classical_weight_map"]

        self.assertGreater(
            float(np.mean(classical_map[:, :8])),
            float(np.mean(classical_map[:, 24:])),
        )

    def test_weight_map_is_smooth_across_illumination_boundary(self) -> None:
        _, details = spatial_adaptive_fusion(
            self.classical, self.dl, self.input_image, return_weights=True
        )
        classical_map = details["classical_weight_map"]
        horizontal_changes = np.abs(np.diff(classical_map, axis=1))
        vertical_changes = np.abs(np.diff(classical_map, axis=0))

        self.assertLess(float(np.max(horizontal_changes)), 0.03)
        self.assertLess(float(np.max(vertical_changes)), 1e-6)

    def test_output_is_deterministic(self) -> None:
        first_image, first_details = spatial_adaptive_fusion(
            self.classical, self.dl, self.input_image, return_weights=True
        )
        second_image, second_details = spatial_adaptive_fusion(
            self.classical, self.dl, self.input_image, return_weights=True
        )

        np.testing.assert_array_equal(first_image, second_image)
        np.testing.assert_array_equal(
            first_details["classical_weight_map"],
            second_details["classical_weight_map"],
        )
        np.testing.assert_array_equal(
            first_details["dl_weight_map"], second_details["dl_weight_map"]
        )

    def test_identical_restoration_outputs_remain_unchanged(self) -> None:
        same_image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape((32, 32, 3))

        fused = spatial_adaptive_fusion(
            same_image, same_image.copy(), self.input_image
        )

        np.testing.assert_array_equal(fused, same_image)

    def test_return_modes(self) -> None:
        image_only = spatial_adaptive_fusion(
            self.classical, self.dl, self.input_image
        )
        image_and_weights = spatial_adaptive_fusion(
            self.classical, self.dl, self.input_image, return_weights=True
        )

        self.assertIsInstance(image_only, np.ndarray)
        self.assertIsInstance(image_and_weights, tuple)
        self.assertEqual(
            set(image_and_weights[1]),
            {
                "classical_weight_map",
                "dl_weight_map",
                "input_statistics",
                "global_baseline_weights",
            },
        )

    def test_invalid_dimensions_dtypes_and_return_flag_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "HxWx3 RGB"):
            spatial_adaptive_fusion(
                np.zeros((32, 32), dtype=np.uint8), self.dl, self.input_image
            )
        with self.assertRaisesRegex(TypeError, "dtype uint8"):
            spatial_adaptive_fusion(
                self.classical.astype(np.float32), self.dl, self.input_image
            )
        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            spatial_adaptive_fusion(
                self.classical,
                self.dl,
                np.zeros((31, 32, 3), dtype=np.uint8),
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            empty = np.zeros((0, 32, 3), dtype=np.uint8)
            spatial_adaptive_fusion(empty, empty, empty)
        with self.assertRaisesRegex(TypeError, "return_weights"):
            spatial_adaptive_fusion(
                self.classical, self.dl, self.input_image, return_weights=1
            )

    def test_tiny_images_are_supported(self) -> None:
        classical = np.full((1, 1, 3), 20, dtype=np.uint8)
        dl = np.full((1, 1, 3), 220, dtype=np.uint8)
        input_image = np.full((1, 1, 3), 80, dtype=np.uint8)

        fused, details = spatial_adaptive_fusion(
            classical, dl, input_image, return_weights=True
        )

        self.assertEqual(fused.shape, (1, 1, 3))
        self.assertEqual(details["classical_weight_map"].shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
