"""Tests for baseline weighted restoration fusion."""

import unittest

import numpy as np

from hybrid.fusion import weighted_fusion


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


if __name__ == "__main__":
    unittest.main()
