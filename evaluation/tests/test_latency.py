"""Unit tests for restoration inference latency measurement."""

import unittest
from unittest.mock import patch

from evaluation.latency import measure_restoration_latency


class RestorationLatencyTests(unittest.TestCase):
    def test_reports_statistics_and_excludes_warmup_runs(self) -> None:
        image = object()
        calls: list[object] = []

        def restore(received_image: object) -> object:
            calls.append(received_image)
            return object()

        with patch(
            "evaluation.latency.time.perf_counter_ns",
            side_effect=(0, 1_000_000, 2_000_000, 5_000_000),
        ):
            result = measure_restoration_latency(
                image,
                restore,
                warmup_runs=1,
                runs=2,
            )

        self.assertEqual(calls, [image, image, image])
        self.assertEqual(result["mean_ms"], 2.0)
        self.assertEqual(result["median_ms"], 2.0)
        self.assertEqual(result["min_ms"], 1.0)
        self.assertEqual(result["max_ms"], 3.0)
        self.assertEqual(result["runs"], 2)

    def test_zero_warmup_explicitly_times_first_call(self) -> None:
        calls = 0

        def restore(_: object) -> None:
            nonlocal calls
            calls += 1

        with patch(
            "evaluation.latency.time.perf_counter_ns",
            side_effect=(100, 2_100_100),
        ):
            result = measure_restoration_latency(
                object(),
                restore,
                warmup_runs=0,
                runs=1,
            )

        self.assertEqual(calls, 1)
        self.assertAlmostEqual(result["mean_ms"], 2.1)

    def test_cuda_device_synchronizes_before_and_after_each_call(self) -> None:
        restore = lambda image: image
        with patch("evaluation.latency._synchronize_if_cuda") as synchronize:
            result = measure_restoration_latency(
                object(),
                restore,
                warmup_runs=2,
                runs=3,
                device="cuda:0",
            )

        self.assertEqual(synchronize.call_count, 2 * (2 + 3))
        self.assertTrue(all(call.args == ("cuda:0",) for call in synchronize.call_args_list))
        self.assertEqual(result["runs"], 3)

    def test_cpu_device_does_not_require_cuda_synchronization(self) -> None:
        with patch("evaluation.latency._synchronize_if_cuda") as synchronize:
            measure_restoration_latency(
                object(),
                lambda image: image,
                warmup_runs=1,
                runs=1,
                device="cpu",
            )

        self.assertEqual(synchronize.call_count, 4)
        self.assertTrue(all(call.args == ("cpu",) for call in synchronize.call_args_list))

    def test_invalid_inputs_and_run_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "image must not be None"):
            measure_restoration_latency(None, lambda image: image)
        with self.assertRaisesRegex(TypeError, "must be callable"):
            measure_restoration_latency(object(), "not callable")
        with self.assertRaisesRegex(ValueError, "runs must be >= 1"):
            measure_restoration_latency(object(), lambda image: image, runs=0)
        with self.assertRaisesRegex(ValueError, "warmup_runs must be >= 0"):
            measure_restoration_latency(
                object(),
                lambda image: image,
                warmup_runs=-1,
            )
        with self.assertRaisesRegex(TypeError, "warmup_runs must be"):
            measure_restoration_latency(
                object(),
                lambda image: image,
                warmup_runs=1.5,
            )
        with self.assertRaisesRegex(TypeError, "runs must be"):
            measure_restoration_latency(
                object(),
                lambda image: image,
                runs=True,
            )

    def test_restoration_exception_is_propagated(self) -> None:
        def fail(_: object) -> None:
            raise LookupError("inference failed")

        with self.assertRaisesRegex(LookupError, "inference failed"):
            measure_restoration_latency(object(), fail)


if __name__ == "__main__":
    unittest.main()
