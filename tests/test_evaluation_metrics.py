import unittest

import numpy as np

from maxim.evaluation_metrics import (
    aggregate_task_psnr,
    aggregate_task_ssim,
    build_best_metric_payload,
    read_selection_value,
)


class AggregateTaskPsnrTest(unittest.TestCase):
    def test_unequal_task_sizes_keep_weighted_and_macro_distinct(self):
        task_psnr = np.array([10.0, 20.0, 40.0])
        task_count = np.array([1, 3, 6])
        result = aggregate_task_psnr(task_psnr * task_count, task_count)

        self.assertAlmostEqual(result["psnr_weighted"], 31.0)
        self.assertAlmostEqual(result["psnr_macro"], 70.0 / 3.0)
        np.testing.assert_allclose(result["task_psnr"], task_psnr)
        np.testing.assert_array_equal(result["task_count"], task_count)

    def test_absent_task_is_nan_and_excluded_from_macro(self):
        result = aggregate_task_psnr([20.0, 0.0, 80.0], [1, 0, 2])

        self.assertAlmostEqual(result["psnr_weighted"], 100.0 / 3.0)
        self.assertAlmostEqual(result["psnr_macro"], 30.0)
        self.assertTrue(np.isnan(result["task_psnr"][1]))

    def test_reproduces_k8_epoch_27(self):
        task_psnr = np.array([24.23, 45.20, 37.19, 25.74, 21.64])
        task_count = np.array([3150, 101, 767, 400, 30])
        result = aggregate_task_psnr(task_psnr * task_count, task_count)

        self.assertAlmostEqual(result["psnr_weighted"], 27.06, places=2)
        self.assertAlmostEqual(result["psnr_macro"], 30.80, places=2)

    def test_rejects_empty_evaluation(self):
        with self.assertRaises(ValueError):
            aggregate_task_psnr([0.0, 0.0], [0, 0])


class AggregateTaskSsimTest(unittest.TestCase):
    def test_unequal_task_sizes_keep_weighted_and_macro_distinct(self):
        task_ssim = np.array([0.5, 0.8, 0.95])
        task_count = np.array([1, 3, 6])
        result = aggregate_task_ssim(task_ssim * task_count, task_count)

        self.assertAlmostEqual(result["ssim_weighted"], 0.86)
        self.assertAlmostEqual(result["ssim_macro"], 0.75)
        np.testing.assert_allclose(result["task_ssim"], task_ssim)
        np.testing.assert_array_equal(result["task_count"], task_count)

    def test_absent_task_is_nan_and_excluded_from_macro(self):
        result = aggregate_task_ssim([0.5, 0.0, 1.8], [1, 0, 2])

        self.assertAlmostEqual(result["ssim_weighted"], 2.3 / 3.0)
        self.assertAlmostEqual(result["ssim_macro"], 0.7)
        self.assertTrue(np.isnan(result["task_ssim"][1]))


class BestMetricPayloadTest(unittest.TestCase):
    def test_payload_records_selection_protocol_and_all_task_data(self):
        metrics = aggregate_task_psnr([10.0, 60.0], [1, 3])
        payload = build_best_metric_payload(7, metrics, ("a", "b"))

        self.assertEqual(payload["selection_metric"], "psnr_macro")
        self.assertEqual(payload["selection_value"], payload["psnr_macro"])
        self.assertEqual(payload["task_names"], ["a", "b"])
        self.assertEqual(payload["task_count"], [1, 3])
        self.assertEqual(read_selection_value(payload), (15.0, 7))

    def test_rejects_metadata_selected_with_another_metric(self):
        metrics = aggregate_task_psnr([10.0, 60.0], [1, 3])
        payload = build_best_metric_payload(
            7, metrics, ("a", "b"), selection_metric="psnr_weighted"
        )
        with self.assertRaises(ValueError):
            read_selection_value(payload)

    def test_payload_includes_ssim_when_validation_provides_it(self):
        metrics = aggregate_task_psnr([10.0, 60.0], [1, 3])
        metrics.update(aggregate_task_ssim([0.5, 2.7], [1, 3]))

        payload = build_best_metric_payload(7, metrics, ("a", "b"))

        self.assertAlmostEqual(payload["ssim_weighted"], 0.8)
        self.assertAlmostEqual(payload["ssim_macro"], 0.7)
        self.assertEqual(payload["task_ssim"], [0.5, 0.9])
        self.assertEqual(payload["ssim_config"]["filter_size"], 11)

    def test_historical_payload_without_ssim_remains_valid(self):
        metrics = aggregate_task_psnr([10.0, 60.0], [1, 3])
        payload = build_best_metric_payload(7, metrics, ("a", "b"))

        self.assertNotIn("ssim_macro", payload)
        self.assertNotIn("task_ssim", payload)


if __name__ == "__main__":
    unittest.main()
