import unittest

import numpy as np

from stereo_hoi.evaluation.pose import add_errors, pose_errors
from stereo_hoi.evaluation.trajectory import evaluate_trajectory


def _poses(count: int) -> np.ndarray:
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, 0, 3] = np.arange(count) * 0.01
    return poses


class EvaluationTests(unittest.TestCase):
    def test_pose_errors(self):
        ground_truth = _poses(2)
        predicted = ground_truth.copy()
        predicted[1, 1, 3] += 0.03
        translation, rotation = pose_errors(predicted, ground_truth)
        np.testing.assert_allclose(translation, [0.0, 0.03])
        np.testing.assert_allclose(rotation, [0.0, 0.0])

    def test_add_error_matches_translation_offset(self):
        points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        ground_truth = _poses(1)
        predicted = ground_truth.copy()
        predicted[0, 2, 3] = 0.02
        np.testing.assert_allclose(add_errors(points, predicted, ground_truth), [0.02])

    def test_failure_run_statistics(self):
        ground_truth = _poses(6)
        predicted = ground_truth.copy()
        predicted[2:5, 0, 3] += 0.2
        report = evaluate_trajectory(
            predicted,
            ground_truth,
            failure_translation_m=0.05,
            failure_rotation_deg=30.0,
        )
        self.assertEqual(report["failure_episodes"], 1)
        self.assertEqual(report["longest_failure_run"], 3)
        self.assertEqual(report["time_to_first_failure_frames"], 2)
        self.assertAlmostEqual(report["success_rate"], 0.5)

    def test_relative_error_ignores_constant_reference_transform(self):
        ground_truth = _poses(5)
        reference = np.eye(4)
        reference[:3, 3] = [0.4, -0.2, 0.8]
        predicted = reference[None] @ ground_truth
        report = evaluate_trajectory(predicted, ground_truth)
        self.assertLess(report["relative_translation_m"]["max"], 1e-9)
        self.assertLess(report["relative_rotation_deg"]["max"], 1e-9)


if __name__ == "__main__":
    unittest.main()

