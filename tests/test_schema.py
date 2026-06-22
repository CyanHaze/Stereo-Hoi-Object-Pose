import unittest

import numpy as np

from stereo_hoi.data import CameraModel, FrameBundle, PoseEstimate, ViewObservation


class SchemaTests(unittest.TestCase):
    def test_frame_bundle_rejects_duplicate_camera_ids(self):
        camera = CameraModel("left", np.eye(3), np.eye(4), 1920, 1080)
        view_a = ViewObservation("00000", camera)
        view_b = ViewObservation("00000", camera)
        with self.assertRaises(ValueError):
            FrameBundle("00000", (view_a, view_b))

    def test_pose_estimate_validates_confidence(self):
        with self.assertRaises(ValueError):
            PoseEstimate("00000", np.eye(4), confidence=1.1)


if __name__ == "__main__":
    unittest.main()

