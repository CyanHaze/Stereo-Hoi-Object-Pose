import unittest

import numpy as np

from stereo_hoi.geometry import (
    camera_from_object,
    invert_transform,
    rotation_geodesic_deg,
    transform_points,
)


class Se3Tests(unittest.TestCase):
    def test_inverse_round_trip(self):
        transform = np.eye(4)
        transform[:3, 3] = [0.2, -0.1, 1.3]
        inverse = invert_transform(transform)
        np.testing.assert_allclose(inverse @ transform, np.eye(4), atol=1e-9)

    def test_camera_from_object_direction(self):
        rig_from_camera = np.eye(4)
        rig_from_camera[0, 3] = 0.1
        rig_from_object = np.eye(4)
        rig_from_object[0, 3] = 0.3
        result = camera_from_object(rig_from_camera, rig_from_object)
        np.testing.assert_allclose(result[:3, 3], [0.2, 0.0, 0.0])

    def test_rotation_geodesic(self):
        rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.assertAlmostEqual(rotation_geodesic_deg(np.eye(3), rotation), 90.0)

    def test_transform_points(self):
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        result = transform_points(transform, np.array([[0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()

