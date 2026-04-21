import unittest
import numpy as np

from otb.evaluate import compute_success_curve


class TestOtbEvaluate(unittest.TestCase):
    def test_success_curve_uses_inclusive_threshold(self):
        ious = np.array([0.0, 0.2, 0.8], dtype=np.float32)
        thresholds = np.array([0.0, 0.5], dtype=np.float32)

        curve = compute_success_curve(ious, thresholds)

        np.testing.assert_allclose(curve, np.array([1.0, 1.0 / 3.0]))


if __name__ == "__main__":
    unittest.main()
