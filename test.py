import numpy as np
import unittest

from distance import Distance

# Test case 1: non-zero values in u and v.
uv1 = [
   np.array([0.33, 0.21, 0.36]),
   np.array([0.32, 0.50, 0.18])
]

# Test case 2: zero value in v.
uv2 = [
   np.array([0.41, 0.23, 0.36]),
   np.array([0.30, 0.70, 0])
]

# Test case 3: zero value in u.
uv3 = [
   np.array([0.33, 0.67, 0]),
   np.array([0.50, 0.25, 0.25])
]

# Test case 4: zero value in u and v.
uv4 = [
   np.array([0.45, 0.55, 0]),
   np.array([0.68, 0.32, 0])
]

# Test case 5: u and v are identical.
uv5 = [
   np.array([0.20, 0.40, 0.40]),
   np.array([0.20, 0.40, 0.40])
]

class TestDistance(unittest.TestCase):

    def setUp(self):
        self.d = Distance()
        self.vectors = [uv1, uv2, uv3, uv4, uv5]

    def func_test(self, func, correct_values):
        for i, (u, v) in enumerate(self.vectors):
            self.assertAlmostEqual( 
                func(u, v), correct_values[i], places=6
            )

    def test_chebyshev(self):
        correct_values = [0.29, 0.47, 0.42, 0.23, 0]
        self.func_test(self.d.chebyshev, correct_values)

    def test_euclidean(self):
        correct_values = [0.341467, 0.602163, 0.517494, 0.325269, 0]
        self.func_test(self.d.euclidean, correct_values)  

    def test_manhattan(self):
        correct_values = [0.48, 0.94, 0.84, 0.46, 0]
        self.func_test(self.d.manhattan, correct_values)            

    def test_sqeuclidean(self):
        correct_values = [0.1166, 0.3626, 0.2678, 0.1058, 0]
        self.func_test(self.d.sqeuclidean, correct_values)  

unittest.main()