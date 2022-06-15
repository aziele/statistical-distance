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

    def test_acc(self):
        correct_values = [0.385, 0.705, 0.63, 0.345, 0]
        self.func_test(self.d.acc, correct_values)

    def test_braycurtis(self):
        correct_values = [0.2526316, 0.47, 0.42, 0.23, 0]
        self.func_test(self.d.braycurtis, correct_values)

    def test_canberra(self):
        correct_values = [0.757169, 1.660306, 1.661341, 0.467908, 0]
        self.func_test(self.d.canberra, correct_values)

    def test_chebyshev(self):
        correct_values = [0.29, 0.47, 0.42, 0.23, 0]
        self.func_test(self.d.chebyshev, correct_values)

    def test_czekanowski(self):
        correct_values = [0.2526316, 0.47, 0.42, 0.23, 0]
        self.func_test(self.d.czekanowski, correct_values)

    def test_divergence(self):
        correct_values = [0.556360, 2.5588168, 2.5007261, 0.222638, 0]
        self.func_test(self.d.divergence, correct_values)

    def test_euclidean(self):
        correct_values = [0.341467, 0.602163, 0.517494, 0.325269, 0]
        self.func_test(self.d.euclidean, correct_values)

    def test_gower(self):
        correct_values = [0.16, 0.313333, 0.28, 0.153333, 0]
        self.func_test(self.d.gower, correct_values) 

    def test_manhattan(self):
        correct_values = [0.48, 0.94, 0.84, 0.46, 0]
        self.func_test(self.d.manhattan, correct_values)

    def test_kl_divergence(self):
        correct_values = [0.077513, 12.480004, 0.523377, 0.112098, 0]
        self.func_test(self.d.kl_divergence, correct_values)

    def test_kulczynski(self):
        correct_values = [0.676056, 1.773585, 1.448276, 0.597403, 0]
        self.func_test(self.d.kulczynski, correct_values)

    def test_kumarjohnson(self):
        correct_values = [0.910009, 1.549443, 1.2375098, 0.470667, 0]
        self.func_test(self.d.kumarjohnson, correct_values)

    def test_lorentzian(self):
        correct_values = [0.430107, 0.797107, 0.730804, 0.414028, 0]
        self.func_test(self.d.lorentzian, correct_values)

    def test_neyman_chisq(self):
        correct_values = [0.490779, 1.349947, 0.350859, 0.213737, 0]
        self.func_test(self.d.neyman_chisq, correct_values)

    def test_pearson_chisq(self):
        correct_values = [0.3485125, 0.355905, 1.0134, 0.243107, 0]
        self.func_test(self.d.pearson_chisq, correct_values)

    def test_soergel(self):
        correct_values = [0.403361, 0.639456, 0.591549, 0.373984, 0]
        self.func_test(self.d.soergel, correct_values) 

    def test_sqeuclidean(self):
        correct_values = [0.1166, 0.3626, 0.2678, 0.1058, 0]
        self.func_test(self.d.sqeuclidean, correct_values)

    def test_taneja(self):
        correct_values = [0.048336466995935856, 3.100094, 2.142089, 0.027711, 0]
        self.func_test(self.d.taneja, correct_values)

unittest.main()