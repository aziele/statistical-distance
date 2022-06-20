import numpy as np
import unittest

from distance import Distance

# Test case 1: non-zero values in u and v.
uv1 = [
   np.array([0.33, 0.21, 0.46]),
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

    def test_add_chisq(self):
        correct_values = [1.175282, 1.345852, 1.114259, 0.456844, 0]
        self.func_test(self.d.add_chisq, correct_values)

    def test_acc(self):
        correct_values = [0.435, 0.705, 0.63, 0.345, 0]
        self.func_test(self.d.acc, correct_values)

    def test_bhattacharyya(self):
        correct_values = [0.065340, 0.285070, 0.203991, 0.027683, 0]
        self.func_test(self.d.bhattacharyya, correct_values)

    def test_braycurtis(self):
        correct_values = [0.29, 0.47, 0.42, 0.23, 0]
        self.func_test(self.d.braycurtis, correct_values)

    def test_canberra(self):
        correct_values = [0.861335, 1.660306, 1.661341, 0.467908, 0]
        self.func_test(self.d.canberra, correct_values)

    def test_chebyshev(self):
        correct_values = [0.29, 0.47, 0.42, 0.23, 0]
        self.func_test(self.d.chebyshev, correct_values)

    def test_chebyshev_min(self):
        correct_values = [0.01, 0.11, 0.17, 0, 0]
        self.func_test(self.d.chebyshev_min, correct_values)

    def test_clark(self):
        correct_values = [0.598728, 1.131109, 1.118196, 0.333645, 0]
        self.func_test(self.d.clark, correct_values)

    def test_clark_relation(self):
        for u, v in self.vectors:
            self.assertEqual( 
                self.d.clark(u, v), np.sqrt(self.d.divergence(u, v) / 2)
            )

    def test_cosine(self):
        correct_values = [0.216689, 0.370206, 0.272996, 0.097486, 0]
        self.func_test(self.d.cosine, correct_values)

    def test_correlation_pearson(self):
        correct_values = [1.995478, 1.755929, 1.008617, 0.254193, 0]
        self.func_test(self.d.correlation_pearson, correct_values)

    def test_czekanowski(self):
        correct_values = [0.29, 0.47, 0.42, 0.23, 0]
        self.func_test(self.d.czekanowski, correct_values)

    def test_dice(self):
        correct_values = [0.216974, 0.389641, 0.287093, 0.098897, 0]
        self.func_test(self.d.dice, correct_values)

    def test_divergence(self):
        correct_values = [0.7169498, 2.5588168, 2.5007261, 0.222638, 0]
        self.func_test(self.d.divergence, correct_values)

    def test_euclidean(self):
        correct_values = [0.403237, 0.602163, 0.517494, 0.325269, 0]
        self.func_test(self.d.euclidean, correct_values)

    def test_google(self):
        correct_values = [0.29, 0.47, 0.42, 0.23, 0]
        self.func_test(self.d.google, correct_values)

    def test_gower(self):
        correct_values = [0.193333, 0.313333, 0.28, 0.153333, 0]
        self.func_test(self.d.gower, correct_values) 

    def test_hellinger(self):
        correct_values = [0.5029972, 0.996069, 0.859140, 0.330477, 0]
        self.func_test(self.d.hellinger, correct_values) 

    def test_jaccard(self):
        correct_values = [0.356579, 0.560779, 0.446110, 0.179993, 0]
        self.func_test(self.d.jaccard, correct_values) 

    def test_jeffreys(self):
        correct_values = [0.514598, 13.165392, 9.149020, 0.219522, 0]
        self.func_test(self.d.jeffreys, correct_values) 

    def test_jensenshannon_divergence(self):
        correct_values = [0.062220, 0.191254, 0.145167, 0.027169, 0]
        self.func_test(self.d.jensenshannon_divergence, correct_values)        

    def test_jensen_difference(self):
        correct_values = [0.062220, 0.191254, 0.145167, 0.027169, 0]
        self.func_test(self.d.jensen_difference, correct_values) 

    def test_k_divergence(self):
        correct_values = [0.061723, 0.146679, 0.176323, 0.0266046, 0]
        self.func_test(self.d.k_divergence, correct_values)

    def test_kl_divergence(self):
        correct_values = [0.259584, 12.480004, 0.523377, 0.112098, 0]
        self.func_test(self.d.kl_divergence, correct_values)

    def test_kulczynski(self):
        correct_values = [0.816901, 1.773585, 1.448276, 0.597403, 0]
        self.func_test(self.d.kulczynski, correct_values)

    def test_kumarjohnson(self):
        correct_values = [1.297539, 1.549443, 1.2375098, 0.470667, 0]
        self.func_test(self.d.kumarjohnson, correct_values)

    def test_lorentzian(self):
        correct_values = [0.511453, 0.797107, 0.730804, 0.414028, 0]
        self.func_test(self.d.lorentzian, correct_values)

    def test_manhattan(self):
        correct_values = [0.58, 0.94, 0.84, 0.46, 0]
        self.func_test(self.d.manhattan, correct_values)

    def test_marylandbridge(self):
        correct_values = [0.216404, 0.350152, 0.258621, 0.096073, 0]
        self.func_test(self.d.marylandbridge, correct_values)

    def test_matusita(self):
        correct_values = [0.355673, 0.704327, 0.607504, 0.233683, 0]
        self.func_test(self.d.matusita, correct_values)

    def test_max_symmetric_chisq(self):
        correct_values = [0.604068, 1.349947, 1.0134, 0.243107, 0]
        self.func_test(self.d.max_symmetric_chisq, correct_values)

    def test_minkowski(self):
        correct_values = [0.403237, 0.602163, 0.517494, 0.325269, 0]
        self.func_test(self.d.minkowski, correct_values)

    def test_motyka(self):
        correct_values = [0.645, 0.7349999, 0.71, 0.615, 0.5]
        self.func_test(self.d.motyka, correct_values)

    def test_neyman_chisq(self):
        correct_values = [0.571214, 1.349947, 0.350859, 0.213737, 0]
        self.func_test(self.d.neyman_chisq, correct_values)

    def test_nonintersection(self):
        correct_values = [0.29, 0.47, 0.42, 0.23, 0]
        self.func_test(self.d.nonintersection, correct_values)

    def test_pearson_chisq(self):
        correct_values = [0.604068, 0.355905, 1.0134, 0.243107, 0]
        self.func_test(self.d.pearson_chisq, correct_values)

    def test_penroseshape(self):
        correct_values = [0.403237, 0.602163, 0.517494, 0.325269, 0]
        self.func_test(self.d.penroseshape, correct_values)

    def test_soergel(self):
        correct_values = [0.449612, 0.639456, 0.591549, 0.373984, 0]
        self.func_test(self.d.soergel, correct_values) 

    def test_squared_euclidean(self):
        correct_values = [0.1626, 0.3626, 0.2678, 0.1058, 0]
        self.func_test(self.d.squared_euclidean, correct_values)

    def test_squared_chisq(self):
        correct_values = [0.241105, 0.614569, 0.476558, 0.107619, 0]
        self.func_test(self.d.squared_chisq, correct_values)

    def test_squaredchord(self):
        correct_values = [0.126503, 0.496077, 0.369061, 0.054608, 0]
        self.func_test(self.d.squaredchord, correct_values)

    def test_taneja(self):
        correct_values = [0.066429, 3.100094, 2.142089, 0.027711, 0]
        self.func_test(self.d.taneja, correct_values)

    def test_tanimoto(self):
        correct_values = [0.449612, 0.639456, 0.591549, 0.373984, 0]
        self.func_test(self.d.tanimoto, correct_values)

    def test_topsoe(self):
        correct_values = [0.124441, 0.382508, 0.290333, 0.054338, 0]
        self.func_test(self.d.topsoe, correct_values)

    def test_vicis_wave_hedges(self):
        correct_values = [2.967758, 2.410145, 2.195152, 1.229861, 0]
        self.func_test(self.d.vicis_wave_hedges, correct_values)

    def test_vicis_symmetric_chisq(self):
        correct_values = [4.327759, 4.310248, 3.087781, 0.777836, 0]
        self.func_test(self.d.vicis_symmetric_chisq, correct_values)

    def test_wave_hedges(self):
        correct_values = [1.218999, 1.939721, 1.966866, 0.756417, 0]
        self.func_test(self.d.wave_hedges, correct_values)

    def test_rel_acc_manhattan_chebyshev(self):
        for u, v in self.vectors:
            self.assertEqual( 
                self.d.acc(u, v),
                (self.d.manhattan(u, v) + self.d.chebyshev(u, v)) / 2
            )

    def test_rel_braycurtis_manhattan(self):
        for u, v in self.vectors:
            self.assertEqual( 
                self.d.braycurtis(u, v),
                self.d.manhattan(u, v) / 2
            )

    def test_rel_euclidean_square_euclidean(self):
        for u, v in self.vectors:
            self.assertAlmostEqual( 
                self.d.euclidean(u, v)**2,
                self.d.squared_euclidean(u, v), places=6
            )

    def test_google_relation(self):
        for u, v in self.vectors:
            self.assertAlmostEqual( 
                self.d.google(u, v), self.d.manhattan(u, v) / 2, places=6
            )

    def test_rel_gower_manhattan(self):
        for u, v in self.vectors:
            self.assertAlmostEqual( 
                self.d.gower(u, v), self.d.manhattan(u, v) / len(u), places=6
            )

    def test_rel_jensenshannon_divergence_topsoe(self):
        for u, v in self.vectors:
            self.assertAlmostEqual( 
                self.d.jensenshannon_divergence(u, v),
                self.d.topsoe(u, v) / 2, places=6
            )

    def test_rel_matusita_squaredchor(self):
        for u, v in self.vectors:
            self.assertAlmostEqual( 
                self.d.matusita(u, v),
                np.sqrt(self.d.squaredchord(u, v)), places=6
            )

    def test_rel_nonintersection_manhattan(self):
        for u, v in self.vectors:
            self.assertAlmostEqual( 
                self.d.nonintersection(u, v),
                self.d.manhattan(u, v) / 2, places=6
            )

    def test_rel_soergel_tanimoto(self):
        for u, v in self.vectors:
            self.assertAlmostEqual( 
                self.d.soergel(u, v),
                self.d.tanimoto(u, v), places=6
            )

    def test_rel_squaredchord_matusita(self):
        for u, v in self.vectors:
            self.assertAlmostEqual( 
                self.d.squaredchord(u, v),
                self.d.matusita(u, v)**2, places=6
            )

unittest.main()