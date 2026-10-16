from __future__ import annotations

import unittest

import numpy as np

from src.metrics import hypervolume_2d, igd
from src.nsga2 import NSGA2, crowding_distance, fast_non_dominated_sort
from src.problems import get_problem


class NSGA2Tests(unittest.TestCase):
    def test_non_dominated_sort(self) -> None:
        objectives = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0], [3.0, 4.0]])
        fronts, ranks = fast_non_dominated_sort(objectives)
        self.assertEqual(set(fronts[0]), {0, 1, 2, 3})
        self.assertEqual(set(fronts[1]), {4})
        np.testing.assert_array_equal(ranks, [0, 0, 0, 0, 1])

    def test_crowding_boundaries_are_infinite(self) -> None:
        objectives = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        distances = crowding_distance(objectives, np.array([0, 1, 2]))
        self.assertTrue(np.isinf(distances[0]))
        self.assertTrue(np.isinf(distances[2]))
        self.assertAlmostEqual(distances[1], 2.0)

    def test_zdt1_known_points(self) -> None:
        problem = get_problem("zdt1")
        x = np.zeros((2, 30))
        x[1, 0] = 1.0
        np.testing.assert_allclose(problem.evaluate(x), [[0.0, 1.0], [1.0, 0.0]])

    def test_short_run_is_bounded_and_reproducible(self) -> None:
        problem = get_problem("zdt1", n_var=8)
        result1 = NSGA2(pop_size=20, n_gen=5, seed=7).run(problem)
        result2 = NSGA2(pop_size=20, n_gen=5, seed=7).run(problem)
        self.assertEqual(result1.population.shape, (20, 8))
        self.assertTrue(np.all(result1.population >= 0.0))
        self.assertTrue(np.all(result1.population <= 1.0))
        np.testing.assert_allclose(result1.population, result2.population)

    def test_metrics(self) -> None:
        points = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertAlmostEqual(igd(points, points), 0.0)
        self.assertAlmostEqual(hypervolume_2d(points, (2.0, 2.0)), 3.0)


if __name__ == "__main__":
    unittest.main()
