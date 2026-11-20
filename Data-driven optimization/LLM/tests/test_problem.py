import numpy as np

from llm_fs.problem import FeatureSelectionProblem


def test_problem_cache_and_mask_shape():
    problem = FeatureSelectionProblem("breast_cancer", cv_folds=2, seed=0)
    mask = np.ones(problem.n_features)
    first = problem.evaluate(mask)
    second = problem.evaluate(mask)
    assert 0.0 <= first.accuracy <= 1.0
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert problem.true_evaluations == 1

