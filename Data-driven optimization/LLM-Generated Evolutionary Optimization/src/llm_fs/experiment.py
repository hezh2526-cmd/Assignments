from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .problem import FeatureSelectionProblem
from .validator import load_generated_optimizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RandomSearch:
    """Minimal non-learning baseline; not the proposed algorithm."""

    def __init__(self, dimension: int, budget: int, seed: int) -> None:
        self.dimension = dimension
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        self.seen: set[tuple[int, ...]] = set()
        self.best_x = None
        self.best_y = float("inf")

    def ask(self, n_points: int) -> np.ndarray:
        points: list[np.ndarray] = []
        attempts = 0
        while len(points) < n_points and attempts < 1000:
            x = self.rng.integers(0, 2, size=self.dimension, dtype=np.int8)
            if not x.any():
                x[self.rng.integers(0, self.dimension)] = 1
            key = tuple(int(v) for v in x)
            attempts += 1
            if key not in self.seen:
                self.seen.add(key)
                points.append(x)
        return np.asarray(points)

    def tell(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(y) and float(np.min(y)) < self.best_y:
            index = int(np.argmin(y))
            self.best_y = float(y[index])
            self.best_x = np.asarray(X[index], dtype=np.int8).copy()


def _make_optimizer(name: str, algorithm_path: Path, dimension: int, budget: int, seed: int):
    if name == "random_search":
        return RandomSearch(dimension, budget, seed)
    if name == "generated":
        optimizer_class = load_generated_optimizer(algorithm_path)
        return optimizer_class([(0, 1)] * dimension, ["binary"] * dimension, budget, seed)
    raise ValueError(f"Unknown algorithm: {name}")


def run_one(
    dataset: str,
    algorithm_name: str,
    algorithm_path: Path,
    seed: int,
    budget: int,
    batch_size: int,
    cv_folds: int,
    feature_penalty: float,
    classifier: str,
) -> list[dict[str, object]]:
    problem = FeatureSelectionProblem(dataset, cv_folds, feature_penalty, seed, classifier)
    optimizer = _make_optimizer(algorithm_name, algorithm_path, problem.n_features, budget, seed)
    rows: list[dict[str, object]] = []
    best_y = float("inf")
    best_accuracy = 0.0
    best_selected = problem.n_features
    used = 0
    started = time.perf_counter()

    while used < budget:
        requested = min(batch_size, budget - used)
        X = np.asarray(optimizer.ask(requested), dtype=float)
        if X.ndim != 2 or X.shape[1] != problem.n_features or X.shape[0] == 0:
            raise RuntimeError(f"{algorithm_name}.ask returned invalid shape {X.shape}")
        X = X[:requested]
        evaluations = [problem.evaluate(x) for x in X]
        y = np.asarray([item.objective for item in evaluations], dtype=float)
        optimizer.tell(X, y)

        for x, result in zip(X, evaluations):
            used += 1
            if result.objective < best_y:
                best_y = result.objective
                best_accuracy = result.accuracy
                best_selected = result.selected_features
            rows.append(
                {
                    "dataset": dataset,
                    "algorithm": algorithm_name,
                    "algorithm_file": algorithm_path.name if algorithm_name == "generated" else "builtin",
                    "seed": seed,
                    "evaluation": used,
                    "objective": result.objective,
                    "accuracy": result.accuracy,
                    "selected_features": result.selected_features,
                    "cache_hit": result.cache_hit,
                    "best_objective": best_y,
                    "best_accuracy": best_accuracy,
                    "best_selected_features": best_selected,
                    "elapsed_seconds": time.perf_counter() - started,
                    "mask": "".join(str(int(v >= 0.5)) for v in x),
                }
            )
    return rows


def run_experiment(config_path: Path, algorithm_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    all_rows: list[dict[str, object]] = []
    for dataset in config["datasets"]:
        for algorithm in config["algorithms"]:
            for seed in config["seeds"]:
                print(f"Running dataset={dataset}, algorithm={algorithm}, seed={seed} ...", flush=True)
                all_rows.extend(
                    run_one(
                        dataset=dataset,
                        algorithm_name=algorithm,
                        algorithm_path=algorithm_path,
                        seed=int(seed),
                        budget=int(config["budget"]),
                        batch_size=int(config["batch_size"]),
                        cv_folds=int(config["cv_folds"]),
                        feature_penalty=float(config["feature_penalty"]),
                        classifier=str(config["classifier"]),
                    )
                )
    output = PROJECT_ROOT / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output

