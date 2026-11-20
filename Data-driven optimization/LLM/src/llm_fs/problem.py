from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class Evaluation:
    objective: float
    accuracy: float
    selected_features: int
    selected_ratio: float
    cache_hit: bool


class FeatureSelectionProblem:
    """Expensive wrapper: one mask evaluation performs stratified CV training."""

    def __init__(
        self,
        dataset: str,
        cv_folds: int = 5,
        feature_penalty: float = 0.01,
        seed: int = 0,
        classifier: str = "svc",
    ) -> None:
        if dataset == "breast_cancer":
            bundle = load_breast_cancer()
        elif dataset == "digits":
            bundle = load_digits()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        if classifier != "svc":
            raise ValueError("This reproducible assignment package currently supports classifier='svc'.")

        self.dataset = dataset
        self.X = np.asarray(bundle.data, dtype=float)
        self.y = np.asarray(bundle.target)
        self.n_features = self.X.shape[1]
        self.cv_folds = int(cv_folds)
        self.feature_penalty = float(feature_penalty)
        self.seed = int(seed)
        self.classifier = classifier
        self.cache: dict[tuple[int, ...], Evaluation] = {}
        self.true_evaluations = 0
        self.requests = 0

        self._cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.seed,
        )

    def _normalize_mask(self, mask: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(mask), dtype=float).reshape(-1)
        if arr.size != self.n_features:
            raise ValueError(f"Expected {self.n_features} variables, got {arr.size}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Mask contains NaN or infinity.")
        return (arr >= 0.5).astype(np.int8)

    def evaluate(self, mask: Iterable[float]) -> Evaluation:
        binary = self._normalize_mask(mask)
        key = tuple(int(v) for v in binary)
        self.requests += 1
        if key in self.cache:
            cached = self.cache[key]
            return Evaluation(
                cached.objective,
                cached.accuracy,
                cached.selected_features,
                cached.selected_ratio,
                True,
            )

        selected = int(binary.sum())
        ratio = selected / self.n_features
        if selected == 0:
            result = Evaluation(1.0 + self.feature_penalty, 0.0, 0, 0.0, False)
        else:
            estimator = make_pipeline(
                StandardScaler(),
                SVC(C=10.0, gamma="scale", kernel="rbf", cache_size=500),
            )
            scores = cross_val_score(
                estimator,
                self.X[:, binary.astype(bool)],
                self.y,
                cv=self._cv,
                scoring="accuracy",
                n_jobs=1,
                error_score="raise",
            )
            accuracy = float(np.mean(scores))
            objective = float(1.0 - accuracy + self.feature_penalty * ratio)
            result = Evaluation(objective, accuracy, selected, ratio, False)

        self.cache[key] = result
        self.true_evaluations += 1
        return result

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "samples": int(self.X.shape[0]),
            "features": int(self.n_features),
            "classes": int(np.unique(self.y).size),
            "cv_folds": self.cv_folds,
            "feature_penalty": self.feature_penalty,
        }
