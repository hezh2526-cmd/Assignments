"""Standard bi-objective benchmark problems used by the experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Problem:

    name: str
    n_var: int
    n_obj: int
    lower: FloatArray
    upper: FloatArray
    evaluate_fn: Callable[[FloatArray], FloatArray]
    pareto_front_fn: Callable[[int], FloatArray]

    def evaluate(self, x: FloatArray) -> FloatArray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_var:
            raise ValueError(f"Expected shape (n, {self.n_var}), got {x.shape}")
        return self.evaluate_fn(x)

    def pareto_front(self, n_points: int = 1000) -> FloatArray:
        if n_points < 2:
            raise ValueError("n_points must be at least 2")
        return self.pareto_front_fn(n_points)


def _zdt_values(x: FloatArray, variant: str) -> FloatArray:
    f1 = x[:, 0]
    g = 1.0 + 9.0 * np.mean(x[:, 1:], axis=1)
    ratio = f1 / g
    if variant == "zdt1":
        h = 1.0 - np.sqrt(ratio)
    elif variant == "zdt2":
        h = 1.0 - ratio**2
    elif variant == "zdt3":
        h = 1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1)
    else:
        raise ValueError(f"Unknown ZDT variant: {variant}")
    return np.column_stack((f1, g * h))


def _continuous_front(variant: str, n_points: int) -> FloatArray:
    f1 = np.linspace(0.0, 1.0, n_points)
    if variant == "zdt1":
        f2 = 1.0 - np.sqrt(f1)
    else:
        f2 = 1.0 - f1**2
    return np.column_stack((f1, f2))


def _zdt3_front(n_points: int) -> FloatArray:
    # Analytically known disconnected intervals of the ZDT3 Pareto front.
    intervals = (
        (0.0, 0.0830015349),
        (0.1822287280, 0.2577623634),
        (0.4093136748, 0.4538821041),
        (0.6183967944, 0.6525117038),
        (0.8233317983, 0.8518328654),
    )
    lengths = np.array([b - a for a, b in intervals])
    counts = np.maximum(2, np.round(n_points * lengths / lengths.sum()).astype(int))
    pieces = [np.linspace(a, b, int(count), endpoint=True) for (a, b), count in zip(intervals, counts)]
    f1 = np.concatenate(pieces)
    f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
    return np.column_stack((f1, f2))


def get_problem(name: str, n_var: int = 30) -> Problem:
    """Create ZDT1, ZDT2, or ZDT3 with variables in [0, 1]."""
    key = name.lower()
    if key not in {"zdt1", "zdt2", "zdt3"}:
        raise ValueError("name must be one of: zdt1, zdt2, zdt3")
    if n_var < 2:
        raise ValueError("n_var must be at least 2")

    if key == "zdt3":
        front_fn = _zdt3_front
    else:
        front_fn = lambda n: _continuous_front(key, n)

    return Problem(
        name=key.upper(),
        n_var=n_var,
        n_obj=2,
        lower=np.zeros(n_var),
        upper=np.ones(n_var),
        evaluate_fn=lambda x: _zdt_values(x, key),
        pareto_front_fn=front_fn,
    )
