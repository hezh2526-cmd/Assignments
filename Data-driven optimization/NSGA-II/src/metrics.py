"""Quality indicators for two-objective minimization results."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def igd(approximation: FloatArray, reference_front: FloatArray) -> float:
    """Inverted Generational Distance (smaller is better)."""
    approximation = np.asarray(approximation, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    if len(approximation) == 0 or len(reference_front) == 0:
        return float("inf")
    differences = reference_front[:, None, :] - approximation[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def hypervolume_2d(points: FloatArray, reference_point: tuple[float, float] = (1.1, 1.1)) -> float:
    """Exact dominated hypervolume for 2-D minimization objectives."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    reference = np.asarray(reference_point, dtype=float)
    valid = points[np.all(points < reference, axis=1)]
    if len(valid) == 0:
        return 0.0
    valid = valid[np.argsort(valid[:, 0], kind="stable")]
    volume = 0.0
    previous_y = reference[1]
    for x, y in valid:
        if y < previous_y:
            volume += (reference[0] - x) * (previous_y - y)
            previous_y = y
    return float(volume)
