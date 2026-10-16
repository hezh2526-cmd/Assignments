"""
A from-scratch implementation of the NSGA-II algorithm.

Only NumPy is used for numerical storage. Non-dominated sorting, crowding
distance, binary tournament selection, SBX crossover, polynomial mutation,
and elitist environmental selection are implemented in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .problems import Problem

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def dominates(a: FloatArray, b: FloatArray) -> bool:
    """Return True when minimization objective vector a Pareto-dominates b."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: FloatArray) -> tuple[list[IntArray], IntArray]:
    """Deb et al.'s O(M*N^2) fast non-dominated sorting procedure."""
    objectives = np.asarray(objectives, dtype=float)
    n = len(objectives)
    dominated_sets: list[list[int]] = [[] for _ in range(n)]
    domination_counts = np.zeros(n, dtype=np.int64)
    ranks = np.full(n, -1, dtype=np.int64)
    first_front: list[int] = []

    for p in range(n):
        for q in range(p + 1, n):
            if dominates(objectives[p], objectives[q]):
                dominated_sets[p].append(q)
                domination_counts[q] += 1
            elif dominates(objectives[q], objectives[p]):
                dominated_sets[q].append(p)
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            ranks[p] = 0
            first_front.append(p)

    fronts: list[IntArray] = []
    current = first_front
    rank = 0
    while current:
        fronts.append(np.asarray(current, dtype=np.int64))
        next_front: list[int] = []
        for p in current:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    ranks[q] = rank + 1
                    next_front.append(q)
        rank += 1
        current = next_front
    return fronts, ranks


def crowding_distance(objectives: FloatArray, front: IntArray) -> FloatArray:
    """Calculate normalized Manhattan crowding distances for one front."""
    distances = np.zeros(len(front), dtype=float)
    if len(front) == 0:
        return distances
    if len(front) <= 2:
        distances.fill(np.inf)
        return distances

    front_values = objectives[front]
    for objective_index in range(objectives.shape[1]):
        order = np.argsort(front_values[:, objective_index], kind="stable")
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        minimum = front_values[order[0], objective_index]
        maximum = front_values[order[-1], objective_index]
        if maximum == minimum:
            continue
        for position in range(1, len(front) - 1):
            previous_value = front_values[order[position - 1], objective_index]
            next_value = front_values[order[position + 1], objective_index]
            distances[order[position]] += (next_value - previous_value) / (maximum - minimum)
    return distances


def rank_and_crowding(objectives: FloatArray) -> tuple[list[IntArray], IntArray, FloatArray]:
    fronts, ranks = fast_non_dominated_sort(objectives)
    distances = np.zeros(len(objectives), dtype=float)
    for front in fronts:
        distances[front] = crowding_distance(objectives, front)
    return fronts, ranks, distances


@dataclass(frozen=True)
class NSGA2Result:
    decisions: FloatArray
    objectives: FloatArray
    population: FloatArray
    population_objectives: FloatArray
    history: list[dict[str, float]]


class NSGA2:
    """NSGA-II for bounded, continuous, minimization problems."""

    def __init__(
        self,
        pop_size: int = 100,
        n_gen: int = 200,
        crossover_prob: float = 0.9,
        mutation_prob: float | None = None,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
        seed: int | None = None,
    ) -> None:
        if pop_size < 4 or pop_size % 2 != 0:
            raise ValueError("pop_size must be an even integer of at least 4")
        if n_gen < 1:
            raise ValueError("n_gen must be positive")
        if not 0.0 <= crossover_prob <= 1.0:
            raise ValueError("crossover_prob must be in [0, 1]")
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.rng = np.random.default_rng(seed)

    def _tournament(self, ranks: IntArray, distances: FloatArray) -> int:
        a, b = self.rng.integers(0, len(ranks), size=2)
        if ranks[a] < ranks[b]:
            return int(a)
        if ranks[b] < ranks[a]:
            return int(b)
        if distances[a] > distances[b]:
            return int(a)
        if distances[b] > distances[a]:
            return int(b)
        return int(a if self.rng.random() < 0.5 else b)

    def _sbx(self, parent1: FloatArray, parent2: FloatArray, problem: Problem) -> tuple[FloatArray, FloatArray]:
        child1 = parent1.copy()
        child2 = parent2.copy()
        if self.rng.random() > self.crossover_prob:
            return child1, child2

        for i in range(problem.n_var):
            if self.rng.random() > 0.5 or abs(parent1[i] - parent2[i]) <= 1e-14:
                continue
            x1, x2 = sorted((parent1[i], parent2[i]))
            lower, upper = problem.lower[i], problem.upper[i]
            rand = self.rng.random()

            beta = 1.0 + 2.0 * (x1 - lower) / (x2 - x1)
            alpha = 2.0 - beta ** (-(self.eta_c + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (self.eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1.0))
            c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

            beta = 1.0 + 2.0 * (upper - x2) / (x2 - x1)
            alpha = 2.0 - beta ** (-(self.eta_c + 1.0))
            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (self.eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1.0))
            c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

            c1 = np.clip(c1, lower, upper)
            c2 = np.clip(c2, lower, upper)
            if self.rng.random() < 0.5:
                child1[i], child2[i] = c2, c1
            else:
                child1[i], child2[i] = c1, c2
        return child1, child2

    def _mutate(self, individual: FloatArray, problem: Problem) -> FloatArray:
        child = individual.copy()
        probability = self.mutation_prob if self.mutation_prob is not None else 1.0 / problem.n_var
        for i in range(problem.n_var):
            if self.rng.random() >= probability:
                continue
            lower, upper = problem.lower[i], problem.upper[i]
            if upper <= lower:
                continue
            delta1 = (child[i] - lower) / (upper - lower)
            delta2 = (upper - child[i]) / (upper - lower)
            rand = self.rng.random()
            mut_pow = 1.0 / (self.eta_m + 1.0)
            if rand <= 0.5:
                xy = 1.0 - delta1
                value = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (self.eta_m + 1.0)
                deltaq = value**mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                value = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (self.eta_m + 1.0)
                deltaq = 1.0 - value**mut_pow
            child[i] = np.clip(child[i] + deltaq * (upper - lower), lower, upper)
        return child

    def _offspring(self, population: FloatArray, objectives: FloatArray, problem: Problem) -> FloatArray:
        _, ranks, distances = rank_and_crowding(objectives)
        children: list[FloatArray] = []
        while len(children) < self.pop_size:
            p1 = population[self._tournament(ranks, distances)]
            p2 = population[self._tournament(ranks, distances)]
            c1, c2 = self._sbx(p1, p2, problem)
            children.append(self._mutate(c1, problem))
            children.append(self._mutate(c2, problem))
        return np.asarray(children[: self.pop_size])

    def _environmental_selection(self, population: FloatArray, objectives: FloatArray) -> tuple[FloatArray, FloatArray]:
        fronts, _, _ = rank_and_crowding(objectives)
        selected: list[int] = []
        for front in fronts:
            remaining = self.pop_size - len(selected)
            if len(front) <= remaining:
                selected.extend(front.tolist())
            else:
                distances = crowding_distance(objectives, front)
                order = np.argsort(-distances, kind="stable")
                selected.extend(front[order[:remaining]].tolist())
                break
        indices = np.asarray(selected, dtype=np.int64)
        return population[indices], objectives[indices]

    def run(
        self,
        problem: Problem,
        callback: Callable[[int, FloatArray, FloatArray], None] | None = None,
    ) -> NSGA2Result:
        population = self.rng.uniform(problem.lower, problem.upper, size=(self.pop_size, problem.n_var))
        objectives = problem.evaluate(population)
        history: list[dict[str, float]] = []

        for generation in range(1, self.n_gen + 1):
            offspring = self._offspring(population, objectives, problem)
            offspring_objectives = problem.evaluate(offspring)
            combined = np.vstack((population, offspring))
            combined_objectives = np.vstack((objectives, offspring_objectives))
            population, objectives = self._environmental_selection(combined, combined_objectives)
            fronts, _, _ = rank_and_crowding(objectives)
            history.append({"generation": float(generation), "front_size": float(len(fronts[0]))})
            if callback is not None:
                callback(generation, population, objectives)

        fronts, _, _ = rank_and_crowding(objectives)
        first = fronts[0]
        order = np.argsort(objectives[first, 0])
        first = first[order]
        return NSGA2Result(
            decisions=population[first].copy(),
            objectives=objectives[first].copy(),
            population=population.copy(),
            population_objectives=objectives.copy(),
            history=history,
        )