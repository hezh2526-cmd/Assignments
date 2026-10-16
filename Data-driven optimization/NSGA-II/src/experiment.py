"""Batch experiments, result tables, CSV export, and Pareto-front plots."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import hypervolume_2d, igd
from .nsga2 import NSGA2, NSGA2Result
from .problems import get_problem


def _plot_result(problem_name: str, result: NSGA2Result, output_path: Path) -> None:
    problem = get_problem(problem_name)
    front = problem.pareto_front(1200)
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=150)
    ax.plot(front[:, 0], front[:, 1], color="black", linewidth=1.5, label="True Pareto front")
    ax.scatter(result.objectives[:, 0], result.objectives[:, 1], s=18, facecolors="none", edgecolors="#d62728", label="NSGA-II")
    ax.set(xlabel=r"$f_1$", ylabel=r"$f_2$", title=f"NSGA-II on {problem.name}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_experiments(
    problem_names: list[str],
    runs: int,
    pop_size: int,
    generations: int,
    output_dir: str | Path,
    seed_start: int = 1,
) -> pd.DataFrame:
    """Run repeated experiments and save per-run and summary results."""
    if runs < 1:
        raise ValueError("runs must be positive")
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    solutions_dir = output_dir / "solutions"
    figures_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float | int | str]] = []
    for name in problem_names:
        problem = get_problem(name)
        reference_front = problem.pareto_front(1000)
        best_result: NSGA2Result | None = None
        best_igd = float("inf")
        for run_index in range(runs):
            seed = seed_start + run_index
            algorithm = NSGA2(pop_size=pop_size, n_gen=generations, seed=seed)
            start = time.perf_counter()
            result = algorithm.run(problem)
            elapsed = time.perf_counter() - start
            score_igd = igd(result.objectives, reference_front)
            score_hv = hypervolume_2d(result.objectives)
            records.append(
                {
                    "problem": problem.name,
                    "run": run_index + 1,
                    "seed": seed,
                    "pop_size": pop_size,
                    "generations": generations,
                    "igd": score_igd,
                    "hypervolume": score_hv,
                    "runtime_seconds": elapsed,
                    "nondominated_solutions": len(result.objectives),
                }
            )
            if score_igd < best_igd:
                best_igd = score_igd
                best_result = result
            print(f"{problem.name} run {run_index + 1}/{runs}: IGD={score_igd:.6f}, HV={score_hv:.6f}, time={elapsed:.2f}s")

        assert best_result is not None
        solution_data = np.hstack((best_result.decisions, best_result.objectives))
        columns = [f"x{i + 1}" for i in range(problem.n_var)] + ["f1", "f2"]
        pd.DataFrame(solution_data, columns=columns).to_csv(solutions_dir / f"{name.lower()}_best_solutions.csv", index=False)
        _plot_result(name, best_result, figures_dir / f"{name.lower()}_pareto_front.png")

    details = pd.DataFrame(records)
    details.to_csv(output_dir / "all_runs.csv", index=False)
    summary = (
        details.groupby("problem", sort=False)
        .agg(
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            hypervolume_mean=("hypervolume", "mean"),
            hypervolume_std=("hypervolume", "std"),
            runtime_mean_seconds=("runtime_seconds", "mean"),
            nondominated_mean=("nondominated_solutions", "mean"),
        )
        .reset_index()
    )
    summary = summary.fillna(0.0)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary
