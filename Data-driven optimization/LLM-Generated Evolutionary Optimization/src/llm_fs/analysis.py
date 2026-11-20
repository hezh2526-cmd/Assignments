from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def analyze_results(input_path: Path) -> tuple[Path, list[Path]]:
    frame = pd.read_csv(input_path)
    if frame.empty:
        raise ValueError("Result file is empty.")

    final = (
        frame.sort_values("evaluation")
        .groupby(["dataset", "algorithm", "seed"], as_index=False)
        .tail(1)
    )
    summary = (
        final.groupby(["dataset", "algorithm"], as_index=False)
        .agg(
            runs=("seed", "nunique"),
            mean_best_objective=("best_objective", "mean"),
            std_best_objective=("best_objective", "std"),
            median_best_objective=("best_objective", "median"),
            min_best_objective=("best_objective", "min"),
            max_best_objective=("best_objective", "max"),
            mean_best_accuracy=("best_accuracy", "mean"),
            mean_selected_features=("best_selected_features", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
        )
    )
    duplicate_rates = (
        frame.groupby(["dataset", "algorithm"], as_index=False)["cache_hit"]
        .mean()
        .rename(columns={"cache_hit": "duplicate_rate"})
    )
    summary = summary.merge(duplicate_rates, on=["dataset", "algorithm"], how="left")
    summary["std_best_objective"] = summary["std_best_objective"].fillna(0.0)
    summary_path = input_path.parent / "summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plot_paths: list[Path] = []
    for dataset, subset in frame.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for algorithm, values in subset.groupby("algorithm"):
            pivot = values.pivot_table(
                index="evaluation",
                columns="seed",
                values="best_objective",
                aggfunc="last",
            ).sort_index()
            mean = pivot.mean(axis=1)
            std = pivot.std(axis=1).fillna(0.0)
            ax.plot(mean.index, mean.values, label=algorithm, linewidth=2)
            ax.fill_between(mean.index, mean - std, mean + std, alpha=0.18)
        ax.set_title(f"Convergence on {dataset}")
        ax.set_xlabel("Expensive objective evaluations")
        ax.set_ylabel("Best objective (lower is better)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = input_path.parent / f"convergence_{dataset}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(path)

        algorithms = sorted(final.loc[final["dataset"] == dataset, "algorithm"].unique())
        data = [
            final.loc[
                (final["dataset"] == dataset) & (final["algorithm"] == algorithm),
                "best_objective",
            ].to_numpy()
            for algorithm in algorithms
        ]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(data, labels=algorithms, showmeans=True)
        ax.set_title(f"Final objective on {dataset}")
        ax.set_ylabel("Best objective (lower is better)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        box_path = input_path.parent / f"boxplot_{dataset}.png"
        fig.savefig(box_path, dpi=180)
        plt.close(fig)
        plot_paths.append(box_path)

    return summary_path, plot_paths

