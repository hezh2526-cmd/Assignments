"""Command-line entry point for the NSGA-II reproduction project."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the from-scratch NSGA-II implementation.")
    parser.add_argument("--problems", nargs="+", default=["zdt1"], choices=["zdt1", "zdt2", "zdt3"])
    parser.add_argument("--runs", type=int, default=1, help="Independent runs per problem")
    parser.add_argument("--population", type=int, default=100, help="Population size (even number)")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")
    parser.add_argument("--seed", type=int, default=1, help="Seed of the first run")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiments(
        problem_names=args.problems,
        runs=args.runs,
        pop_size=args.population,
        generations=args.generations,
        output_dir=args.output,
        seed_start=args.seed,
    )
    print("\nExperiment summary:")
    print(summary.to_string(index=False))
    print(f"\nFiles saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
