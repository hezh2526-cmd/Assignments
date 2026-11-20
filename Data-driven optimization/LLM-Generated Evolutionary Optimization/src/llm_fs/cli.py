from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import analyze_results
from .experiment import PROJECT_ROOT, run_experiment
from .validator import validation_report


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-generated expensive feature-selection research package")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate generated optimizer safely")
    validate_parser.add_argument("--algorithm", default="generated_algorithms/llm_generated_optimizer.py")

    experiment_parser = subparsers.add_parser("experiment", help="Run controlled experiments")
    experiment_parser.add_argument("--config", default="configs/quick.json")
    experiment_parser.add_argument("--algorithm", default="generated_algorithms/llm_generated_optimizer.py")

    analyze_parser = subparsers.add_parser("analyze", help="Create summary tables and plots")
    analyze_parser.add_argument("--input", default="results/raw_results.csv")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        path = _path(args.algorithm)
        print(validation_report(path))
    elif args.command == "experiment":
        output = run_experiment(_path(args.config), _path(args.algorithm))
        print(f"Results saved: {output}")
    elif args.command == "analyze":
        summary, plots = analyze_results(_path(args.input))
        print(f"Summary saved: {summary}")
        for plot in plots:
            print(f"Plot saved: {plot}")

