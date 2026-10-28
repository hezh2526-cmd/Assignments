from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn

from config import CONFIG
from datasets import dataset_summary, load_all
from evaluation import run_ratio_k_sweep, run_reference_evaluation, summarize_sweep
from plotting import plot_dataset_overview, plot_reference_metrics, plot_sweep_figures, setup_style, slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fisher LDA and KNN experiments.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project directory containing code, data, and results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    tables_dir = results_dir / "tables"
    dataset_fig_dir = results_dir / "figures" / "dataset"
    sweep_fig_dir = results_dir / "figures" / "sweeps"
    ref_fig_dir = results_dir / "figures" / "reference"
    for path in (tables_dir, dataset_fig_dir, sweep_fig_dir, ref_fig_dir):
        path.mkdir(parents=True, exist_ok=True)

    setup_style()
    started = time.perf_counter()
    bundles = load_all(data_dir, CONFIG.datasets)

    summaries = []
    raw_sweeps = []
    reference_metric_rows = []
    selection_rows = []
    for bundle in bundles.values():
        print(f"[INFO] Dataset: {bundle.display_name} ({bundle.X.shape[0]} x {bundle.X.shape[1]})")
        summaries.append(dataset_summary(bundle))
        plot_dataset_overview(bundle, dataset_fig_dir, CONFIG)

        raw = run_ratio_k_sweep(bundle, CONFIG)
        raw_sweeps.append(raw)

        ref_results, selection = run_reference_evaluation(bundle, CONFIG)
        selection_rows.append(selection)
        plot_reference_metrics(bundle, ref_results, ref_fig_dir, CONFIG)
        for result in ref_results:
            reference_metric_rows.append(
                {
                    "dataset": bundle.display_name,
                    "method": result.method,
                    "selected_k": result.selected_k,
                    "train_ratio": CONFIG.reference_train_ratio,
                    "n_train": len(result.y_train),
                    "n_test": len(result.y_true),
                    **result.metrics,
                }
            )
            method_slug = slug(result.method)
            predictions = pd.DataFrame(
                {
                    "sample_id": result.test_indices,
                    "true_label_id": result.y_true,
                    "true_label": [bundle.class_names[int(x)] for x in result.y_true],
                    "predicted_label_id": result.y_pred,
                    "predicted_label": [bundle.class_names[int(x)] for x in result.y_pred],
                    "correct": result.y_true == result.y_pred,
                }
            )
            predictions.to_csv(
                tables_dir / f"reference_predictions_{bundle.key}_{method_slug}.csv", index=False
            )
            result.report.to_csv(
                tables_dir / f"classification_report_{bundle.key}_{method_slug}.csv", index=False
            )

    summary_df = pd.concat(summaries, ignore_index=True)
    raw_df = pd.concat(raw_sweeps, ignore_index=True)
    sweep_summary_df = summarize_sweep(raw_df)
    reference_df = pd.DataFrame(reference_metric_rows)
    selection_df = pd.concat(selection_rows, ignore_index=True)

    summary_df.to_csv(tables_dir / "dataset_summary.csv", index=False)
    raw_df.to_csv(tables_dir / "sweep_raw.csv", index=False)
    sweep_summary_df.to_csv(tables_dir / "sweep_summary.csv", index=False)
    reference_df.to_csv(tables_dir / "reference_metrics.csv", index=False)
    selection_df.to_csv(tables_dir / "model_selection.csv", index=False)
    plot_sweep_figures(sweep_summary_df, sweep_fig_dir, CONFIG)

    elapsed = time.perf_counter() - started
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "config": CONFIG.to_dict(),
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] Results written to: {results_dir}")


if __name__ == "__main__":
    main()

