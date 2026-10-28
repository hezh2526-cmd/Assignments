from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ExperimentConfig
from datasets import DatasetBundle
from evaluation import ReferenceResult


COLORS = {"Fisher LDA": "#1f77b4", "KNN": "#ff7f0e", "Fisher+KNN": "#2ca02c"}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.autolayout": True, "axes.titleweight": "bold"})


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def save(fig: plt.Figure, path: Path, cfg: ExperimentConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(f".{cfg.figure_format}"), dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_overview(bundle: DatasetBundle, out: Path, cfg: ExperimentConfig) -> None:
    counts = pd.Series(bundle.y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [bundle.class_names[int(i)] for i in counts.index]
    sns.barplot(x=labels, y=counts.values, ax=ax, color="#4c78a8")
    ax.set(title=f"{bundle.display_name}: Class Distribution", xlabel="Class", ylabel="Samples")
    for i, value in enumerate(counts.values):
        ax.text(i, value, str(value), ha="center", va="bottom", fontsize=9)
    save(fig, out / f"{bundle.key}_class_distribution", cfg)

    if bundle.image_shape:
        fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
        for class_id, ax in enumerate(axes.flat):
            sample = bundle.X[np.flatnonzero(bundle.y == class_id)[0]].reshape(bundle.image_shape)
            ax.imshow(sample, cmap="gray")
            ax.set_title(f"Digit {class_id}")
            ax.axis("off")
        fig.suptitle("USPS: One Sample per Class")
        save(fig, out / "usps_samples_all_classes", cfg)
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(bundle.X.shape[1]) + 1
        for class_id, class_name in enumerate(bundle.class_names):
            part = bundle.X[bundle.y == class_id]
            mean = part.mean(axis=0)
            std = part.std(axis=0)
            ax.plot(x, mean, label=class_name)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)
        ax.set(
            title="UCI Sonar: Mean Feature Profile by Class",
            xlabel="Frequency-band Feature",
            ylabel="Normalized Energy",
        )
        ax.legend()
        save(fig, out / "sonar_class_feature_profiles", cfg)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(np.corrcoef(bundle.X, rowvar=False), cmap="coolwarm", center=0, ax=ax)
        ax.set(title="UCI Sonar: Feature Correlation Matrix", xlabel="Feature", ylabel="Feature")
        save(fig, out / "sonar_feature_correlation", cfg)


def plot_sweep_figures(summary: pd.DataFrame, out: Path, cfg: ExperimentConfig) -> None:
    for dataset in summary["dataset"].unique():
        ds = summary[summary["dataset"] == dataset]
        ds_slug = slug(dataset)
        lda = ds[ds["method"] == "Fisher LDA"].sort_values("train_ratio")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
        for ax, metric in zip(axes, ("OA", "AA", "Kappa")):
            ax.errorbar(
                lda["train_ratio"] * 100,
                lda[f"{metric}_mean"],
                yerr=lda[f"{metric}_std"].fillna(0),
                marker="o",
                capsize=3,
                color=COLORS["Fisher LDA"],
            )
            ax.set(title=metric, xlabel="Training Samples (%)", ylabel=metric)
            ax.set_ylim(-0.05 if metric == "Kappa" else 0, 1.02)
        fig.suptitle(f"{dataset}: Fisher LDA vs Training Ratio")
        save(fig, out / f"{ds_slug}_fisher_metrics_vs_train_ratio", cfg)

        for method in ("KNN", "Fisher+KNN"):
            part = ds[ds["method"] == method]
            for metric in ("OA", "Error Rate"):
                fig, ax = plt.subplots(figsize=(9, 5.5))
                for ratio in sorted(part["train_ratio"].unique()):
                    line = part[part["train_ratio"] == ratio].sort_values("k")
                    ax.plot(line["k"], line[f"{metric}_mean"], marker="o", label=f"{ratio:.0%}")
                ax.set(
                    title=f"{dataset}: {method} {metric} vs K",
                    xlabel="K",
                    ylabel=metric,
                    xticks=list(cfg.k_values),
                )
                ax.set_ylim(0, 1.02)
                ax.legend(title="Train ratio", ncol=3, fontsize=8)
                save(fig, out / f"{ds_slug}_{slug(method)}_{slug(metric)}_vs_k", cfg)

            pivot = part.pivot(index="train_ratio", columns="k", values="OA_mean")
            fig, ax = plt.subplots(figsize=(9, 5.5))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
            ax.set(title=f"{dataset}: {method} Mean OA", xlabel="K", ylabel="Training Ratio")
            ax.set_yticklabels([f"{float(t.get_text()):.0%}" for t in ax.get_yticklabels()], rotation=0)
            save(fig, out / f"{ds_slug}_{slug(method)}_oa_heatmap", cfg)


def plot_reference_metrics(
    bundle: DatasetBundle, results: list[ReferenceResult], out: Path, cfg: ExperimentConfig
) -> None:
    metrics_df = pd.DataFrame([{"method": r.method, **r.metrics} for r in results])
    long = metrics_df.melt(id_vars="method", value_vars=["OA", "AA", "Kappa"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=long, x="variable", y="value", hue="method", palette=COLORS, ax=ax)
    ax.set(title=f"{bundle.display_name}: Reference Test Metrics", xlabel="Metric", ylabel="Score", ylim=(0, 1.02))
    ax.legend(title="Method")
    save(fig, out / f"{bundle.key}_reference_metric_comparison", cfg)

    for result in results:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            result.confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=bundle.class_names,
            yticklabels=bundle.class_names,
            ax=ax,
        )
        ax.set(
            title=f"{bundle.display_name}: {result.method} Confusion Matrix",
            xlabel="Predicted",
            ylabel="True",
        )
        save(fig, out / f"{bundle.key}_{slug(result.method)}_confusion_matrix", cfg)

        if bundle.image_shape:
            _plot_usps_predictions(bundle, result, out, cfg)
        else:
            _plot_sonar_errors(bundle, result, out, cfg)

    first = results[0]
    if bundle.key == "usps":
        coords = first.fisher_test
        fig, ax = plt.subplots(figsize=(9, 7))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=first.y_true, cmap="tab10", s=9, alpha=0.65)
        ax.set(title="USPS: Fisher Projection of Reference Test Set", xlabel="Discriminant 1", ylabel="Discriminant 2")
        fig.colorbar(scatter, ax=ax, ticks=range(10), label="Digit")
        save(fig, out / "usps_fisher_projection", cfg)
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        coords = first.fisher_test[:, 0]
        for class_id, name in enumerate(bundle.class_names):
            sns.kdeplot(coords[first.y_true == class_id], fill=True, alpha=0.3, label=name, ax=ax)
        ax.set(title="UCI Sonar: Fisher Discriminant Distribution", xlabel="Discriminant 1", ylabel="Density")
        ax.legend()
        save(fig, out / "sonar_fisher_projection", cfg)


def _plot_usps_predictions(
    bundle: DatasetBundle, result: ReferenceResult, out: Path, cfg: ExperimentConfig
) -> None:
    correct = np.flatnonzero(result.y_true == result.y_pred)[:10]
    wrong = np.flatnonzero(result.y_true != result.y_pred)[:10]
    chosen = np.concatenate([correct, wrong])
    if len(chosen) == 0:
        return
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, pos in zip(axes.flat, chosen):
        ax.imshow(result.X_test_raw[pos].reshape(bundle.image_shape), cmap="gray")
        ok = result.y_true[pos] == result.y_pred[pos]
        ax.set_title(f"T:{result.y_true[pos]} P:{result.y_pred[pos]}", color="#188038" if ok else "#c5221f", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"USPS: {result.method} Correct (top) and Misclassified (bottom)")
    save(fig, out / f"usps_{slug(result.method)}_qualitative_predictions", cfg)


def _plot_sonar_errors(
    bundle: DatasetBundle, result: ReferenceResult, out: Path, cfg: ExperimentConfig
) -> None:
    wrong = np.flatnonzero(result.y_true != result.y_pred)[:12]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(bundle.X.shape[1]) + 1
    if len(wrong) == 0:
        ax.text(0.5, 0.5, "No misclassified samples", ha="center", va="center", transform=ax.transAxes)
    else:
        for pos in wrong:
            ax.plot(
                x,
                result.X_test_raw[pos],
                alpha=0.65,
                label=f"#{result.test_indices[pos]} {bundle.class_names[result.y_true[pos]]}->{bundle.class_names[result.y_pred[pos]]}",
            )
        ax.legend(fontsize=7, ncol=2)
    ax.set(
        title=f"UCI Sonar: {result.method} Misclassified Profiles",
        xlabel="Frequency-band Feature",
        ylabel="Normalized Energy",
    )
    save(fig, out / f"sonar_{slug(result.method)}_misclassified_profiles", cfg)

