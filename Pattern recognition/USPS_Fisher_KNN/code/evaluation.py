from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import ExperimentConfig
from datasets import DatasetBundle


@dataclass
class ReferenceResult:
    dataset_key: str
    method: str
    selected_k: int | None
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    test_indices: np.ndarray
    X_test_raw: np.ndarray
    confusion: np.ndarray
    report: pd.DataFrame
    fisher_train: np.ndarray | None
    fisher_test: np.ndarray | None
    y_train: np.ndarray


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    oa = accuracy_score(y_true, y_pred)
    return {
        "OA": float(oa),
        "AA": float(balanced_accuracy_score(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Error Rate": float(1.0 - oa),
        "Macro Precision": float(precision),
        "Macro Recall": float(recall),
        "Macro F1": float(f1),
    }


def _prepare_split(
    bundle: DatasetBundle, train_ratio: float, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(bundle.y))
    train_idx, test_idx = train_test_split(
        idx,
        train_size=train_ratio,
        random_state=random_state,
        stratify=bundle.y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(bundle.X[train_idx])
    X_test = scaler.transform(bundle.X[test_idx])
    return X_train, X_test, bundle.y[train_idx], bundle.y[test_idx], train_idx, test_idx


def run_ratio_k_sweep(bundle: DatasetBundle, cfg: ExperimentConfig) -> pd.DataFrame:
    records: list[dict] = []
    for train_ratio in cfg.train_ratios:
        for repeat in range(cfg.repeats):
            seed = cfg.random_seed + repeat
            X_train, X_test, y_train, y_test, _, _ = _prepare_split(
                bundle, train_ratio, seed
            )
            fisher = LinearDiscriminantAnalysis(solver="svd")
            fisher.fit(X_train, y_train)
            fisher_pred = fisher.predict(X_test)
            records.append(
                {
                    "dataset": bundle.display_name,
                    "method": "Fisher LDA",
                    "train_ratio": train_ratio,
                    "repeat": repeat,
                    "seed": seed,
                    "k": np.nan,
                    **compute_metrics(y_test, fisher_pred),
                }
            )

            X_train_fisher = fisher.transform(X_train)
            X_test_fisher = fisher.transform(X_test)
            for k in cfg.k_values:
                if k > len(y_train):
                    continue
                raw_knn = KNeighborsClassifier(n_neighbors=k, weights="uniform", n_jobs=cfg.n_jobs)
                raw_knn.fit(X_train, y_train)
                raw_pred = raw_knn.predict(X_test)
                records.append(
                    {
                        "dataset": bundle.display_name,
                        "method": "KNN",
                        "train_ratio": train_ratio,
                        "repeat": repeat,
                        "seed": seed,
                        "k": k,
                        **compute_metrics(y_test, raw_pred),
                    }
                )

                fisher_knn = KNeighborsClassifier(
                    n_neighbors=k, weights="uniform", n_jobs=cfg.n_jobs
                )
                fisher_knn.fit(X_train_fisher, y_train)
                fisher_knn_pred = fisher_knn.predict(X_test_fisher)
                records.append(
                    {
                        "dataset": bundle.display_name,
                        "method": "Fisher+KNN",
                        "train_ratio": train_ratio,
                        "repeat": repeat,
                        "seed": seed,
                        "k": k,
                        **compute_metrics(y_test, fisher_knn_pred),
                    }
                )
    return pd.DataFrame(records)


def summarize_sweep(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["OA", "AA", "Kappa", "Error Rate", "Macro Precision", "Macro Recall", "Macro F1"]
    grouped = raw.groupby(["dataset", "method", "train_ratio", "k"], dropna=False)[metrics]
    summary = grouped.agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([str(v) for v in col if str(v)]) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    return summary


def _select_k(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    k_values: tuple[int, ...],
    seed: int,
    validation_ratio: float,
    n_jobs: int,
) -> tuple[dict[str, int], pd.DataFrame]:
    fit_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=validation_ratio,
        random_state=seed,
        stratify=y_train,
    )
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_train_raw[fit_idx])
    X_val = scaler.transform(X_train_raw[val_idx])
    y_fit, y_val = y_train[fit_idx], y_train[val_idx]
    fisher = LinearDiscriminantAnalysis(solver="svd").fit(X_fit, y_fit)
    X_fit_f = fisher.transform(X_fit)
    X_val_f = fisher.transform(X_val)

    rows = []
    for method, A, B in (("KNN", X_fit, X_val), ("Fisher+KNN", X_fit_f, X_val_f)):
        for k in k_values:
            if k > len(y_fit):
                continue
            model = KNeighborsClassifier(n_neighbors=k, weights="uniform", n_jobs=n_jobs)
            model.fit(A, y_fit)
            pred = model.predict(B)
            rows.append({"method": method, "k": k, **compute_metrics(y_val, pred)})
    table = pd.DataFrame(rows)
    selected = {}
    for method in ("KNN", "Fisher+KNN"):
        part = table[table["method"] == method].sort_values(
            ["OA", "AA", "k"], ascending=[False, False, True]
        )
        selected[method] = int(part.iloc[0]["k"])
    table["selected"] = table.apply(lambda r: int(r["k"]) == selected[r["method"]], axis=1)
    return selected, table


def run_reference_evaluation(
    bundle: DatasetBundle, cfg: ExperimentConfig
) -> tuple[list[ReferenceResult], pd.DataFrame]:
    idx = np.arange(len(bundle.y))
    train_idx, test_idx = train_test_split(
        idx,
        train_size=cfg.reference_train_ratio,
        random_state=cfg.random_seed,
        stratify=bundle.y,
    )
    selected, selection_table = _select_k(
        bundle.X[train_idx],
        bundle.y[train_idx],
        cfg.k_values,
        cfg.random_seed,
        cfg.validation_ratio_within_train,
        cfg.n_jobs,
    )
    selection_table.insert(0, "dataset", bundle.display_name)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(bundle.X[train_idx])
    X_test = scaler.transform(bundle.X[test_idx])
    y_train, y_test = bundle.y[train_idx], bundle.y[test_idx]
    fisher = LinearDiscriminantAnalysis(solver="svd").fit(X_train, y_train)
    fisher_train = fisher.transform(X_train)
    fisher_test = fisher.transform(X_test)

    model_specs = [
        ("Fisher LDA", None, fisher.predict(X_test)),
    ]
    raw_knn = KNeighborsClassifier(
        n_neighbors=selected["KNN"], weights="uniform", n_jobs=cfg.n_jobs
    ).fit(X_train, y_train)
    model_specs.append(("KNN", selected["KNN"], raw_knn.predict(X_test)))
    f_knn = KNeighborsClassifier(
        n_neighbors=selected["Fisher+KNN"], weights="uniform", n_jobs=cfg.n_jobs
    ).fit(fisher_train, y_train)
    model_specs.append(
        ("Fisher+KNN", selected["Fisher+KNN"], f_knn.predict(fisher_test))
    )

    results = []
    labels = list(range(len(bundle.class_names)))
    target_names = bundle.class_names
    for method, k, pred in model_specs:
        report = pd.DataFrame(
            classification_report(
                y_test,
                pred,
                labels=labels,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            )
        ).T.reset_index(names="class")
        results.append(
            ReferenceResult(
                dataset_key=bundle.key,
                method=method,
                selected_k=k,
                metrics=compute_metrics(y_test, pred),
                y_true=y_test,
                y_pred=pred,
                test_indices=test_idx,
                X_test_raw=bundle.X[test_idx],
                confusion=confusion_matrix(y_test, pred, labels=labels),
                report=report,
                fisher_train=fisher_train,
                fisher_test=fisher_test,
                y_train=y_train,
            )
        )
    return results, selection_table

