from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo


@dataclass
class DatasetBundle:
    key: str
    display_name: str
    X: np.ndarray
    y: np.ndarray
    class_names: list[str]
    sample_ids: np.ndarray
    image_shape: tuple[int, int] | None
    feature_names: list[str]


def _normalize_usps_labels(y_raw: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(y_raw), errors="raise").astype(int).to_numpy()
    unique = set(np.unique(values).tolist())
    if unique == set(range(1, 11)):
        values = np.where(values == 10, 0, values)
    if set(np.unique(values).tolist()) != set(range(10)):
        raise ValueError(f"Unexpected USPS labels: {sorted(np.unique(values).tolist())}")
    return values.astype(int)


def load_usps(data_dir: Path) -> DatasetBundle:
    cache = data_dir / "cache" / "openml"
    cache.mkdir(parents=True, exist_ok=True)
    X, y = fetch_openml(
        data_id=41082,
        as_frame=False,
        return_X_y=True,
        data_home=cache,
        parser="auto",
    )
    X = np.asarray(X, dtype=np.float64)
    y = _normalize_usps_labels(np.asarray(y))
    if X.shape != (9298, 256):
        raise ValueError(f"Unexpected USPS shape: {X.shape}; expected (9298, 256).")
    if not np.isfinite(X).all():
        raise ValueError("USPS contains non-finite values.")
    return DatasetBundle(
        key="usps",
        display_name="USPS",
        X=X,
        y=y,
        class_names=[str(i) for i in range(10)],
        sample_ids=np.arange(len(y)),
        image_shape=(16, 16),
        feature_names=[f"pixel_{i:03d}" for i in range(X.shape[1])],
    )


def load_sonar(data_dir: Path) -> DatasetBundle:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_csv = raw_dir / "uci_sonar_id151.csv"
    if local_csv.exists():
        frame = pd.read_csv(local_csv)
        targets = frame.pop("target").astype(str).str.strip()
        features = frame
    else:
        dataset = fetch_ucirepo(id=151)
        features = dataset.data.features.copy()
        targets = dataset.data.targets.iloc[:, 0].astype(str).str.strip()
        saved = features.copy()
        saved["target"] = targets.to_numpy()
        saved.to_csv(local_csv, index=False)
    X = features.to_numpy(dtype=np.float64)
    mapping = {"R": 0, "M": 1}
    if not set(targets.unique()).issubset(mapping):
        raise ValueError(f"Unexpected Sonar labels: {sorted(targets.unique().tolist())}")
    y = targets.map(mapping).to_numpy(dtype=int)
    if X.shape != (208, 60):
        raise ValueError(f"Unexpected Sonar shape: {X.shape}; expected (208, 60).")
    if not np.isfinite(X).all():
        raise ValueError("Sonar contains non-finite values.")
    return DatasetBundle(
        key="sonar",
        display_name="UCI Sonar",
        X=X,
        y=y,
        class_names=["Rock", "Mine"],
        sample_ids=np.arange(len(y)),
        image_shape=None,
        feature_names=[str(c) for c in features.columns],
    )


def load_all(data_dir: Path, dataset_keys: tuple[str, ...]) -> dict[str, DatasetBundle]:
    loaders = {"usps": load_usps, "sonar": load_sonar}
    unknown = sorted(set(dataset_keys) - set(loaders))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return {key: loaders[key](data_dir) for key in dataset_keys}


def dataset_summary(bundle: DatasetBundle) -> pd.DataFrame:
    counts = pd.Series(bundle.y).value_counts().sort_index()
    rows = []
    for class_id, count in counts.items():
        rows.append(
            {
                "dataset": bundle.display_name,
                "n_samples": len(bundle.y),
                "n_features": bundle.X.shape[1],
                "n_classes": len(counts),
                "image_shape": "16x16" if bundle.image_shape else "not_image",
                "class_id": int(class_id),
                "class_name": bundle.class_names[int(class_id)],
                "class_count": int(count),
                "class_fraction": float(count / len(bundle.y)),
            }
        )
    return pd.DataFrame(rows)

