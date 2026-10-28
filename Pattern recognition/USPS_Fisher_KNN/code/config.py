from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    random_seed: int = 42
    train_ratios: tuple[float, ...] = tuple(i / 10 for i in range(1, 10))
    k_values: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 15)
    repeats: int = 5
    reference_train_ratio: float = 0.70
    validation_ratio_within_train: float = 0.20
    n_jobs: int = -1
    dpi: int = 180
    figure_format: str = "png"
    datasets: tuple[str, ...] = ("usps", "sonar")
    methods: tuple[str, ...] = ("Fisher LDA", "KNN", "Fisher+KNN")

    def to_dict(self) -> dict:
        return asdict(self)


CONFIG = ExperimentConfig()

