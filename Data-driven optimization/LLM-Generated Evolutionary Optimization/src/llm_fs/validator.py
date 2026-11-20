from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


ALLOWED_IMPORT_ROOTS = {"numpy", "scipy", "sklearn", "math", "random", "collections", "dataclasses"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__"}


def static_validate(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    errors: list[str] = []
    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    if "GeneratedOptimizer" not in class_names:
        errors.append("Missing class GeneratedOptimizer.")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                    errors.append(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                errors.append(f"Forbidden import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                errors.append(f"Forbidden call: {node.func.id}")
    return sorted(set(errors))


def load_generated_optimizer(path: Path) -> type:
    errors = static_validate(path)
    if errors:
        raise ValueError("Static validation failed: " + "; ".join(errors))
    spec = importlib.util.spec_from_file_location(f"generated_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GeneratedOptimizer


def dynamic_validate(path: Path, dimensions: tuple[int, ...] = (30, 64)) -> list[str]:
    errors = static_validate(path)
    if errors:
        return errors
    try:
        optimizer_class = load_generated_optimizer(path)
        for dim in dimensions:
            optimizer = optimizer_class([(0, 1)] * dim, ["binary"] * dim, 20, 123)
            X = np.asarray(optimizer.ask(4), dtype=float)
            if X.ndim != 2 or X.shape[1] != dim or not 1 <= X.shape[0] <= 4:
                errors.append(f"ask() returned invalid shape {X.shape} for dimension {dim}.")
                continue
            if not np.all(np.isfinite(X)) or not np.all(np.isin(X, [0, 1])):
                errors.append(f"ask() returned non-binary values for dimension {dim}.")
            if np.unique(X, axis=0).shape[0] != X.shape[0]:
                errors.append(f"ask() returned duplicates within one batch for dimension {dim}.")
            y = np.linspace(0.4, 0.1, X.shape[0])
            optimizer.tell(X, y)
            best_x, best_y = optimizer.get_best()
            if np.asarray(best_x).reshape(-1).size != dim or not np.isfinite(float(best_y)):
                errors.append(f"get_best() returned invalid data for dimension {dim}.")
            X2 = np.asarray(optimizer.ask(3), dtype=float)
            if X2.ndim != 2 or X2.shape[1] != dim:
                errors.append(f"Second ask() returned invalid shape {X2.shape} for dimension {dim}.")
    except Exception as exc:  # report generated-code failures as data, not harness crashes
        errors.append(f"Dynamic validation exception: {type(exc).__name__}: {exc}")
    return errors


def validation_report(path: Path) -> str:
    errors = dynamic_validate(path)
    if errors:
        return "FAILED\n" + "\n".join(f"- {item}" for item in errors)
    return "PASSED\n- Static safety checks passed.\n- Interface tests passed for 30 and 64 dimensions."

