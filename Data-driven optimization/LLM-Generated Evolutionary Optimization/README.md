# LLM-Assisted Evolutionary Optimization

## Overview

This project explores the use of Large Language Models (LLMs) to automatically generate evolutionary optimization algorithms for expensive binary black-box optimization.

The generated algorithm integrates surrogate modeling, uncertainty-guided search, and adaptive evolutionary operations to improve optimization performance under limited evaluation budgets.

## Method

The generated optimizer incorporates:

- **Surrogate Modeling:** Extra Trees regression for objective prediction.
- **Uncertainty-Guided Search:** Balances exploration and exploitation.
- **Evolutionary Search:** Binary crossover and adaptive mutation.
- **Batch Diversity:** Promotes diversity among candidate solutions.

## Experiments

Experiments are conducted on two feature selection datasets:

- Breast Cancer (30 features)
- Digits (64 features)

Generated is compared with Random Search using 10 independent runs and 60 evaluations per run.

## Project Structure

```text
LLM/
├── src/                    # Experiment framework
├── generated_algorithms/   # LLM-generated algorithms
├── configs/                # Experiment configurations
├── results/                # Experimental results
├── figure/                 # Figures and flowcharts
├── tests/                  # Tests
├── run.py                  # Entry point
└── main.tex                # LaTeX report
```

## Run Experiments

```bash
python run.py experiment --config configs/full.json
```

## Environment

- Python 3.13
- NumPy
- SciPy
- Scikit-learn
- Pandas
- Matplotlib
- Pytest

## Results

Experimental results, convergence curves, and boxplots are provided in the `results/` directory.

The complete experimental report is provided in `main.tex` and its compiled PDF version.
