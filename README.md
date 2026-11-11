# UK SME Strategic Growth Prediction

**Dissertation project — MSc Business Analytics, Aston University**

> *Business Analytics and Decision-Making: A Study of How UK Businesses Use Data in Strategic Planning*
> Varshitha Konakala · Submitted April 2026

## About this repository

This repository recreates, in code, the methodology of my MSc dissertation. The original dissertation used a secondary dataset of **248 UK SMEs (January 2018 – February 2023)** obtained via Kaggle, with **14 derived financial indicators**.

**Important — data honesty:**
- The original Kaggle dataset is **not** redistributed in this repository.
- The dissertation PDF is **not** included in this repository.
- The dataset in `data/` is a **clearly labelled synthetic recreation** (248 SMEs, fixed seed 42), shaped like the original study so the methodology can be run and inspected end-to-end.

## Dissertation aim and objectives

- **Aim:** To examine how UK businesses use business analytics and data in strategic planning, and which financial indicators best predict SME growth.
- **Objectives:** (1) Explore how SMEs use data in strategic decision-making; (2) identify the financial indicators most associated with growth; (3) compare regression and machine-learning models for predicting high growth; (4) explain model predictions so they can inform strategic planning.

## Methods (dissertation and this recreation)

- Descriptive analytics and correlation analysis
- Linear regression (continuous turnover growth) and logistic regression (high/low growth)
- Random Forest, Gradient Boosting and **XGBoost** model comparison
- SHAP / feature-importance analysis for explainability (dissertation); feature importance is reported in this recreation
- Target: high vs low growth, split by median turnover growth (balanced: 124 high / 124 low in the synthetic recreation)

## Dissertation findings (from the original study — attributed)

These are the findings reported in the dissertation abstract, **not** results from the synthetic recreation below:

- **XGBoost was the best-performing model** in the original model comparison.
- **Profit per employee and profit margin were the strongest predictors** of growth (SHAP analysis).
- **Regression explained more than 70% of the variance** in the outcome.

## Synthetic recreation — actual run results

Run on the synthetic dataset in this repo (seed 42, 25% holdout). Full output: `results/results_summary.txt` and `results/model_comparison.csv`.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 0.7419 | 0.7419 | 0.7419 | 0.7419 |
| Logistic Regression | 0.7097 | 0.7241 | 0.6774 | 0.7000 |
| Random Forest | 0.6935 | 0.7308 | 0.6129 | 0.6667 |
| Gradient Boosting | 0.6774 | 0.6897 | 0.6452 | 0.6667 |

- Best model in the recreation, by F1: **XGBoost (F1 = 0.7419, accuracy = 0.7419)**
- Linear regression R² on continuous turnover growth (test): **0.3579** — note this is the synthetic recreation; it does **not** reproduce the dissertation's >70% figure, which came from the original Kaggle data.
- Strongest correlations with turnover growth in the synthetic data: profit margin (0.6231), cost ratio (−0.6231), profit per employee (0.4629), average profit (0.4409).
- Top XGBoost features in the recreation: cost ratio (0.297), profit margin (0.128), average profit (0.072), profit per employee (0.068).

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · SciPy · SQL

## How to run

```bash
pip install -r requirements.txt
python3 analysis.py
```

Outputs are written to `data/synthetic_sme_data.csv`, `results/results_summary.txt` and `results/model_comparison.csv`.

## Repository structure

```
├── analysis.py
├── data/            # generated synthetic dataset (labelled synthetic)
├── sql/analysis.sql
├── results/         # results_summary.txt, model_comparison.csv
├── requirements.txt
└── LICENSE          # MIT
```

## License

MIT — see `LICENSE`. Copyright (c) 2026 Varshitha Konakala.
