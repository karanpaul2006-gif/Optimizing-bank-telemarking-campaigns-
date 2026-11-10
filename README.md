# Bank Telemarketing Deposit Prediction using Logistic Regression

## Overview
This repository implements an iterative Logistic Regression binary classification model in R to predict client subscription to a term deposit (`y = yes/no`) based on demographic attributes, direct marketing details, economic indicator ratios, and historical campaign outcomes.


## Required R Libraries
- `fastDummies`
- `corrplot`
- `caTools`
- `car`
- `ROCR`

## Model Iteration Summary
The model is incrementally built from single baseline macroeconomic features up to a 14-variable Generalized Linear Model (GLM). Multi-collinearity is checked at each iteration using Variance Inflation Factor (`vif`).

## Execution Instructions
1. Place `bank-additional-full.csv` in your working directory.
2. Run `main.R` in RStudio, VS Code, or GitHub Web Editor.

Optimizing-bank-telemarketing-campaigns/
│
├── data/
│   └── README.md
│       Dataset source and download instructions
│
├── results/
│   ├── test_metrics.csv
│   ├── confusion_matrix.csv
│   ├── validation_threshold_tradeoff.csv
│   ├── roc_curve.png
│   └── model_coefficients.csv
│
├── main.R
│
├── README.md
│
└── LICENSE
