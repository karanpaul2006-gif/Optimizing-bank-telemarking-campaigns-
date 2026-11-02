# Bank Telemarketing Deposit Prediction using Logistic Regression

## Overview
This repository implements an iterative Logistic Regression binary classification model in R to predict client subscription to a term deposit (`y = yes/no`) based on demographic attributes, direct marketing details, economic indicator ratios, and historical campaign outcomes.

## Repository Structure
- `main.R`: Complete end-to-end R execution script (preprocessing, encoding, iterative GLM models, evaluation, ROC analysis).
- `bank-additional-full.csv`: Dataset source file (Bank Marketing Dataset from UCI repository).
- `bank_cor.csv`: Auto-generated correlation matrix output file.

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
