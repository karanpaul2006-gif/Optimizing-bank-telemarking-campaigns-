# Bank Telemarketing Campaign Analysis & Deposit Response Prediction

**An R-based predictive analytics project using logistic regression to identify potential term-deposit subscribers and optimize telemarketing campaign targeting.**

![R](https://img.shields.io/badge/Language-R-blue)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-orange)
![Analytics](https://img.shields.io/badge/Analytics-Predictive%20Modelling-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 1. Project Overview

Bank telemarketing campaigns involve contacting customers to encourage them to subscribe to financial products such as term deposits.

However, contacting every customer is resource-intensive. Identifying customers who are more likely to subscribe can help improve campaign efficiency and prioritize sales efforts.

This project develops an iterative binary classification workflow using R and Logistic Regression to predict whether a customer will subscribe to a bank term deposit.

The analysis uses customer demographics, campaign information, previous campaign outcomes and economic indicators to build and evaluate predictive models.

Beyond predicting subscription probability, the project explores probability-threshold optimization to identify a targeting strategy that maintains a minimum recall requirement while reducing unnecessary customer contacts.

## 2. Business Problem

A bank wants to improve the efficiency of its term-deposit telemarketing campaigns.

Not every contacted customer subscribes to the product, and contacting customers who are unlikely to respond can consume valuable campaign resources.

The central business question is:

**Can historical customer and campaign information help identify potential term-deposit subscribers while reducing unnecessary telemarketing contacts?**

The project investigates:

- Which customer and campaign characteristics are associated with subscription outcomes?
- How does progressively introducing predictors affect logistic regression models?
- Does multicollinearity affect the selected model variables?
- How effectively can the model distinguish between potential subscribers and non-subscribers?
- What probability threshold provides at least 80% recall while minimizing the number of selected contacts?

## 3. Dataset

**Dataset:** `bank-additional-full.csv`

The dataset contains historical bank direct-marketing campaign information.

The target variable is:

`y`

| Target Value | Description |
|---|---|
| yes | Customer subscribed to the term deposit |
| no | Customer did not subscribe |

The dataset contains several categories of predictors.

### Customer Information

- Age
- Job profile
- Marital status
- Education
- Housing loan status
- Personal loan status
- Default status

### Campaign Information

- Contact type
- Number of contacts during the current campaign
- Days since previous contact
- Previous campaign outcome

### Economic Indicators

- Euro interbank offered rate
- Consumer price index
- Consumer confidence index

These variables provide information about customer characteristics, previous marketing interactions and broader economic conditions.

## 4. Technology Stack

| Technology | Application |
|---|---|
| R | Statistical analysis and modelling |
| Logistic Regression | Binary classification |
| GLM | Iterative model development |
| fastDummies | Categorical variable encoding |
| caTools | Train-validation-test splitting |
| car | Multicollinearity analysis using VIF |
| corrplot | Correlation visualization |
| ROCR | ROC curve and AUC evaluation |

## 5. Analytical Workflow

### Step 1: Data Import and Initial Exploration

The dataset is imported into R using `read.csv()`.

Initial descriptive summaries are generated using `summary()` to understand variable distributions and identify the structure of the dataset.

### Step 2: Data Preprocessing and Encoding

The target variable is converted into binary format:

- Yes = 1
- No = 0

Binary categorical variables such as housing loans, personal loans and default status are also encoded.

Multi-category variables are transformed using one-hot encoding through the `fastDummies` package.

The encoded dataset is then used for correlation analysis and model development.

### Step 3: Correlation Analysis

A correlation matrix is generated using the numeric predictors.

The `corrplot` package is used to visualize relationships between numerical variables.

This provides an initial understanding of predictor relationships before model development.

### Step 4: Train, Validation and Test Split

The dataset is divided into three subsets:

| Dataset | Allocation | Purpose |
|---|---:|---|
| Training | 60% | Model development |
| Validation | 20% | Threshold selection |
| Testing | 20% | Final model evaluation |

The random split uses a fixed seed of 111 for reproducibility.

The validation dataset is used for probability-threshold optimization, while the test dataset is reserved for final evaluation.

### Step 5: Iterative Logistic Regression Modelling

Fourteen progressively specified logistic regression models are developed using R's `glm()` function with a binomial family.

The models begin with economic indicators and progressively introduce additional predictors.

The modelling process explores:

- Economic indicators.
- Campaign contact frequency.
- Previous campaign outcomes.
- Customer age.
- Job profile.
- Customer confidence indicators.
- Days since previous contact.
- Housing and personal loan status.
- Education.
- Contact type.

The purpose is to examine how progressively introducing predictors changes the model specification and its statistical characteristics.

### Step 6: Multicollinearity Analysis

Variance Inflation Factor (VIF) is calculated using the `car` package.

VIF analysis is performed during iterative model development to examine potential multicollinearity among predictors.

This helps identify predictors that may contain overlapping information.

### Step 7: Probability Threshold Optimization

A classification model produces probabilities rather than directly assigning final class labels.

The default 0.50 classification threshold is not necessarily optimal for every business problem.

In this project, threshold selection is based on a business-oriented constraint:

**Maintain at least 80% recall while maximizing the percentage of unnecessary contacts avoided.**

The validation dataset is evaluated across probability thresholds ranging from 0.01 to 0.99 in increments of 0.01.

The following metrics are calculated for each threshold:

- Accuracy
- Precision
- Recall
- Specificity
- F1 Score
- Percentage of contacts selected
- Percentage of contacts avoided
- Percentage of actual subscribers captured

Only thresholds achieving at least 80% validation recall are considered eligible.

Among these eligible thresholds, the threshold with the highest percentage of contacts avoided is selected.

This selected threshold is then applied to the holdout test dataset.

### Step 8: Final Model Evaluation

The final model is evaluated on the previously reserved test dataset.

The evaluation includes:

**Confusion Matrix**

Used to compare actual subscription outcomes against predicted classifications.

**Precision**

Measures the proportion of predicted subscribers who actually subscribed.

**Recall**

Measures the proportion of actual subscribers correctly identified.

**Specificity**

Measures the proportion of non-subscribers correctly classified.

**F1 Score**

Provides a combined measure of precision and recall.

**ROC-AUC**

Evaluates the model's ability to distinguish between subscribers and non-subscribers across classification thresholds.

The ROCR package is used to calculate the ROC-AUC and generate the ROC curve.

## 6. Business Interpretation

The project focuses on improving campaign targeting rather than maximizing classification accuracy alone.

A bank may prefer a model that identifies a larger proportion of actual potential subscribers, even if it results in some additional unsuccessful contacts.

The validation threshold-selection strategy therefore prioritizes maintaining at least 80% recall while reducing unnecessary contacts.

This provides a more business-oriented approach to classification threshold selection.

However, the selected threshold is based on a recall constraint rather than a complete financial cost-benefit model.

Actual campaign profitability would additionally require information about:

- Cost per customer contact.
- Conversion value of a successful subscription.
- Operational campaign expenses.
- Customer acquisition costs.

These factors are not directly quantified in the current analysis.

## 7. Results and Evaluation

The R script calculates and exports the final test-set performance metrics.

The following measures are included:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Reliability of predicted positive responses |
| Recall | Percentage of actual subscribers identified |
| Specificity | Correct identification of non-subscribers |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Overall discrimination capability |

The final numerical values should be read from the generated `test_metrics.csv` file after executing the script.

The selected probability threshold is determined using validation data and evaluated on the separate test dataset.

**Note:** Numerical model performance results are intentionally not hardcoded into this README. They should be updated from the latest successful execution of the R script.

## 8. Generated Outputs

After executing the script, the following files are generated:

### Root Directory

`bank_cor.csv`

Contains the correlation matrix of numerical variables.

### Results Directory

`results/validation_threshold_tradeoff.csv`

Contains the performance metrics calculated across the evaluated validation thresholds.

`results/test_metrics.csv`

Contains the final test-set evaluation metrics, including ROC-AUC.

`results/confusion_matrix.csv`

Contains the final holdout confusion matrix.

The ROC curve is displayed in the R plotting window during execution.

It is not automatically exported as an image by the current script.

## 9. Repository Structure

```text
Optimizing-bank-telemarketing-campaigns/
│
├── README.md
│
├── main.R
│
├── bank-additional-full.csv
│
├── bank_cor.csv
│
└── results/
    ├── validation_threshold_tradeoff.csv
    ├── test_metrics.csv
    └── confusion_matrix.csv
```

The generated CSV files appear after running the script. Only files actually committed to the repository should be listed in the published repository structure.

## 10. How to Run the Project

### Requirements

- R
- RStudio or another R environment

### Required Libraries

The script uses:

```r
required_packages <- c(
  "fastDummies",
  "corrplot",
  "caTools",
  "car",
  "ROCR"
)
```

The script automatically checks for and installs missing packages.

### Execution Instructions

1. Clone or download the repository.
2. Open the project directory in RStudio.
3. Ensure `bank-additional-full.csv` is available in the working directory.
4. Open `main.R`.
5. Execute the complete script sequentially.
6. Review the correlation plots and ROC curve.
7. Inspect the generated CSV files in the `results` directory.

## 11. Limitations

The current project has several limitations:

- Logistic regression is the only classification algorithm implemented.
- The model has not been compared against alternative algorithms such as Random Forest or Gradient Boosting.
- Cross-validation and hyperparameter tuning are not implemented.
- Threshold optimization is based on a minimum recall requirement rather than a complete financial cost-benefit framework.
- The model is not deployed as a production application.
- Real-world campaign effectiveness has not been measured.
- The analysis does not include live monitoring or model retraining.

These limitations provide opportunities for future development.

## 12. Future Improvements

Potential extensions include:

- Comparing Logistic Regression with Random Forest and XGBoost.
- Implementing cross-validation.
- Conducting feature selection and additional statistical diagnostics.
- Evaluating model calibration.
- Performing detailed error analysis.
- Incorporating campaign costs into threshold optimization.
- Developing an interactive prediction application using Shiny.
- Exploring model explainability and variable importance.

## 13. Skills Demonstrated

**Programming:** R, data manipulation and statistical computing.

**Statistics:** Logistic regression, correlation analysis and multicollinearity diagnostics.

**Machine Learning:** Binary classification, probability prediction and classification threshold optimization.

**Model Evaluation:** Confusion matrices, precision, recall, specificity, F1 Score and ROC-AUC.

**Business Analytics:** Campaign-response analysis, customer targeting and decision-support modelling.

---

**Author:** Karan Paul Arora

**GitHub:** https://github.com/karanpaul2006-gif
