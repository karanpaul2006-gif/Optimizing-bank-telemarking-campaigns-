# Bank Telemarketing Deposit Response Analysis

A predictive analytics project using R and Logistic Regression to analyse historical bank telemarketing campaigns, investigate customer subscription behaviour and evaluate classification models.

<p align="left">
  <img src="https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white"/>
  <img src="https://img.shields.io/badge/Logistic_Regression-6A5ACD?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GLM-008080?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Statistics-2E8B57?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ROCR-FF8C00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
</p>

---

## 1. Project Overview

Banks frequently conduct telemarketing campaigns to encourage customers to subscribe to term deposits.

However, contacting every customer requires considerable campaign resources, while a large proportion of contacted customers may not subscribe.

The objective of this project is to analyse historical bank marketing data and develop logistic regression models to investigate factors associated with term deposit subscriptions.

The project follows a structured analytical workflow involving data preparation, exploratory correlation analysis, progressive model development, multicollinearity assessment and classification evaluation.

### Business Questions

- Which customer characteristics are associated with term deposit subscriptions?
- How do economic indicators relate to subscription behaviour?
- Does previous campaign success provide useful predictive information?
- How do additional predictors influence logistic regression models?
- How can multicollinearity be identified?
- How does changing the classification threshold affect recall and unnecessary campaign contacts?

---

## 2. Dataset Information

**Dataset:** Bank Marketing (bank-additional-full)  
**Source:** UCI Machine Learning Repository  
**Total Observations:** 41,188  
**Target Variable:** `y`  
**Problem Type:** Binary Classification

The dataset contains information relating to a Portuguese bank's direct marketing campaigns.

It includes customer demographics, banking-related information, campaign interaction details and macroeconomic indicators.

### Target Variable

| Value | Meaning |
|---|---|
| 0 | Customer did not subscribe |
| 1 | Customer subscribed to a term deposit |

### Dataset Source

https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## 3. Data Preprocessing & Feature Engineering

The dataset was prepared for logistic regression modelling through categorical encoding and feature transformation.

### Preprocessing Operations

- Converted the target variable into binary numerical values.
- Encoded binary categorical variables.
- Applied dummy encoding to selected categorical predictors.
- Prepared demographic, campaign and economic variables for model development.
- Examined correlations between numerical variables.

### Categorical Variables

The modelling workflow includes:

- Job Profile
- Marital Status
- Education
- Contact Type
- Previous Campaign Outcome
- Housing Loan
- Personal Loan
- Default Status

The `fastDummies` package was used for categorical encoding.

---

## 4. Exploratory Analysis & Correlation

Correlation analysis was performed to examine relationships between numerical variables.

The `corrplot` package was used to visualize correlations between predictors.

This step helped examine relationships between economic indicators and other numerical features before developing the logistic regression models.

---

## 5. Progressive Logistic Regression Model Development

A major component of this project is the progressive development of **14 Generalized Linear Models (GLMs)** using logistic regression.

Instead of immediately developing one final model, additional predictors were introduced progressively.

### Model Development Approach

| Model | Additional Predictor |
|---|---|
| Model 1 | Euribor 3-month rate |
| Model 2 | Consumer Price Index |
| Model 3 | Campaign contacts |
| Model 4 | Previous campaign success |
| Model 5 | Customer age |
| Model 6 | Blue-collar job profile |
| Model 7 | Consumer confidence index |
| Model 8 | Days since previous contact (`pdays`) |
| Model 9 | Housing loan |
| Model 10 | Personal loan |
| Model 11 | Retired job profile |
| Model 12 | Student job profile |
| Model 13 | University degree |
| Model 14 | Cellular contact type |

### Why Progressive Modelling?

Progressively introducing predictors allows examination of:

- Changes in regression coefficients.
- Statistical significance of individual predictors.
- Residual deviance.
- Akaike Information Criterion (AIC).
- Multicollinearity between explanatory variables.

This provides a structured approach to understanding how different customer and campaign characteristics contribute to the fitted models.

---

## 6. Multicollinearity Analysis (VIF)

Variance Inflation Factor (VIF) was used to examine multicollinearity among predictors.

The `car` package was used for VIF calculations.

### Final Displayed VIF Results

| Predictor | VIF |
|---|---:|
| Euribor 3-month rate | 1.773 |
| Consumer Price Index | 1.882 |
| Campaign | 1.022 |
| Previous campaign outcome: success | 8.993 |
| Age | 1.643 |
| Blue-collar job profile | 1.170 |
| Consumer confidence index | 1.278 |
| `pdays` | 9.199 |
| Housing loan | 1.009 |
| Personal loan | 1.004 |
| Retired job profile | 1.554 |
| Student job profile | 1.153 |
| University degree | 1.160 |
| Cellular contact | 1.454 |

### Key Observation

The highest displayed VIF values were:

- `pdays`: 9.199
- Previous campaign outcome (success): 8.993

These variables show comparatively higher multicollinearity and warrant attention when interpreting the fitted regression coefficients.

---

## 7. Training, Validation & Testing Strategy

The dataset was divided into three subsets.

| Dataset | Allocation |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

**Random Seed:** 111

### Training Set

Used to estimate logistic regression coefficients.

### Validation Set

Used to explore different probability thresholds and evaluate classification performance.

### Testing Set

Reserved for final evaluation using predictions generated by the selected model and threshold.

---

## 8. Probability Threshold Optimization

Logistic regression generates probabilities rather than directly assigning final Yes/No classifications.

A probability threshold determines when a customer is classified as likely to subscribe.

Instead of automatically using the conventional 0.50 threshold, the project explores multiple probability cutoffs.

### Intended Optimization Objective

The intended objective is to:

**Maintain at least 80% recall on validation data while minimizing unnecessary campaign contacts.**

The threshold-selection workflow considers:

- Accuracy
- Precision
- Recall
- Specificity
- F1 Score
- Number of predicted-positive customers
- Percentage of campaign contacts avoided

The validation recall requirement is a threshold-selection objective, not a guarantee of achieving the same recall on the test dataset.

---

## 9. Model Evaluation & Results

The model evaluation includes confusion matrices, classification metrics and an ROC curve.

The supplied results contain two displayed confusion matrices.

### Confusion Matrix A

| Actual / Predicted | No | Yes |
|---|---:|---:|
| No | 10,939 | 25 |
| Yes | 1,281 | 111 |

**Classification Metrics**

| Metric | Result |
|---|---:|
| True Negatives | 10,939 |
| False Positives | 25 |
| False Negatives | 1,281 |
| True Positives | 111 |
| Accuracy (recalculated) | 89.43% |
| Precision | 81.62% |
| Recall | 7.97% |
| Specificity | 99.77% |

*Note: The uploaded output reports 89.10% accuracy, but the displayed confusion-matrix values produce 89.43% when accuracy is recalculated.*

### Confusion Matrix B

| Actual / Predicted | No | Yes |
|---|---:|---:|
| No | 10,889 | 75 |
| Yes | 1,165 | 227 |

**Classification Metrics**

| Metric | Result |
|---|---:|
| True Negatives | 10,889 |
| False Positives | 75 |
| False Negatives | 1,165 |
| True Positives | 227 |
| Accuracy | 89.96% |
| Precision | 75.17% |
| Recall | 16.31% |
| Specificity | 99.32% |

### Additional Accuracy Output

The supplied R output also prints an accuracy calculation of:

**89.83%**

However, this calculation uses TN = 10,873 and FP = 91, which do not match the second displayed confusion matrix.

Therefore, the three displayed accuracy figures should not be treated as interchangeable results.

### ROC Curve

An ROC curve was generated to visualize the relationship between:

- True Positive Rate
- False Positive Rate

The curve evaluates classification behaviour across different probability thresholds.

A numeric ROC-AUC value is not visible in the supplied output, so no specific ROC-AUC score is claimed.

### Results Interpretation

The displayed confusion matrices show:

- High specificity across both classifications.
- Relatively low recall.
- Matrix A recall of approximately 7.97%.
- Matrix B recall of approximately 16.31%.

Neither displayed confusion matrix demonstrates the intended validation recall target of at least 80%.

The results highlight the importance of evaluating classification models using multiple metrics rather than relying exclusively on accuracy.

### Detailed Results File

The complete transcribed model output, VIF values, confusion matrices and metric calculations are documented separately.

**See:** [`MODEL_RESULTS.txt`](MODEL_RESULTS.txt)

---

## 10. Why Call Duration Is Not Used

Call duration is only known after a telemarketing call has taken place.

If the model is intended to help identify potential customers before contacting them, using call duration would introduce information that is unavailable at the time of the decision.

Therefore, the project focuses on predictors that can be considered before the call rather than relying on call duration.

---

## 11. Technologies & Libraries

<p align="left">
  <img src="https://img.shields.io/badge/R-276DC3?style=flat-square&logo=r&logoColor=white"/>
  <img src="https://img.shields.io/badge/fastDummies-6C757D?style=flat-square"/>
  <img src="https://img.shields.io/badge/corrplot-6C757D?style=flat-square"/>
  <img src="https://img.shields.io/badge/caTools-6C757D?style=flat-square"/>
  <img src="https://img.shields.io/badge/car-6C757D?style=flat-square"/>
  <img src="https://img.shields.io/badge/ROCR-FF8C00?style=flat-square"/>
</p>

| Category | Technologies |
|---|---|
| Programming Language | R |
| Statistical Modelling | Generalized Linear Models |
| Machine Learning | Logistic Regression |
| Categorical Encoding | fastDummies |
| Correlation Analysis | corrplot |
| Data Splitting | caTools |
| Multicollinearity Analysis | car |
| Model Evaluation | ROCR |

### Required R Libraries

```r
install.packages(c(
  "fastDummies",
  "corrplot",
  "caTools",
  "car",
  "ROCR"
))
```

---

## 12. Repository Structure

```text
Optimizing-bank-telemarking-campaigns-/
│
├── README.md
│
├── main.R
│
├── bank-additional-full.csv
│
└── MODEL_RESULTS.txt
```

---

## 13. Running the Project

### Prerequisites

- R
- RStudio or another R-compatible development environment.

### Step 1: Clone the Repository

```bash
git clone https://github.com/karanpaul2006-gif/Optimizing-bank-telemarking-campaigns-.git
```

### Step 2: Install Required Packages

```r
install.packages(c(
  "fastDummies",
  "corrplot",
  "caTools",
  "car",
  "ROCR"
))
```

### Step 3: Execute the Analysis

1. Open `main.R`.
2. Ensure `bank-additional-full.csv` is available in the working directory expected by the script.
3. Execute the script sequentially.
4. Review the model summaries, VIF calculations, threshold analysis and evaluation outputs.

---

## 14. Project Limitations

- The analysis uses historical campaign information and does not establish causal relationships.
- Logistic regression may not capture all nonlinear relationships or interactions.
- Threshold optimization on validation data does not guarantee identical performance on unseen test data.
- The analysis does not include a complete financial cost-benefit calculation for campaign contacts.
- The model is an analytical prototype and has not been deployed in a production banking environment.
- The supplied output contains inconsistencies between displayed confusion matrices and the separately printed accuracy calculation.
- A clean, reproducible execution is required before claiming one definitive set of final model metrics.

---

## 15. Conclusion

This project demonstrates a structured predictive analytics workflow using R and logistic regression.

By progressively developing fourteen models, examining multicollinearity, encoding categorical variables and exploring probability thresholds, the project investigates how customer and campaign characteristics relate to term deposit subscription behaviour.

The threshold exploration introduces a practical campaign-management perspective by considering the trade-off between identifying potential subscribers and reducing unnecessary customer contacts.

The project demonstrates practical applications of statistical modelling, predictive analytics and classification evaluation in banking and marketing.

---

## Dataset Reference

Moro, S., Cortez, P., & Rita, P. (2014).

*A Data-Driven Approach to Predict the Success of Bank Telemarketing.*

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/222/bank+marketing
