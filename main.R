# ==============================================================================
# Bank Marketing Direct Telemarketing Outcome Prediction
# Model: Logistic Regression
# File: main.R
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: Load Required Libraries
# ------------------------------------------------------------------------------
required_packages <- c("fastDummies", "corrplot", "caTools", "car", "ROCR")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# ------------------------------------------------------------------------------
# STEP 2: Import Dataset and Initial Exploration
# ------------------------------------------------------------------------------
# Note: Update path to your local data file if necessary
data_path <- "bank-additional-full.csv"

if (file.exists(data_path)) {
  bank. <- read.csv(data_path, stringsAsFactors = TRUE)
} else {
  message("Please ensure 'bank-additional-full.csv' is in the working directory.")
}

# Display summary statistics of raw dataset
summary(bank.)

# ------------------------------------------------------------------------------
# STEP 3: Data Preprocessing & Feature Encoding
# ------------------------------------------------------------------------------
# Binary Label Encoding for binary target and categorical flags
bank.$y <- ifelse(bank.$y == "yes", 1, 0)
bank.$Housing_loan <- ifelse(bank.$Housing.loan == "yes", 1, 0)
bank.$Personal_loan <- ifelse(bank.$Personal.loan == "yes", 1, 0)
bank.$Default_status <- ifelse(bank.$Default == "yes", 1, 0)

# Dummy/One-Hot Encoding for multi-class categorical features
bank. <- dummy_cols(
  bank., 
  select_columns = c(
    "Job.profile", 
    "Marital.status", 
    "Education", 
    "Contact.type", 
    "previous.campaign.outcome"
  )
)

# Summary post dummy generation
summary(bank.)

# Correlation Matrix Analysis
numeric_cols <- bank.[sapply(bank., is.numeric)]
cr <- cor(numeric_cols)

# Save Correlation Matrix to CSV
write.csv(cr, "bank_cor.csv")

# Plot Correlation Matrix
corrplot(cr, method = "circle")
corrplot.mixed(cr)

# ------------------------------------------------------------------------------
# STEP 4: Train, Validation and Test Split

set.seed(111)

# 60% training
split_train <- sample.split(
  bank.$y,
  SplitRatio = 0.60
)

trainingbank. <- subset(
  bank.,
  split_train == TRUE
)

remainingbank. <- subset(
  bank.,
  split_train == FALSE
)

# Split remaining 40% equally
split_validation <- sample.split(
  remainingbank.$y,
  SplitRatio = 0.50
)

validationbank. <- subset(
  remainingbank.,
  split_validation == TRUE
)

testingbank. <- subset(
  remainingbank.,
  split_validation == FALSE
)
# ------------------------------------------------------------------------------
# STEP 5: Iterative Model Building & VIF Evaluation
# ------------------------------------------------------------------------------

# Model 1: Single economic baseline feature
model1 <- glm(y ~ Euro.Interbank.offered.rate, data = trainingbank., family = binomial)
summary(model1)

# Model 2: Adding Consumer Price Index
model2 <- glm(y ~ Euro.Interbank.offered.rate + consumer.price.index, data = trainingbank., family = binomial)
summary(model2)
vif(model2)

# Model 3: Adding Campaign contact numbers
model3 <- glm(y ~ Euro.Interbank.offered.rate + consumer.price.index + campaign, data = trainingbank., family = binomial)
summary(model3)
vif(model3)

# Model 4: Adding Successful prior campaign outcome
model4 <- glm(y ~ Euro.Interbank.offered.rate + consumer.price.index + campaign + previous.campaign.outcome_success, data = trainingbank., family = binomial)
summary(model4)
vif(model4)

# Model 5: Adding Client Age profile
model5 <- glm(y ~ Euro.Interbank.offered.rate + consumer.price.index + campaign + previous.campaign.outcome_success + age, data = trainingbank., family = binomial)
summary(model5)
vif(model5)

# Model 6: Adding Specific Job Profile Dummy (Blue-collar)
model6 <- glm(y ~ Euro.Interbank.offered.rate + consumer.price.index + campaign + previous.campaign.outcome_success + age + `Job.profile_blue-collar`, data = trainingbank., family = binomial)
summary(model6)
vif(model6)

# Model 7: Adding Consumer Confidence Index
model7 <- glm(y ~ Euro.Interbank.offered.rate + consumer.price.index + campaign + previous.campaign.outcome_success + age + `Job.profile_blue-collar` + consumer.confidence.index, data = trainingbank., family = binomial)
summary(model7)
vif(model7)

# Model 8: Adding pdays (Days since last contact)
model8 <- glm(y ~ Euro.Interbank.offered.rate + consumer.price.index + campaign + previous.campaign.outcome_success + age + `Job.profile_blue-collar` + consumer.confidence.index + pdays, data = trainingbank., family = binomial)
summary(model8)
vif(model8)

# Model 9: Adding Housing Loan
model9 <- glm(y ~ Euro.Interbank.offered.rate + consumer.price.index + campaign + previous.campaign.outcome_success + age + `Job.profile_blue-collar` + consumer.confidence.index + pdays + Housing_loan, data = trainingbank., family = binomial)
summary(model9)
vif(model9)

# Model 10: Adding Personal Loan
model10 <- glm(y ~ Euro.Interbank.offered.rate + consumer.price.index + campaign + previous.campaign.outcome_success + age + `Job.profile_blue-collar` + consumer.confidence.index + pdays + Housing_loan + Personal_loan, data = trainingbank., family = binomial)
summary(model10)
vif(model10)

# Model 11: Adding Job Profile Retired Dummy
model11 <- glm(y ~ Euro.Interbank.offered.rate + consumer.price.index + campaign + previous.campaign.outcome_success + age + `Job.profile_blue-collar` + consumer.confidence.index + pdays + Housing_loan + Personal_loan + Job.profile_retired, data = trainingbank., family = binomial)
summary(model11)
vif(model11)

# Model 12: Adding Job Profile Student Dummy
model12 <- glm(y ~ Euro.Interbank.offered.rate + consumer.price.index + campaign + previous.campaign.outcome_success + age + `Job.profile_blue-collar` + consumer.confidence.index + pdays + Housing_loan + Personal_loan + Job.profile_retired + Job.profile_student, data = trainingbank., family = binomial)
summary(model12)
vif(model12)

# Model 13: Adding Education University Degree Dummy
model13 <- glm(y ~ Euro.Interbank.offered.rate + consumer.price.index + campaign + previous.campaign.outcome_success + age + `Job.profile_blue-collar` + consumer.confidence.index + pdays + Housing_loan + Personal_loan + Job.profile_retired + Job.profile_student + Education_university.degree, data = trainingbank., family = binomial)
summary(model13)
vif(model13)

# Model 14: Final Model with Contact Type Cellular Dummy
model14 <- glm(
  y ~ Euro.Interbank.offered.rate + consumer.price.index + campaign + 
    previous.campaign.outcome_success + age + `Job.profile_blue-collar` + 
    consumer.confidence.index + pdays + Housing_loan + Personal_loan + 
    Job.profile_retired + Job.profile_student + Education_university.degree + 
    Contact.type_cellular, 
  data = trainingbank., 
  family = binomial
)
summary(model14)
vif(model14)

# ------------------------------------------------------------------------------
# STEP 6: Threshold Optimization

# Predict probabilities on validation dataset

validation_prob <- predict(
  model14,
  validationbank.,
  type = "response"
)

validation_actual <- as.integer(
  validationbank.$y
)

# Function for evaluating classification performance

calculate_metrics <- function(actual, probability, threshold) {

  predicted <- as.integer(probability >= threshold)

  TP <- sum(actual == 1 & predicted == 1)
  TN <- sum(actual == 0 & predicted == 0)
  FP <- sum(actual == 0 & predicted == 1)
  FN <- sum(actual == 1 & predicted == 0)

  precision <- if (TP + FP > 0) {
    TP / (TP + FP)
  } else {
    0
  }

  recall <- if (TP + FN > 0) {
    TP / (TP + FN)
  } else {
    0
  }

  specificity <- if (TN + FP > 0) {
    TN / (TN + FP)
  } else {
    0
  }

  f1 <- if (precision + recall > 0) {
    2 * precision * recall / (precision + recall)
  } else {
    0
  }

  data.frame(
    threshold = threshold,
    accuracy = (TP + TN) / length(actual),
    precision = precision,
    recall = recall,
    specificity = specificity,
    f1_score = f1,
    contacts_selected_pct = mean(predicted == 1) * 100,
    contacts_avoided_pct = (1 - mean(predicted == 1)) * 100,
    deposits_captured_pct = recall * 100
  )
}

# Evaluate multiple probability thresholds

thresholds <- seq(0.01, 0.99, by = 0.01)

validation_results <- do.call(
  rbind,
  lapply(thresholds, function(t) {
    calculate_metrics(
      validation_actual,
      validation_prob,
      t
    )
  })
)

# Select the threshold achieving at least 80% recall
# while avoiding the maximum possible contacts.

eligible <- subset(
  validation_results,
  recall >= 0.80
)

if (nrow(eligible) == 0) {

  stop("No threshold achieved the required 80% validation recall.")

}

best_threshold <- eligible[
  which.max(eligible$contacts_avoided_pct),
]

selected_threshold <- best_threshold$threshold

print("SELECTED THRESHOLD")
print(best_threshold)

# ------------------------------------------------------------------------------
# STEP 7: ROCR Performance Curve Analysis
# ------------------------------------------------------------------------------
ROCRpred <- prediction(res, testingbank.$y)
ROCRpref <- performance(ROCRpred, "tpr", "fpr")

# Plot ROC Curve with Cutoff threshold annotations
plot(ROCRpref, colorize = TRUE, print.cutoffs.at = seq(0.1, 0.9, by = 0.1))
