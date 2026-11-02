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
# STEP 4: Train-Test Split (70:30)
# ------------------------------------------------------------------------------
set.seed(111)
split <- sample.split(bank.$y, SplitRatio = 0.7)
trainingbank. <- subset(bank., split == TRUE)
testingbank.  <- subset(bank., split == FALSE)

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
# STEP 6: Predictions & Threshold Accuracy Evaluation
# ------------------------------------------------------------------------------
res <- predict(model14, testingbank., type = "response")
head(res)
head(testingbank.$y)

# Confusion Matrix for Cutoff = 0.75
cm_75 <- table(Actual = testingbank.$y, Predicted = res > 0.75)
print("Confusion Matrix (Cutoff = 0.75):")
print(cm_75)
acc_75 <- sum(diag(cm_75)) / sum(cm_75)
cat("Accuracy (Cutoff 0.75):", round(acc_75 * 100, 2), "%\n")

# Confusion Matrix for Cutoff = 0.60
cm_60 <- table(Actual = testingbank.$y, Predicted = res > 0.60)
print("Confusion Matrix (Cutoff = 0.60):")
print(cm_60)
acc_60 <- sum(diag(cm_60)) / sum(cm_60)
cat("Accuracy (Cutoff 0.60):", round(acc_60 * 100, 2), "%\n")

# ------------------------------------------------------------------------------
# STEP 7: ROCR Performance Curve Analysis
# ------------------------------------------------------------------------------
ROCRpred <- prediction(res, testingbank.$y)
ROCRpref <- performance(ROCRpred, "tpr", "fpr")

# Plot ROC Curve with Cutoff threshold annotations
plot(ROCRpref, colorize = TRUE, print.cutoffs.at = seq(0.1, 0.9, by = 0.1))
