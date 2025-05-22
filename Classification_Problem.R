# Load libraries 
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,
  caret,
  janitor,
  rpart,
  randomForest,
  e1071,
  corrplot,
  knitr,
  glue
)

# Read data
loans_raw <- read_csv("loan_data.csv")

# Clean up column names (make them lowercase with underscores)
loans <- clean_names(loans_raw)

# Quick look at data
print(head(loans))
# Check missing values
print(colSums(is.na(loans)))  

# Rename some columns
loans <- loans %>%
  rename(
    age                    = person_age,
    gender                 = person_gender,
    education_level        = person_education,
    annual_income          = person_income,
    employment_years       = person_emp_exp,
    home_ownership         = person_home_ownership,
    loan_amount            = loan_amnt,
    loan_purpose           = loan_intent,
    interest_rate          = loan_int_rate,
    credit_history_length  = cb_person_cred_hist_length,
    previous_defaults      = previous_loan_defaults_on_file,
    loan_approved          = loan_status
  )

glimpse(loans)
summary(loans)

# Check some outliers
n_age_issues <- sum(loans$age < 18 | loans$age > 100)
n_exp_issues <- sum(loans$employment_years > 80)

cat("Weird age values:", n_age_issues, "\n")
cat("Too much work experience (>80 yrs):", n_exp_issues, "\n")

# See how balanced the classes are
loans %>%
  count(loan_approved) %>%
  ggplot(aes(x = factor(loan_approved), y = n)) +
  geom_col(fill = "steelblue") +
  labs(
    title = "Loan Approval Outcome",
    x = "Approval (1 = Yes, 0 = No)",
    y = "Number of Applicants"
  )

# Correlation matrix
num_cols <- loans %>% select(where(is.numeric))
cor_matrix <- cor(num_cols, use = "pairwise.complete.obs")
corrplot(cor_matrix, method = "color", tl.cex = 0.8)

# Drop few variables 
loans <- loans %>% select(-age, -annual_income)

# Split data for train/test
set.seed(123)
idx <- createDataPartition(loans$loan_approved, p = 0.75, list = FALSE)
train_data <- loans[idx, ]
test_data  <- loans[-idx, ]

# One-hot encode categoricals
encoder <- dummyVars(loan_approved ~ ., data = train_data)
train_x <- predict(encoder, train_data) %>% as.data.frame()
test_x  <- predict(encoder, test_data) %>% as.data.frame()
train_y <- train_data$loan_approved
test_y  <- test_data$loan_approved

# Standardize features 
prep <- preProcess(train_x, method = c("center", "scale"))
train_x <- predict(prep, train_x)
test_x  <- predict(prep, test_x)

# Cross-validation setup
ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

# Train a few models to compare
models <- list()

# Logistic regression 
models$log_reg <- train(train_x, factor(train_y), method = "glm",
                        family = "binomial", trControl = ctrl)

# KNN
models$knn <- train(train_x, factor(train_y), method = "knn",
                    tuneLength = 10, trControl = ctrl)

# Decision tree
models$tree <- train(train_x, factor(train_y), method = "rpart",
                     tuneLength = 10, trControl = ctrl)

# Random forest
models$rf <- train(train_x, factor(train_y), method = "rf",
                   ntree = 200, tuneLength = 5, trControl = ctrl)

# SVM
models$svm <- train(train_x, factor(train_y), method = "svmLinear",
                    tuneLength = 5, trControl = ctrl)

# Evaluate models
evaluate_model <- function(model, x, y_true) {
  y_pred <- predict(model, x)
  conf <- confusionMatrix(y_pred, factor(y_true, levels = levels(y_pred)))
  
  tibble(
    accuracy  = conf$overall["Accuracy"],
    precision = conf$byClass["Precision"],
    recall    = conf$byClass["Recall"],
    f1        = conf$byClass["F1"]
  )
}

# Run evaluations
results <- imap_dfr(models, ~ {
  evaluate_model(.x, test_x, test_y) %>%
    mutate(model = .y)
}) %>%
  arrange(desc(f1))

# Show model results
kable(results, digits = 3, caption = "Model Performance on Test Set")

# Final summary
best_model <- results$model[1]
best_f1 <- round(results$f1[1], 3)
best_params <- models[[best_model]]$bestTune %>%
  mutate_all(as.character) %>%
  unite("params", everything(), sep = ", ") %>%
  pull()

cat(glue(
  "\nBest model: {best_model}\n",
  "Parameters: {best_params}\n",
  "F1 Score: {best_f1}\n"
))
