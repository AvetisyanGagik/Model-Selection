# Install packages
install.packages(c(
  "readr", "dplyr", "ggplot2", "tidyr", "caret", "glmnet",
  "rpart", "randomForest", "gbm", "e1071", "future", "doFuture"
), repos = "https://cloud.r-project.org")

install.packages("future", repos = "https://cloud.r-project.org", type = "source")

# Load the libraries
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(glmnet)
library(rpart)
library(randomForest)
library(gbm)
library(e1071)
library(future)
library(doFuture)

# Enable parallel processing
registerDoFuture()
plan(multisession)

# Load the dataset
data <- read_csv("Food_Delivery_Times.csv")

# Check for missing values and dataset structure
colSums(is.na(data))
cat("Rows:", nrow(data), " | Columns:", ncol(data), "\n")
str(data)

# Remove the Order ID since it doesn't help with predictions
data <- data %>% select(-Order_ID)

# Identify target, numeric and categorical variables
target_var <- "Delivery_Time_min"
num_vars <- names(data)[sapply(data, is.numeric)] %>% setdiff(target_var)
cat_vars <- setdiff(names(data), c(num_vars, target_var))

# Fill missing values: categorical with mod, numeric with mean
for (col in cat_vars) {
  common_val <- names(sort(table(data[[col]]), decreasing = TRUE))[1]
  data[[col]][is.na(data[[col]])] <- common_val
}
for (col in num_vars) {
  avg_val <- mean(data[[col]], na.rm = TRUE)
  data[[col]][is.na(data[[col]])] <- avg_val
}
stopifnot(all(colSums(is.na(data)) == 0))

# Convert categorical variables to dummy/one-hot encoded format
dummy_encoder <- dummyVars(~ ., data = data %>% select(-all_of(target_var)), fullRank = TRUE)
features_encoded <- predict(dummy_encoder, newdata = data)
features_df <- as.data.frame(features_encoded)

# Define inputs and target
X <- features_df
y <- data[[target_var]]

# Standardize numeric columns
cols_to_scale <- intersect(names(X), num_vars)
X[cols_to_scale] <- scale(X[cols_to_scale])

# Split into training (80%) and test (20%) sets
set.seed(42)
split_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[split_idx, ]
y_train <- y[split_idx]
X_test  <- X[-split_idx, ]
y_test  <- y[-split_idx]

# Remove any constant columns from training and test data
constant_cols <- names(which(sapply(X_train, function(col) var(col) == 0)))
if (length(constant_cols) > 0) {
  cat("Dropping constant columns:", paste(constant_cols, collapse = ", "), "\n")
  X_train <- X_train[, !names(X_train) %in% constant_cols]
  X_test  <- X_test[, !names(X_test) %in% constant_cols]
}

# Define various models to train with their hyperparameters
model_configs <- list(
  Ridge = list(method = "glmnet", tuneGrid = expand.grid(alpha = 0, lambda = c(0.1, 1, 10))),
  Tree = list(method = "rpart", tuneGrid = expand.grid(cp = c(0.01, 0.05, 0.1))),
  Forest = list(method = "rf", tuneGrid = expand.grid(mtry = floor(sqrt(ncol(X_train)))), ntree = 100),
  GBM = list(method = "gbm", tuneGrid = expand.grid(
    n.trees = c(50, 100, 150), interaction.depth = c(1, 3, 5),
    shrinkage = c(0.01, 0.1), n.minobsinnode = 10), verbose = FALSE),
  SVM_RBF = list(method = "svmRadial", tuneGrid = expand.grid(C = c(0.1, 1, 10), sigma = 0.1)),
  SVM_Linear = list(method = "svmLinear", tuneGrid = expand.grid(C = c(0.1, 1, 10)))
)

# Train each model using 5-fold cross-validation
ctrl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
trained_models <- list()

for (model_name in names(model_configs)) {
  cat("Training:", model_name, "\n")
  config <- model_configs[[model_name]]
  
  model_args <- list(
    x = X_train,
    y = y_train,
    method = config$method,
    trControl = ctrl,
    tuneGrid = config$tuneGrid
  )
  if (!is.null(config$ntree)) model_args$ntree <- config$ntree
  if (!is.null(config$verbose)) model_args$verbose <- config$verbose
  
  fit <- do.call(train, model_args)
  trained_models[[model_name]] <- fit
  
  if (any(is.na(fit$results$RMSE))) {
    warning(paste(model_name, "has NA RMSE values, bad values. Might be an issue with data or grid."))
  } else {
    cat(model_name, "- Best RMSE from CV:", min(fit$results$RMSE), "\n")
  }
}

# Test models on the test set and calculate evaluation metrics
metrics <- data.frame(Model=character(), MSE=numeric(), RMSE=numeric(), Rsq=numeric(), AIC=numeric(), BIC=numeric())
n_test <- length(y_test)

for (model_name in names(trained_models)) {
  fitted_model <- trained_models[[model_name]]
  preds <- predict(fitted_model, newdata = X_test)
  
  mse <- mean((y_test - preds)^2)
  rmse <- sqrt(mse)
  rsq <- cor(y_test, preds)^2
  k <- ncol(X_test) + 1
  rss <- sum((y_test - preds)^2)
  aic <- n_test * log(rss / n_test) + 2 * k
  bic <- n_test * log(rss / n_test) + k * log(n_test)
  
  metrics <- rbind(metrics, data.frame(
    Model = model_name,
    MSE = mse,
    RMSE = rmse,
    Rsq = rsq,
    AIC = aic,
    BIC = bic
  ))
  
  cat(sprintf("%s -> RMSE: %.2f | R²: %.3f | AIC: %.2f | BIC: %.2f\n", model_name, rmse, rsq, aic, bic))
}

# Model with the lowest RMSE
best_model <- metrics[which.min(metrics$RMSE), ]
cat(sprintf("Top model: %s with RMSE = %.3f\n", best_model$Model, best_model$RMSE))

# Plot AIC and BIC scores for all models
plot_df <- metrics %>%
  pivot_longer(cols = c(AIC, BIC), names_to = "Metric", values_to = "Value")

ggplot(plot_df, aes(x = Model, y = Value, color = Metric, group = Metric)) +
  geom_point(size = 3) +
  geom_line() +
  labs(title = "AIC & BIC Scores for Different Models", y = "Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
