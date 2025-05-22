"""
regression_model_selection.py
This script performs model selection for a regression task using a standard dataset.
It includes the following steps:
1. Load a dataset and split into train/test sets.
2. Train multiple regression models and evaluate R^2 scores.
3. Apply k-fold cross-validation to assess model stability.
4. Use GridSearchCV to tune hyperparameters of each model.
5. Select the best model and report final evaluation metrics (R^2, and AIC/BIC).
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# =============================
# 1. Load Data and Preprocessing
# =============================

# Load the diabetes dataset (regression problem)
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)  # Features as DataFrame
y = pd.Series(data.target)                              # Target

# Display basic information about the data
print("Dataset shape:", X.shape)
print("First few target values:\n", y.head())

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)
print("\nAfter train-test split:")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# ====================================
# 2. Define Regression Models
# ====================================

# We'll compare the following models:
# - Linear Regression (basic linear model)
# - K-Nearest Neighbors (regressor)
# - Decision Tree (regression tree)
# - Random Forest (ensemble of regression trees)
# - Gradient Boosting (ensemble boosting regressor)
models = {
    "Linear Regression": LinearRegression(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# ====================================
# 3. Train Models and Evaluate
# ====================================

print("\n--- Initial Model Training and Evaluation ---")
for name, model in models.items():
    # Train the model on the training set
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    # Calculate R^2 on the test set
    r2 = r2_score(y_test, y_pred)
    print(f"{name} R^2 on Test: {r2:.4f}")

# ====================================
# 4. Cross-Validation
# ====================================

print("\n--- 5-Fold Cross-Validation Scores (R^2) ---")
print("Using 5-fold CV on the training set to evaluate stability.\n")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"{name} CV R^2 scores: {scores}")
    print(f"{name} Mean CV R^2: {scores.mean():.4f}\n")

# ====================================
# 5. Hyperparameter Tuning with GridSearchCV
# ====================================

print("\n--- Hyperparameter Tuning with GridSearchCV ---")

# Define hyperparameter grids (Linear Regression has no parameters to tune)
param_grids = {
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    },
    "Decision Tree": {
        'max_depth': [None, 3, 5, 7, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        "criterion": ["squared_error", "friedman_mse"]  
    },
    "Random Forest": {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5],
        "criterion": ["squared_error", "friedman_mse"]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 10, 20, 30]
    }
}

best_models = {}

# Skip Linear Regression in grid search (no hyperparameters here)
for name, model in models.items():
    if name == "Linear Regression":
        continue
    print(f"\n{name}:")
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grids[name],
                               cv=5,
                               scoring='r2',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("  Best Parameters:", grid_search.best_params_)
    print("  Best CV R^2:    ", f"{grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    y_pred_best = best_model.predict(X_test)
    best_r2 = r2_score(y_test, y_pred_best)
    print(f"  Test R^2 with best params: {best_r2:.4f}")

# ====================================
# 6. Final Evaluation and AIC/BIC
# ====================================

print("\n--- Final Model Selection and AIC/BIC ---")

# Identify the best model based on R^2 on the test set (among tuned models)
best_name = None
best_test_r2 = -np.inf
for name, model in best_models.items():
    r2 = r2_score(y_test, model.predict(X_test))
    if r2 > best_test_r2:
        best_test_r2 = r2
        best_name = name

print(f"Best Model: {best_name} with Test R^2: {best_test_r2:.4f}")

# Fit Linear Regression on training data for AIC/BIC calculation
linear_model = models["Linear Regression"]
linear_model.fit(X_train, y_train)
X_train_sm = sm.add_constant(X_train)  # add intercept term
ols_model = sm.OLS(y_train, X_train_sm).fit()
aic_value = ols_model.aic
bic_value = ols_model.bic
print(f"Linear Regression AIC: {aic_value:.2f}")
print(f"Linear Regression BIC: {bic_value:.2f}")

# Also, print final R^2 and MSE of the best model
print(f"\nFinal R^2 of Best Model ({best_name}): {best_test_r2:.4f}")
y_best_pred = best_models[best_name].predict(X_test)
mse = mean_squared_error(y_test, y_best_pred)
print(f"Final MSE of Best Model: {mse:.2f}")

# =============================
# End of Regression Model Selection Script
# =============================