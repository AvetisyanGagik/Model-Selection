# Model Selection

## Install Packages from `requirements.txt`

```r
packages <- readLines("requirements.txt")
install.packages(setdiff(packages, rownames(installed.packages())))
====================================================================

Classification script will:
  -> Loan_data.csv
  -> Load and clean the loan dataset
  -> Explore the data (missing values, outliers)
  -> Split into train/test sets  
  -> Train multiple models (Logistic Regression, KNN, Decision Tree, Random Forest, SVM)
  -> Compare performance using F1 Score
  -> Output the best model with hyperparameters
====================================================================

Regression script will:
  -> Food_Delivery_Times.csv
  -> Load the delivery dataset and handle missing values
  -> One-hot encode categorical features
  -> Train regression models (Ridge, Decision Tree, Random Forest, GBM, SVM)
  -> Evaluate them using RMSE, R², AIC, BIC
  -> Display the top model and a plot comparing AIC/BIC across models
