# Use of machine learning to compare disease risk scores and propensity scores across complex confounding scenarios: a simulation study.

This is a package including logistic regression with pre-selected confounders, LASSO, MLP and XgBoost, to estimate propensity score and disease risk score.

As data we are using is imbalanced on treatment, stratification was applied with splitting training and validation data for tuning. 

Details of hyperparameter setting are documented in code, and can be changed by user based on different test case.
