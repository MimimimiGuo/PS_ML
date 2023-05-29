# Machine learning methods for propensity score estimation: a cross validation hyperparameter tuning framework.

This is a function including: logistic regression with pre-selected confounders, LASSO, MLP and XgBoost, to estimate propensity score.

As data we are using is imbalanced on treatment, stratification was applied with splitting training and validation data for tuning. 

Details of hyperparameter setting are documented in code, and can be changed by user based on different test case.
