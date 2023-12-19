# Customer_Churn_Prediction
predicting if customer can leave the company with logistic regression

Description: You are data scientist working for bank. Based on data, you have to build Logistic
Regression model which will predict customer churn by applying the following steps:
1. Find multicollinearity by applying VIF;
2. Standardize features;
3. Split data into train and test sets using seed=123;
4. Exclude unimportant variables (information value should be > 0.02);
5. Apply binning according to Weight of Evidence principle;
6. Build a logistic regression model. p value variables should be max 0.05;
7. Find threshold by max f1 score;
8. Calculate AUC score both for train and test sets;
9. Check overfitting.

Note: dataset name is “bank_full.csv”, predicted variable name is “y”.
