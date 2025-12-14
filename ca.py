from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('credit_risk_dataset.csv')
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna().reset_index(drop=True)

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

df = pd.get_dummies(df, drop_first=True)

X = df.drop('loan_status', axis=1)
Y = df['loan_status']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Simple Linear Regression
X_reg = df[['person_income']]
Y_reg = df['loan_int_rate']

Xr_train, Xr_test, Yr_train, Yr_test = train_test_split(
    X_reg, Y_reg, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(Xr_train, Yr_train)
pred_lr = lr.predict(Xr_test)

print("\nSimple Linear Regression:")
print("MAE:", mean_absolute_error(Yr_test, pred_lr))
print("R2:", r2_score(Yr_test, pred_lr))

# Multiple Linear Regression
X_reg2 = df[['person_income', 'person_age', 'loan_amnt']]
Y_reg2 = df['loan_int_rate']

Xr2_train, Xr2_test, Yr2_train, Yr2_test = train_test_split(
    X_reg2, Y_reg2, test_size=0.2, random_state=42
)

mlr = LinearRegression()
mlr.fit(Xr2_train, Yr2_train)
pred_mlr = mlr.predict(Xr2_test)

print("\nMultiple Linear Regression:")
print("MAE:", mean_absolute_error(Yr2_test, pred_mlr))
print("R2:", r2_score(Yr2_test, pred_mlr))

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_reg)

Xp_train, Xp_test, Yp_train, Yp_test = train_test_split(
    X_poly, Y_reg, test_size=0.2, random_state=42
)

poly_model = LinearRegression()
poly_model.fit(Xp_train, Yp_train)
pred_poly = poly_model.predict(Xp_test)

print("\nPolynomial Regression:")
print("MAE:", mean_absolute_error(Yp_test, pred_poly))
print("R2:", r2_score(Yp_test, pred_poly))


# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, Y_train)
pred_log = log_reg.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, Y_train)
pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, Y_train)
pred_rf = rf.predict(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
pred_knn = knn.predict(X_test)

print("\nClassification Results:")
print("Logistic Accuracy:", accuracy_score(Y_test, pred_log))
print("Decision Tree Accuracy:", accuracy_score(Y_test, pred_dt))
print("Random Forest Accuracy:", accuracy_score(Y_test, pred_rf))
print("KNN Accuracy:", accuracy_score(Y_test, pred_knn))

models = ['Logistic', 'Decision Tree', 'Random Forest', 'KNN']
accuracies = [
    accuracy_score(Y_test, pred_log),
    accuracy_score(Y_test, pred_dt),
    accuracy_score(Y_test, pred_rf),
    accuracy_score(Y_test, pred_knn)
]

plt.figure()
plt.bar(models, accuracies)
plt.title("Accuracy Comparison of Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()