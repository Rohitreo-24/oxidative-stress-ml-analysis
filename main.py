# =====================================================
# GST Oxidative Stress - Data Mining Project
# Ensemble Learning for Gene Expression Prediction
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor

# Load Dataset

print("Loading dataset...")

df = pd.read_csv("dataset.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nOriginal Dataset Shape:", df.shape)


# Data cleaning

# Replace infinity values
df = df.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan)

# Replace "-" values
df = df.replace("-", np.nan)

# Keep only numeric columns
df_numeric = df.select_dtypes(include=[np.number])

print("\nShape after keeping numeric columns:", df_numeric.shape)

# Remove rows with missing values
df_numeric = df_numeric.dropna()

print("Shape after dropping NaN:", df_numeric.shape)

df = df_numeric


# Feature and target selection

target_column = df.columns[-1]

X = df.drop(target_column, axis=1)
y = df[target_column]

print("\nNumber of Features:", X.shape[1])


# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test Split Completed")
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# Feature scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Linear regression (baseline)

lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

print("\n===== Linear Regression =====")
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))


# Random forest (strong ensemble)

rf = RandomForestRegressor(random_state=42)

rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

print("\n===== Random Forest =====")
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))


# XGBoost (Modern boosting model)

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)

xgb.fit(X_train_scaled, y_train)

y_pred_xgb = xgb.predict(X_test_scaled)

print("\n===== XGBoost =====")
print("R2 Score:", r2_score(y_test, y_pred_xgb))


# Cross validation (model stability)

cv_score = cross_val_score(xgb, X_train_scaled, y_train, cv=5, scoring='r2')

print("\n===== XGBoost Cross Validation =====")
print("Mean R2:", cv_score.mean())


# Stacking Ensemble (Novel Model)

estimators = [
    ('rf', rf),
    ('xgb', xgb)
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)

stack.fit(X_train_scaled, y_train)

y_pred_stack = stack.predict(X_test_scaled)

print("\n===== Stacking Ensemble =====")
print("R2 Score:", r2_score(y_test, y_pred_stack))


# Model comparison

results = {
    "Linear Regression": r2_score(y_test, y_pred_lr),
    "Random Forest": r2_score(y_test, y_pred_rf),
    "XGBoost": r2_score(y_test, y_pred_xgb),
    "Stacking Ensemble": r2_score(y_test, y_pred_stack)
}

print("\n===== Model Comparison =====")

for model, score in results.items():
    print(model, ":", score)


# Visualization (Actual vs Predicted)

plt.figure()

plt.scatter(y_test, y_pred_rf)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted (Random Forest)")

plt.show()


# Feature Importance (Gene Analysis)

importances = rf.feature_importances_

indices = np.argsort(importances)[-10:]

plt.figure()

plt.barh(range(len(indices)), importances[indices])

plt.yticks(range(len(indices)), X.columns[indices])

plt.title("Top 10 Important Gene Features")

plt.xlabel("Importance Score")

plt.show()