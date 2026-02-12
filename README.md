# ================================
# MATERIAL PROPERTY PREDICTION
# Random Forest Regression
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# --------------------------------
# 1. Load Dataset
# --------------------------------

df = pd.read_csv("material_property_prediction_dataset.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# --------------------------------
# 2. Clean Missing Values
# --------------------------------

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df = df.dropna()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# --------------------------------
# 3. Convert Categorical to Numerical
# --------------------------------

le = LabelEncoder()
df["Crystal_Structure"] = le.fit_transform(df["Crystal_Structure"])

# --------------------------------
# 4. Define Features and Target
# --------------------------------

X = df[[
    "Fe_percent",
    "C_percent",
    "Ni_percent",
    "Density_g_cm3",
    "Crystal_Structure"
]]

y = df["Melting_Point_C"]

# --------------------------------
# 5. Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# 6. Train Random Forest Model
# --------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------
# 7. Make Predictions
# --------------------------------

y_pred = model.predict(X_test)

# --------------------------------
# 8. Evaluate Model
# --------------------------------

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("R² Score:", round(r2, 4))

# --------------------------------
# 9. Plot Predictions vs Actual
# --------------------------------

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Melting Point (°C)")
plt.ylabel("Predicted Melting Point (°C)")
plt.title("Random Forest: Actual vs Predicted")
plt.show()
