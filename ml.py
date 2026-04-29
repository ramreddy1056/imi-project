# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================
# 2. LOAD TRAINING DATA
# =========================================
train_df = pd.read_csv("Research.csv")
train_df.columns = train_df.columns.str.strip()

# Shuffle dataset
train_df = train_df.sample(frac=1, random_state=42)

# =========================================
# 3. DETECT Tg COLUMN
# =========================================
target_col = None
for col in train_df.columns:
    if 'tg' in col.lower():
        target_col = col
        break

if target_col is None:
    raise ValueError("No Tg column found!")

print("Using target:", target_col)

# =========================================
# 4. FEATURES & TARGET
# =========================================
X = train_df.drop(columns=[target_col])
X = X.select_dtypes(include=[np.number])

y = train_df[target_col]

# Handle missing values
X = X.fillna(X.mean())

# =========================================
# 5. TRAIN-TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 6. XGBOOST MODEL
# =========================================
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# =========================================
# 7. EVALUATION
# =========================================
y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# =========================================
# 8. SAVE MODEL
# =========================================
joblib.dump(model, "xgb_model.pkl")
print("✅ Model saved as xgb_model.pkl")

# =========================================
# 9. LOAD PREDICTION DATA
# =========================================
pred_df = pd.read_csv("aromatic_clean_sequential.csv")
pred_df.columns = pred_df.columns.str.strip()

# Keep only numeric features
X_pred = pred_df.select_dtypes(include=[np.number])

# Match training feature columns
X_pred = X_pred.reindex(columns=X.columns, fill_value=0)

# Handle missing values
X_pred = X_pred.fillna(X.mean())

# =========================================
# 10. PREDICT Tg
# =========================================
predictions = model.predict(X_pred)

# Add predictions
pred_df["Tg"] = predictions

# =========================================
# 11. SAVE OUTPUT
# =========================================
pred_df.to_csv("predicted_output_xgb.csv", index=False)

print("✅ Predictions saved to predicted_output_xgb.csv")
