# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================
# 2. LOAD TRAINING DATA
# =========================================
train_df = pd.read_csv("Research.csv")
train_df.columns = train_df.columns.str.strip()

# Shuffle dataset and reset index to avoid positional misalignment
# FIX #2: reset_index(drop=True) ensures clean 0-based index after shuffle
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# =========================================
# 3. DETECT Tg COLUMN
# =========================================
# FIX #4: Use startswith check instead of substring match to avoid
# false positives on columns like 'strategy', 'category', etc.
target_col = None
for col in train_df.columns:
    col_lower = col.lower()
    if col_lower.startswith('tg') or col_lower == 'tg_k':
        target_col = col
        break

if target_col is None:
    raise ValueError(
        "No Tg column found! Expected a column starting with 'tg' (e.g. 'Tg_K')."
    )

print("Using target:", target_col)

# =========================================
# 4. FEATURES & TARGET
# =========================================
X = train_df.drop(columns=[target_col])
X = X.select_dtypes(include=[np.number])

y = train_df[target_col]

# =========================================
# 5. TRAIN-TEST SPLIT
# =========================================
# FIX #1: Split BEFORE fillna to prevent data leakage.
# Computing mean on full X (including test rows) leaks test statistics
# into the imputation step. We now compute mean on X_train only,
# then apply that same mean to X_test and X_pred.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Compute imputation mean from training set only
train_means = X_train.mean()

# Apply to both splits using training statistics
X_train = X_train.fillna(train_means)
X_test  = X_test.fillna(train_means)

# =========================================
# 6. XGBOOST MODEL
# =========================================
# FIX #3: Added early_stopping_rounds so training halts automatically
# when validation loss stops improving, preventing overfitting.
# n_estimators now acts as an upper bound, not a fixed target.
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=30,   # stop if no improvement for 30 rounds
    eval_metric="rmse"          # metric used to monitor validation loss
)

# Pass eval_set so XGBoost can monitor held-out performance each round
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print(f"Best iteration: {model.best_iteration} / 500")

# =========================================
# 7. EVALUATION
# =========================================
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

# FIX #5: Added RMSE alongside MAE and R2 for complete evaluation
print("\nModel Performance:")
print(f"  MAE  : {mae:.4f} K")
print(f"  RMSE : {rmse:.4f} K")
print(f"  R²   : {r2:.4f}")

# =========================================
# 8. SAVE MODEL
# =========================================
joblib.dump(model, "xgb_model.pkl")
print("\nModel saved as xgb_model.pkl")

# =========================================
# 9. EXPORT FEATURE IMPORTANCE
# =========================================
# FIX #6: Save feature importances for model interpretability.
# This shows which molecular descriptors drive Tg predictions most.
feat_imp = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

feat_imp.to_csv("feature_importance.csv", header=["importance"])
print("Feature importances saved to feature_importance.csv")
print("\nTop 5 features:")
print(feat_imp.head(5).to_string())

# =========================================
# 10. LOAD PREDICTION DATA
# =========================================
pred_df = pd.read_csv("aromatic_clean_sequential.csv")
pred_df.columns = pred_df.columns.str.strip()

# Keep only numeric features
X_pred = pred_df.select_dtypes(include=[np.number])

# Match training feature columns exactly (order + presence)
X_pred = X_pred.reindex(columns=X_train.columns, fill_value=0)

# FIX #1 (continued): Use training means (not pred means) to fill NaNs
X_pred = X_pred.fillna(train_means)

# =========================================
# 11. PREDICT Tg
# =========================================
predictions = model.predict(X_pred)

# Add predictions to output
pred_df["Tg_K_predicted"] = predictions

# =========================================
# 12. SAVE OUTPUT
# =========================================
pred_df.to_csv("predicted_output_xgb.csv", index=False)
print("\n Predictions saved to predicted_output_xgb.csv")
print(f"  Predicted {len(predictions)} molecules")
print(f"  Tg range: {predictions.min():.1f} K — {predictions.max():.1f} K")
