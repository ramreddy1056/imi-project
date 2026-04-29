# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import euclidean_distances

# =========================================
# 2. LOAD TRAINED MODEL
# =========================================
model = joblib.load("xgb_model.pkl")

# =========================================
# 3. LOAD DATASET (USED FOR SEARCH SPACE)
# =========================================
df = pd.read_csv("final_dataset_ready.csv")
df.columns = df.columns.str.strip()

# =========================================
# 4. IDENTIFY TARGET COLUMN
# =========================================
target_col = None
for col in df.columns:
    if 'tg' in col.lower():
        target_col = col
        break

if target_col is None:
    raise ValueError("No Tg column found!")

# =========================================
# 5. EXTRACT FEATURES
# =========================================
X = df.select_dtypes(include=["number"]).drop(columns=[target_col])

# Save SMILES (adjust if needed)
if "SMILES" in df.columns:
    smiles_list = df["SMILES"]
else:
    raise ValueError("SMILES column not found!")

# Fill missing values
X = X.fillna(X.mean())

print("Feature shape:", X.shape)
print("Tg Range:", df[target_col].min(), "to", df[target_col].max())

# =========================================
# 6. SET TARGET Tg
# =========================================
target_tg = 180   # 

# =========================================
# 7. INVERSE SEARCH
# =========================================
all_candidates = []

iterations = 20000  # can increase to 50000 later

for i in range(iterations):

    # pick random real molecule
    sample = X.sample(1).values.astype(float)[0]

    # add noise (explore nearby space)
    noise = np.random.normal(0, 1.0, size=sample.shape)
    new_sample = sample + noise

    # keep within valid descriptor range
    new_sample = np.clip(new_sample, X.min().values, X.max().values)

    # convert to dataframe
    new_sample_df = pd.DataFrame([new_sample], columns=X.columns)

    # predict Tg
    pred_tg = model.predict(new_sample_df)[0]

    # score (how close to target)
    score = abs(pred_tg - target_tg)

    all_candidates.append((new_sample, pred_tg, score))

# =========================================
# 8. SELECT BEST MATCHES
# =========================================
all_candidates = sorted(all_candidates, key=lambda x: x[2])
top_candidates = all_candidates[:10]

print("\nTop candidates selected.")

# =========================================
# 9. MAP BACK TO REAL MOLECULES
# =========================================
results = []
used_indices = set()

for sample, tg, score in top_candidates:

    distances = euclidean_distances([sample], X.values)
    idx = distances.argmin()

    if idx in used_indices:
        continue

    used_indices.add(idx)

    smiles = smiles_list.iloc[idx]

    results.append({
        "SMILES": smiles,
        "Predicted_Tg": round(tg, 2),
        "Error": round(score, 2)
    })

# =========================================
# 10. SAVE RESULTS
# =========================================
result_df = pd.DataFrame(results)
result_df.to_csv("inverse_design_results.csv", index=False)

# =========================================
# 11. PRINT RESULTS
# =========================================
print("\nTop Designed Molecules:")
print(result_df)

for i, row in result_df.iterrows():
    print(f"\nCandidate {i+1}")
    print("SMILES:", row["SMILES"])
    print("Predicted Tg:", row["Predicted_Tg"])
    print("Error:", row["Error"])
