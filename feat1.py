import pandas as pd
import numpy as np

# ----------------------------------------
# Load dataset
# ----------------------------------------
df = pd.read_csv("first10_features_100rows.csv")

# ----------------------------------------
# INPUT FEATURES
# ----------------------------------------
input_cols = [
    "NumAtoms", "NumHeavyAtoms", "NumBonds", "NumRotatableBonds",
    "NumAliphaticRings", "NumAromaticRings", "NumHeteroatoms",
    "HBD", "HBA", "FormalCharge"
]

# ----------------------------------------
# FEATURE EXTRACTION (NEW FEATURES)
# ----------------------------------------

# 1. Atom density (compactness)
df["Atom_Density"] = df["NumAtoms"] / (df["NumBonds"] + 1e-6)

# 2. Heavy atom ratio
df["HeavyAtom_Ratio"] = df["NumHeavyAtoms"] / (df["NumAtoms"] + 1e-6)

# 3. Flexibility index
df["Flexibility_Index"] = df["NumRotatableBonds"] / (df["NumBonds"] + 1e-6)

# 4. Ring complexity
df["Ring_Complexity"] = (
    (df["NumAromaticRings"] + df["NumAliphaticRings"]) /
    (df["NumAtoms"] + 1e-6)
)

# 5. Aromatic ratio
df["Aromatic_Ratio"] = df["NumAromaticRings"] / (df["NumAliphaticRings"] + 1)

# 6. Heteroatom ratio
df["Heteroatom_Ratio"] = df["NumHeteroatoms"] / (df["NumAtoms"] + 1e-6)

# 7. Hydrogen bonding balance
df["Hbond_Balance"] = df["HBD"] / (df["HBA"] + 1)

# 8. Charge density
df["Charge_Density"] = df["FormalCharge"] / (df["NumAtoms"] + 1e-6)

# 9. Structural complexity
df["Structural_Complexity"] = df["NumBonds"] + df["NumRotatableBonds"]

# 10. Connectivity index
df["Connectivity_Index"] = df["NumBonds"] / (df["NumAtoms"] + 1e-6)

# ----------------------------------------
# CLEANING
# ----------------------------------------
df = df.replace([np.inf, -np.inf], 0)
df = df.fillna(0)

# ----------------------------------------
# SAVE DATASET
# ----------------------------------------
df.to_csv("feature_engineered_first10.csv", index=False)

print("Feature extraction completed!")
print("Final shape:", df.shape)
