import pandas as pd
import numpy as np

# ----------------------------------------
# Load dataset
# ----------------------------------------
df = pd.read_csv("imidataset.csv")

# ----------------------------------------
# INPUT FEATURES (your dataset)
# ----------------------------------------
input_cols = [
    "MolecularVolume",
    "Polarizability",
    "FractionCSP3",
    "TopologicalPolarizabilityIndex",
    "WienerIndex",
    "BalabanIndex",
    "KappaShapeIndex",
    "ElectrotopologicalStateIndex",
    "VanDerWaalsSurfaceArea",
    "ChainStiffnessIndex"
]

# ----------------------------------------
# FEATURE EXTRACTION (NEW FEATURES)
# ----------------------------------------

# 1. Volume to surface ratio
df["Volume_Surface_Ratio"] = df["MolecularVolume"] / (df["VanDerWaalsSurfaceArea"] + 1e-6)

# 2. Shape normalized by stiffness
df["Shape_Normalized"] = df["KappaShapeIndex"] / (df["MolecularVolume"] + 1e-6)

# 3. Flexibility index
df["Flexibility_Index"] = df["FractionCSP3"] / (df["ChainStiffnessIndex"] + 1e-6)

# 4. Density-like feature
df["Mass_Density"] = df["Polarizability"] / (df["MolecularVolume"] + 1e-6)

# 5. Polarity ratio
df["Polarity_Index"] = df["ElectrotopologicalStateIndex"] / (df["Polarizability"] + 1e-6)

# 6. Hydrophobicity balance
df["Hydrophobicity_Balance"] = df["FractionCSP3"] / (df["TopologicalPolarizabilityIndex"] + 1e-6)

# 7. Refractivity efficiency proxy
df["Refractivity_Efficiency"] = df["Polarizability"] / (df["WienerIndex"] + 1e-6)

# 8. Connectivity density
df["Connectivity_Density"] = df["BalabanIndex"] / (df["MolecularVolume"] + 1e-6)

# 9. Topological size ratio
df["Topological_Size_Ratio"] = df["WienerIndex"] / (df["MolecularVolume"] + 1e-6)

# 10. Surface to stiffness ratio
df["Surface_Stiffness_Ratio"] = df["VanDerWaalsSurfaceArea"] / (df["ChainStiffnessIndex"] + 1e-6)

# ----------------------------------------
# CLEANING
# ----------------------------------------
df = df.replace([np.inf, -np.inf], 0)
df = df.fillna(0)

# ----------------------------------------
# SAVE NEW DATASET
# ----------------------------------------
df.to_csv("imidataset_feature_engineered.csv", index=False)

print("✅ Feature extraction completed for descriptor dataset!")
