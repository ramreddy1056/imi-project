import pandas as pd
import numpy as np

# ----------------------------------------
# Load dataset
# ----------------------------------------
df = pd.read_csv("final_selected_features_100rows.csv")

# ----------------------------------------
# INPUT FEATURES (already present)
# ----------------------------------------
input_cols = [
    "Molecular_Volume",
    "Radius_of_Gyration",
    "Surface_Area",
    "Shape_Factor",
    "Molecular_Weight",
    "Repeat_Unit_MW",
    "TPSA",
    "LogP",
    "Molar_Refractivity",
    "Partial_Charges"
]

# ----------------------------------------
# FEATURE EXTRACTION (NEW FEATURES)
# ----------------------------------------

# 1. Volume to surface ratio (compactness)
df["Volume_Surface_Ratio"] = df["Molecular_Volume"] / (df["Surface_Area"] + 1e-6)

# 2. Shape normalized by size
df["Shape_Normalized"] = df["Shape_Factor"] / (df["Molecular_Volume"] + 1e-6)

# 3. Flexibility index (spread vs mass)
df["Flexibility_Index"] = df["Radius_of_Gyration"] / (df["Molecular_Weight"] + 1e-6)

# 4. Density-like feature
df["Mass_Density"] = df["Molecular_Weight"] / (df["Molecular_Volume"] + 1e-6)

# 5. Polarity ratio
df["Polarity_Index"] = df["TPSA"] / (df["Molecular_Weight"] + 1e-6)

# 6. Hydrophobicity balance
df["Hydrophobicity_Balance"] = df["LogP"] / (df["TPSA"] + 1e-6)

# 7. Refractivity efficiency
df["Refractivity_Efficiency"] = df["Molar_Refractivity"] / (df["Molecular_Weight"] + 1e-6)

# 8. Charge density
df["Charge_Density"] = df["Partial_Charges"] / (df["Molecular_Volume"] + 1e-6)

# 9. Size ratio (repeat vs full molecule)
df["Repeat_Unit_Ratio"] = df["Repeat_Unit_MW"] / (df["Molecular_Weight"] + 1e-6)

# 10. Surface to gyration ratio
df["Surface_Gyration_Ratio"] = df["Surface_Area"] / (df["Radius_of_Gyration"] + 1e-6)

# ----------------------------------------
# CLEANING
# ----------------------------------------
df = df.replace([np.inf, -np.inf], 0)
df = df.fillna(0)

# ----------------------------------------
# SAVE NEW DATASET
# ----------------------------------------
df.to_csv("material_dataset_feature_engineered_descriptors.csv", index=False)

print("Feature extraction completed for descriptor dataset!")
