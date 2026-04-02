import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

# ----------------------------------------
# Load your dataset
# ----------------------------------------
df = pd.read_csv("aromatic_200_fixed.csv")  # change path if needed

# ----------------------------------------
# Feature extraction function
# ----------------------------------------
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return [None]*30
    
    # Add hydrogens and generate 3D coordinates
    molH = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(molH, randomSeed=42)
        AllChem.UFFOptimizeMolecule(molH)
    except:
        pass

    # Partial charges
    try:
        AllChem.ComputeGasteigerCharges(molH)
        charges = [float(atom.GetProp('_GasteigerCharge')) for atom in molH.GetAtoms()]
        max_charge = max(charges)
        min_charge = min(charges)
        avg_charge = sum(charges)/len(charges)
    except:
        max_charge = min_charge = avg_charge = 0

    # 3D descriptors (approx)
    try:
        rg = rdMolDescriptors.CalcRadiusOfGyration(molH)
    except:
        rg = 0

    # Approx volume & surface (proxy)
    volume = Descriptors.MolMR(mol) * 1.5
    surface_area = Descriptors.TPSA(mol) * 1.2

    # Shape factor (approx)
    shape_factor = rg / (Descriptors.MolWt(mol) + 1e-6)

    # Density (approx)
    density = Descriptors.MolWt(mol) / (volume + 1e-6)

    return [
        # 1–10
        mol.GetNumAtoms(),
        mol.GetNumHeavyAtoms(),
        mol.GetNumBonds(),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Chem.GetFormalCharge(mol),

        # 11–20
        Descriptors.MolWt(mol),
        volume,
        surface_area,
        rg,
        shape_factor,
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.MolMR(mol),
        Descriptors.FractionCSP3(mol),
        density,

        # 21–30
        max_charge,
        min_charge,
        avg_charge,
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.BertzCT(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.Kappa3(mol)
    ]

# ----------------------------------------
# Apply feature extraction
# ----------------------------------------
feature_names = [
    "NumAtoms","NumHeavyAtoms","NumBonds","NumRotatableBonds",
    "NumAromaticRings","NumAliphaticRings","NumHeteroatoms",
    "NumHDonors","NumHAcceptors","FormalCharge",
    "MolecularWeight","MolecularVolume","SurfaceArea","RadiusOfGyration",
    "ShapeFactor","TPSA","LogP","MolarRefractivity","FractionCSP3","Density",
    "MaxPartialCharge","MinPartialCharge","AvgPartialCharge",
    "NumRings","NumAromaticCarbocycles","NumSaturatedCarbocycles",
    "BalabanJ","BertzCT","HallKierAlpha","Kappa3"
]

features = df["SMILES"].apply(extract_features)
features_df = pd.DataFrame(features.tolist(), columns=feature_names)

# Combine with original data
final_df = pd.concat([df, features_df], axis=1)

# Save dataset
final_df.to_csv("aromatic_200_with_30_features.csv", index=False)

print("✅ Dataset with 30 features saved successfully!")
