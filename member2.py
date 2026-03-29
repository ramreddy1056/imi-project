from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("rdkit_100_unique_smiles.csv")

data = []

for smi in df["smiles"]:
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        data.append([None]*10)
        continue

    # Add hydrogens for better calculations
    mol = Chem.AddHs(mol)

    # -------------------------
    # Feature calculations
    # -------------------------

    mol_wt = Descriptors.MolWt(mol)
    molar_ref = Descriptors.MolMR(mol)
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)

    # Approximations (simple like your style)
    mol_volume = molar_ref
    surface_area = tpsa
    radius_gyration = np.sqrt(molar_ref) if molar_ref > 0 else 0
    shape_factor = mol_volume / (surface_area + 1e-6)
    repeat_mw = mol_wt

    # Partial charges
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
        partial_charge = sum(charges) / len(charges)
    except:
        partial_charge = 0

    data.append([
        mol_volume,
        radius_gyration,
        surface_area,
        shape_factor,
        mol_wt,
        repeat_mw,
        tpsa,
        logp,
        molar_ref,
        partial_charge
    ])

# Column names (your descriptors)
cols = [
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

# Combine
df_out = pd.concat([df, pd.DataFrame(data, columns=cols)], axis=1)

# Save
df_out.to_excel("material_member2.xlsx", index=False)

print("Member Done")
