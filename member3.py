from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import pandas as pd

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("rdkit_100_unique_smiles.csv")   # change if needed

# Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip().str.upper()

# Check columns
print("Columns in dataset:", df.columns)

# Ensure SMILES exists
if "SMILES" not in df.columns:
    raise ValueError("SMILES column not found in dataset")

# -------------------------------
# Feature extraction
# -------------------------------
data = []

for smi in df["SMILES"]:
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        data.append([None]*11)
        continue

    # Base descriptors
    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    mol_mr = Descriptors.MolMR(mol)

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    ring_count = rdMolDescriptors.CalcNumRings(mol)

    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)

    # Custom feat3 descriptors
    data.append([
        mol_wt / (num_atoms + 1),                     # ElasticModulus
        mol_wt / (tpsa + 1),                          # Volume_Surface_Ratio
        num_atoms / (ring_count + 1),                 # Shape_Normalized
        rot_bonds / (num_bonds + 1),                  # Flexibility_Index
        mol_wt / (num_atoms + 1),                     # Mass_Density
        tpsa / (num_atoms + 1),                       # Polarity_Index
        logp / (h_donors + h_acceptors + 1),          # Hydrophobicity_Balance
        mol_mr / (mol_wt + 1),                        # Refractivity_Efficiency
        num_bonds / (num_atoms + 1),                  # Connectivity_Density
        ring_count / (num_atoms + 1),                 # Topological_Size_Ratio
        tpsa / (rot_bonds + 1)                        # Surface_Stiffness_Ratio
    ])

# -------------------------------
# Column names (feat3)
# -------------------------------
cols = [
    "ElasticModulus",
    "Volume_Surface_Ratio",
    "Shape_Normalized",
    "Flexibility_Index",
    "Mass_Density",
    "Polarity_Index",
    "Hydrophobicity_Balance",
    "Refractivity_Efficiency",
    "Connectivity_Density",
    "Topological_Size_Ratio",
    "Surface_Stiffness_Ratio"
]

# -------------------------------
# Combine and save
# -------------------------------
df_out = pd.concat([df, pd.DataFrame(data, columns=cols)], axis=1)

df_out.to_excel("feat3_rdkit_output.xlsx", index=False)

print("Done! File saved as feat3_rdkit_output.xlsx")
