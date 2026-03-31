from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import pandas as pd

# Load dataset
df = pd.read_csv("rdkit_100_unique_smiles.csv")

# Clean column names
df.columns = df.columns.str.strip().str.upper()

# Check SMILES column
if "SMILES" not in df.columns:
    raise ValueError("SMILES column not found ❌")

data = []

for smi in df["SMILES"]:
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        data.append([None]*10)
        continue

    data.append([
        mol.GetNumAtoms(),                                # NumAtoms
        mol.GetNumHeavyAtoms(),                           # NumHeavyAtoms
        mol.GetNumBonds(),                                # NumBonds
        Descriptors.NumRotatableBonds(mol),               # NumRotatableBonds

        rdMolDescriptors.CalcNumAliphaticRings(mol),      # NumAliphaticRings
        rdMolDescriptors.CalcNumAromaticRings(mol),       # NumAromaticRings
        Descriptors.NumHeteroatoms(mol),                  # NumHeteroatoms

        Descriptors.NumHDonors(mol),                      # HBD
        Descriptors.NumHAcceptors(mol),                   # HBA
        Chem.GetFormalCharge(mol)                         # FormalCharge
    ])

# Column names
cols = [
    "NumAtoms", "NumHeavyAtoms", "NumBonds", "NumRotatableBonds",
    "NumAliphaticRings", "NumAromaticRings", "NumHeteroatoms",
    "HBD", "HBA", "FormalCharge"
]

# Combine
df_out = pd.concat([df, pd.DataFrame(data, columns=cols)], axis=1)

# Save (CSV to avoid openpyxl issue)
df_out.to_csv("rdkit_basic_descriptors.csv", index=False)

print("RDKit basic descriptors generated")
