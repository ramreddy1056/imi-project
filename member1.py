from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import pandas as pd

# Load dataset
df = pd.read_csv("smiles.csv")

# Clean column names
df.columns = df.columns.str.strip().str.upper()

# Check SMILES column
if "SMILES" not in df.columns:
    raise ValueError("SMILES column not found.")

data = []
invalid_smiles = []

for idx, smi in enumerate(df["SMILES"]):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        invalid_smiles.append((idx, smi))
        data.append([None]*10)
        continue

    mol_h = Chem.AddHs(mol)  # Explicit hydrogens for NumAtoms and NumBonds

    data.append([
        mol_h.GetNumAtoms(),                                   # NumAtoms (with H)
        mol.GetNumHeavyAtoms(),                                # NumHeavyAtoms
        mol_h.GetNumBonds(),                                   # NumBonds (with H)
        rdMolDescriptors.CalcNumRotatableBonds(mol),           # NumRotatableBonds
        rdMolDescriptors.CalcNumAliphaticRings(mol),           # NumAliphaticRings
        rdMolDescriptors.CalcNumAromaticRings(mol),            # NumAromaticRings
        rdMolDescriptors.CalcNumHeteroatoms(mol),              # NumHeteroatoms
        Descriptors.NumHDonors(mol),                           # HBD
        Descriptors.NumHAcceptors(mol),                        # HBA
        Chem.GetFormalCharge(mol)                              # FormalCharge
    ])

# Report invalid SMILES
if invalid_smiles:
    print(f"Warning: {len(invalid_smiles)} invalid SMILES found:")
    for idx, smi in invalid_smiles:
        print(f"  Row {idx}: {smi}")
else:
    print("All SMILES parsed successfully.")

# Column names
cols = [
    "NumAtoms", "NumHeavyAtoms", "NumBonds", "NumRotatableBonds",
    "NumAliphaticRings", "NumAromaticRings", "NumHeteroatoms",
    "HBD", "HBA", "FormalCharge"
]

# Validate row count
assert len(data) == len(df), f"Row mismatch: {len(data)} vs {len(df)}"

# Combine and save
df_out = pd.concat([df, pd.DataFrame(data, columns=cols)], axis=1)
df_out.to_csv("member1.csv", index=False)
print(f"Done. Output saved: {df_out.shape[0]} rows, {df_out.shape[1]} columns.")
