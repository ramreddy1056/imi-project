import re
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

# =========================================
# ALL RDKit DESCRIPTORS
# =========================================
DESC_FN   = {name: fn for name, fn in Descriptors.descList}
ALL_DESCS = list(DESC_FN.keys())

# Column names for new fingerprint/feature blocks
MACCS_COLS   = [f"MACCS_{i}"   for i in range(167)]
MORGAN_COLS  = [f"Morgan_{i}"  for i in range(256)]
POLYMER_COLS = [
    "RotBondFrac",
    "RingAtomFrac",
    "AromaticFrac",
    "HBD_per_atom",
    "HBA_per_atom",
    "StereoFrac",
    "HeavyAtomMW",
    "BranchingIndex",
]

ALL_FEATURE_COLS = ALL_DESCS + MACCS_COLS + MORGAN_COLS + POLYMER_COLS

print(f"RDKit descriptors : {len(ALL_DESCS)}")
print(f"MACCS keys        : {len(MACCS_COLS)}")
print(f"Morgan bits       : {len(MORGAN_COLS)}")
print(f"Polymer features  : {len(POLYMER_COLS)}")
print(f"Total features    : {len(ALL_FEATURE_COLS)}")


# =========================================
# ROBUST POLYMER SMILES PARSER
# =========================================
def sanitize_smiles(smi: str):
    """
    Parse polymer pSMILES into an RDKit mol.

    Handles:
      1. [*] attachment points  -> replaced with [H] to cap open ends
      2. Explicit (H) branches on Si/Ge/Sn -> stripped before parsing
      3. Valence errors on exotic atoms -> retry with sanitize=False
      4. Truncated/malformed SMILES -> returns None
    """
    if not isinstance(smi, str) or not smi.strip():
        return None

    # Strip explicit (H) branches that confuse bracket-atom parsing
    cleaned = re.sub(r'\(H\)', '', smi)

    # Cap [*] attachment points with hydrogen
    capped = cleaned.replace('[*]', '[H]')

    # Attempt 1: Standard parse
    mol = Chem.MolFromSmiles(capped)
    if mol is not None:
        return mol

    # Attempt 2: Skip valence check for exotic atoms (Si, Ge, Sn, F, O errors)
    try:
        mol = Chem.MolFromSmiles(capped, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(
            mol,
            int(Chem.SanitizeFlags.SANITIZE_ALL)
            ^ int(Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        )
        return mol
    except Exception:
        return None


# =========================================
# POLYMER-SPECIFIC FEATURES
# =========================================
def compute_polymer_features(mol) -> dict:
    """
    8 hand-crafted features that capture polymer chain behaviour.

    - RotBondFrac   : rotatable bonds / heavy atoms  -> chain flexibility
    - RingAtomFrac  : atoms in any ring / heavy atoms -> rigidity
    - AromaticFrac  : aromatic atoms / heavy atoms   -> pi-stacking tendency
    - HBD_per_atom  : H-bond donors normalised       -> intermolecular forces
    - HBA_per_atom  : H-bond acceptors normalised    -> intermolecular forces
    - StereoFrac    : stereocentres / heavy atoms    -> tacticity proxy
    - HeavyAtomMW   : MW / heavy atom count          -> avg atom mass (backbone density)
    - BranchingIndex: num branches / heavy atoms     -> side-chain density
    """
    n = mol.GetNumHeavyAtoms()
    if n == 0:
        return {col: np.nan for col in POLYMER_COLS}

    rot_bonds  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ring_atoms = sum(1 for a in mol.GetAtoms() if a.IsInRing())
    arom_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    hbd        = rdMolDescriptors.CalcNumHBD(mol)
    hba        = rdMolDescriptors.CalcNumHBA(mol)
    stereo     = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

    # MW per heavy atom — proxy for backbone atom size
    mw         = Descriptors.MolWt(mol)

    # Branching index: degree-3+ atoms (branch points) normalised by size
    branch_pts = sum(1 for a in mol.GetAtoms() if a.GetDegree() >= 3)

    return {
        "RotBondFrac"   : rot_bonds  / n,
        "RingAtomFrac"  : ring_atoms / n,
        "AromaticFrac"  : arom_atoms / n,
        "HBD_per_atom"  : hbd        / n,
        "HBA_per_atom"  : hba        / n,
        "StereoFrac"    : stereo     / n,
        "HeavyAtomMW"   : mw         / n,
        "BranchingIndex": branch_pts / n,
    }


# =========================================
# FULL FEATURE EXTRACTION
# =========================================
def extract_features(smiles: str):
    """
    Compute all features for one SMILES string:
      - All RDKit 2D descriptors
      - 167 MACCS structural keys
      - 256-bit Morgan fingerprint (radius 2)
      - 8 polymer-specific features

    Returns a flat dict of {feature_name: float} or None if parsing fails.
    """
    mol = sanitize_smiles(smiles)
    if mol is None:
        return None

    feats = {}

    # --- RDKit 2D descriptors ---
    for name, fn in Descriptors.descList:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val = fn(mol)
            feats[name] = float(val) if val is not None else np.nan
        except Exception:
            feats[name] = np.nan

    # --- MACCS keys (167 bits) ---
    maccs = list(MACCSkeys.GenMACCSKeys(mol))
    for i, bit in enumerate(maccs):
        feats[f"MACCS_{i}"] = float(bit)

    # --- Morgan fingerprint radius=2, 256 bits ---
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
    morgan = list(morgan_gen.GetFingerprint(mol))
    for i, bit in enumerate(morgan):
        feats[f"Morgan_{i}"] = float(bit)

    # --- Polymer-specific features ---
    feats.update(compute_polymer_features(mol))

    return feats


# =========================================
# LOAD & EXTRACT
# =========================================
df = pd.read_csv("Train.csv")
print(f"\nLoaded {len(df)} rows from Train.csv")

smi_col = "SMILES" if "SMILES" in df.columns else "PSMILES"
tg_col  = "Tg_K"   if "Tg_K"   in df.columns else [c for c in df.columns if "Tg" in c][0]
df      = df.dropna(subset=[smi_col])

rows, smiles_list, tg_values, skipped = [], [], [], []
total = len(df)

for i, (_, row) in enumerate(df.iterrows()):
    if i % 500 == 0:
        print(f"  Processing {i}/{total}...")

    smi = row[smi_col]
    if not isinstance(smi, str):
        skipped.append((i, str(smi), "not a string"))
        continue

    feats = extract_features(smi)
    if feats is None:
        skipped.append((i, smi, "parse failed"))
        continue

    rows.append(feats)
    smiles_list.append(smi)
    tg_values.append(row[tg_col])

print(f"\nValid: {len(rows)} | Skipped: {len(skipped)}")

if skipped:
    print("\nSkipped SMILES (review in Train.csv):")
    for idx, smi, reason in skipped:
        print(f"  Row {idx:>5} | {reason:<15} | {str(smi)[:60]}")

# =========================================
# SAVE
# =========================================
result = pd.DataFrame(rows, columns=ALL_FEATURE_COLS)
result.insert(0, "Tg_K",   tg_values)
result.insert(0, "SMILES", smiles_list)

result.to_csv("Trainfeature.csv", index=False)
print(f"\nSaved: Trainfeature.csv — {result.shape[0]} rows x {result.shape[1]} cols")
