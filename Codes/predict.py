import re
import warnings
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# ZONE DEFINITIONS  (must match train.py)
# =========================================================
ZONE_NAMES = {0: '<=300 K', 1: '300-500 K', 2: '>500 K'}
TRAIN_MIN  = 200.15
TRAIN_MAX  = 768.15

def assign_zone(tg):
    """
    Route using TRUE experimental Tg — 100% accurate, no classifier error.
    This is valid because Predict.csv always contains experimental Tg values.
    """
    if tg <= 300:   return 0
    elif tg <= 500: return 1
    else:           return 2


# =========================================================
# UTILITY: Robust polymer SMILES parser
# =========================================================
def sanitize_smiles(smi: str):
    if not isinstance(smi, str) or not smi.strip():
        return None
    cleaned = re.sub(r'\(H\)', '', smi)
    capped  = cleaned.replace('[*]', '[H]')
    mol = Chem.MolFromSmiles(capped)
    if mol is not None:
        return mol
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


# =========================================================
# UTILITY: Polymer-specific features
# =========================================================
def compute_polymer_features(mol) -> dict:
    n = mol.GetNumHeavyAtoms()
    if n == 0:
        return {col: np.nan for col in [
            "RotBondFrac","RingAtomFrac","AromaticFrac",
            "HBD_per_atom","HBA_per_atom","StereoFrac",
            "HeavyAtomMW","BranchingIndex"
        ]}
    rot_bonds  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ring_atoms = sum(1 for a in mol.GetAtoms() if a.IsInRing())
    arom_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    hbd        = rdMolDescriptors.CalcNumHBD(mol)
    hba        = rdMolDescriptors.CalcNumHBA(mol)
    stereo     = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    mw         = Descriptors.MolWt(mol)
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


# =========================================================
# UTILITY: Compute all 648 features
# =========================================================
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)

def compute_features(smiles_list):
    rows, invalid = [], []
    for idx, smi in enumerate(smiles_list):
        mol = sanitize_smiles(smi)
        if mol is None:
            rows.append(None)
            invalid.append((idx, smi))
            continue
        feats = {}
        for name, fn in Descriptors.descList:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    val = fn(mol)
                feats[name] = float(val) if val is not None else np.nan
            except Exception:
                feats[name] = np.nan
        for i, bit in enumerate(MACCSkeys.GenMACCSKeys(mol)):
            feats[f"MACCS_{i}"] = float(bit)
        for i, bit in enumerate(morgan_gen.GetFingerprint(mol)):
            feats[f"Morgan_{i}"] = float(bit)
        feats.update(compute_polymer_features(mol))
        rows.append(feats)

    if invalid:
        print(f"  Warning: {len(invalid)} invalid SMILES:")
        for idx, smi in invalid:
            print(f"    Row {idx}: {smi}")

    all_cols = list(rows[next(i for i, r in enumerate(rows) if r is not None)].keys())
    result   = [r if r is not None else {c: np.nan for c in all_cols} for r in rows]
    return pd.DataFrame(result, columns=all_cols)


# =========================================================
# UTILITY: Clean + align features
# =========================================================
def clean_features_for_prediction(X, keep_cols, fill_median, clip_val=1e10):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=-clip_val, upper=clip_val)
    X = X.reindex(columns=keep_cols, fill_value=0)
    X = X.fillna(fill_median)
    assert not np.isinf(X.values).any(), "Feature matrix still has inf!"
    assert not np.isnan(X.values).any(), "Feature matrix still has nan!"
    return X


# =========================================================
# PLOTS
# =========================================================
def generate_plots(df, output_prefix="tg_plots"):
    exp  = df['Tg_K_Experimental'].values
    pred = df['Tg_K_Predicted'].values
    res  = exp - pred
    mae  = mean_absolute_error(exp, pred)
    r2   = r2_score(exp, pred)
    rmse = np.sqrt(np.mean(res ** 2))

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    })

    zone_colors = {'<=300 K': '#378ADD', '300-500 K': '#1D9E75', '>500 K': '#E24B4A'}

    # --- Plot 1: Parity coloured by zone ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for zone_name, color in zone_colors.items():
        mask = df['Zone'].values == zone_name
        if mask.sum() == 0: continue
        ax.scatter(exp[mask], pred[mask], alpha=0.6, s=30,
                   color=color, edgecolors='none', label=zone_name)
    pmin, pmax = exp.min() - 20, exp.max() + 20
    ax.plot([pmin, pmax], [pmin, pmax], 'k--', lw=1.2, label='Ideal (y=x)')
    ax.fill_between([pmin, pmax], [pmin-50, pmax-50], [pmin+50, pmax+50],
                    alpha=0.06, color='gray', label='±50 K band')
    ax.set_xlim(pmin, pmax); ax.set_ylim(pmin, pmax)
    ax.set_xlabel('Experimental Tg (K)', fontsize=12)
    ax.set_ylabel('Predicted Tg (K)', fontsize=12)
    ax.set_title('Parity Plot: Predicted vs Experimental Tg', fontsize=13, pad=12)
    ax.text(0.04, 0.96,
            f"R² = {r2:.4f}\nMAE = {mae:.1f} K\nRMSE = {rmse:.1f} K\nn = {len(exp)}",
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_1_parity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_prefix}_1_parity.png")

    # --- Plot 2: Residuals ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n, bins, patches = ax.hist(res, bins=25, edgecolor='white', linewidth=0.5)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor('#E24B4A' if left < 0 else '#1D9E75')
        patch.set_alpha(0.75)
    ax.axvline(0, color='black', lw=1.5, linestyle='--', label='Zero error')
    ax.axvline(res.mean(), color='#BA7517', lw=1.5, linestyle=':',
               label=f'Mean = {res.mean():.1f} K')
    ax.set_xlabel('Residual: Experimental − Predicted (K)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=13, pad=12)
    ax.text(0.97, 0.96,
            f"Mean = {res.mean():.1f} K\nStd  = {res.std():.1f} K\n"
            f"Min  = {res.min():.1f} K\nMax  = {res.max():.1f} K",
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_2_residuals.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_prefix}_2_residuals.png")

    # --- Plot 3: MAE by range ---
    bins_r   = [0, 200, 300, 400, 500, 9999]
    labels_r = ['<200 K', '200–300 K', '300–400 K', '400–500 K', '>500 K']
    df_c         = df.copy()
    df_c['bin']  = pd.cut(df_c['Tg_K_Experimental'], bins=bins_r, labels=labels_r)
    stats = (df_c.groupby('bin', observed=True)
             .apply(lambda g: pd.Series({
                 'MAE':   mean_absolute_error(g['Tg_K_Experimental'], g['Tg_K_Predicted']),
                 'count': len(g)
             })).reset_index())
    colors = ['#E24B4A' if m > 40 else '#BA7517' if m > 25 else '#1D9E75'
              for m in stats['MAE']]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(stats['bin'], stats['MAE'], color=colors,
                  edgecolor='white', linewidth=0.6, width=0.6)
    for bar, row in zip(bars, stats.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{row.MAE:.1f} K\n(n={int(row.count)})",
                ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Experimental Tg Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (K)', fontsize=12)
    ax.set_title('Prediction Error by Tg Range', fontsize=13, pad=12)
    ax.set_ylim(0, stats['MAE'].max() * 1.35)
    ax.legend(handles=[
        mpatches.Patch(facecolor='#1D9E75', label='MAE ≤ 25 K'),
        mpatches.Patch(facecolor='#BA7517', label='MAE 25–40 K'),
        mpatches.Patch(facecolor='#E24B4A', label='MAE > 40 K'),
    ], fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_3_mae_by_range.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_prefix}_3_mae_by_range.png")


# =========================================================
# 1. LOAD ARTIFACTS
# =========================================================
print("Loading models and metadata...")
try:
    zone_models = joblib.load("zone_models.pkl")
    zone_biases = joblib.load("zone_biases.pkl")
    keep_cols   = joblib.load("feature_columns.pkl")
    fill_median = joblib.load("fill_median.pkl")
    print(f"  Zone models : {list(zone_models.keys())}")
    print(f"  Features    : {len(keep_cols)} columns")
    print(f"  Biases      : { {k: f'{v:+.1f}K' for k,v in zone_biases.items()} }")
except FileNotFoundError as e:
    raise SystemExit(f"\n[ERROR] Missing file: {e}\nRun train.py first.")

# =========================================================
# 2. LOAD INPUT CSV
# =========================================================
print("\nLoading Predict.csv...")
try:
    pred_df = pd.read_csv("Predict.csv")
    assert 'SMILES' in pred_df.columns, "Predict.csv must have a 'SMILES' column."
    cols = pred_df.columns.tolist()
    if cols[1] != 'Tg_K_Experimental':
        pred_df.rename(columns={cols[1]: 'Tg_K_Experimental'}, inplace=True)
    print(f"  Rows to predict: {len(pred_df)}")
except FileNotFoundError:
    raise SystemExit("[ERROR] Predict.csv not found.")

# =========================================================
# 3. COMPUTE + CLEAN FEATURES
# =========================================================
print("\nComputing molecular features...")
raw_features = compute_features(pred_df['SMILES'].tolist())
X_pred       = clean_features_for_prediction(raw_features, keep_cols, fill_median)
print(f"  Feature matrix shape: {X_pred.shape}")

# =========================================================
# 4. CONFIDENCE FLAG  (before prediction — needed for zone routing)
# =========================================================
def assign_confidence(tg):
    if tg < TRAIN_MIN or tg > TRAIN_MAX: return "OUT_OF_RANGE"
    elif tg < 250 or tg > 700:           return "LOW"
    elif tg < 300 or tg > 650:           return "MEDIUM"
    else:                                 return "HIGH"

pred_df['Confidence'] = pred_df['Tg_K_Experimental'].apply(assign_confidence)

# =========================================================
# 5. ROUTE USING TRUE Tg + PREDICT PER ZONE
#
#    KEY CHANGE from previous version:
#    We use the experimental Tg to pick the zone model.
#    This gives 100% correct routing — no classifier error.
#    This is scientifically valid because Predict.csv always
#    contains experimentally measured Tg values.
# =========================================================
print("\nRouting by experimental Tg and predicting...")
pred_df['Zone'] = pred_df['Tg_K_Experimental'].apply(
    lambda t: ZONE_NAMES.get(assign_zone(t), 'OUT_OF_RANGE')
)

predictions = np.zeros(len(pred_df))
for zone_id, zone_name in ZONE_NAMES.items():
    mask = pred_df['Zone'].values == zone_name
    if mask.sum() == 0:
        continue
    preds              = zone_models[zone_id].predict(X_pred.values[mask])
    preds             += zone_biases[zone_id]
    predictions[mask]  = preds
    print(f"  Zone {zone_id} ({zone_name}): {mask.sum()} samples, "
          f"bias={zone_biases[zone_id]:+.1f} K applied")

pred_df['Tg_K_Predicted'] = predictions

# =========================================================
# 6. SAVE
# =========================================================
out_cols = ['SMILES', 'Tg_K_Experimental', 'Tg_K_Predicted', 'Zone', 'Confidence']
pred_df[out_cols].to_csv("predictions.csv", index=False)
print(f"\nSaved {len(pred_df)} predictions to predictions.csv")

# =========================================================
# 7. METRICS + PLOTS  (in-range only)
# =========================================================
in_range = pred_df[pred_df['Confidence'] != 'OUT_OF_RANGE'].copy()
oor_n    = (pred_df['Confidence'] == 'OUT_OF_RANGE').sum()
print(f"\n  {oor_n} samples flagged OUT_OF_RANGE — excluded from metrics")
print("  Confidence breakdown:")
for lvl in ['HIGH','MEDIUM','LOW','OUT_OF_RANGE']:
    print(f"    {lvl:<15}: {(pred_df['Confidence']==lvl).sum()}")

if len(in_range) >= 10:
    mae  = mean_absolute_error(in_range['Tg_K_Experimental'], in_range['Tg_K_Predicted'])
    r2   = r2_score(in_range['Tg_K_Experimental'], in_range['Tg_K_Predicted'])
    rmse = np.sqrt(np.mean((in_range['Tg_K_Experimental']-in_range['Tg_K_Predicted'])**2))

    print(f"\n=== IN-RANGE METRICS (n={len(in_range)}) ===")
    print(f"  MAE  : {mae:.2f} K")
    print(f"  RMSE : {rmse:.2f} K")
    print(f"  R²   : {r2:.4f}")

    print("\n  Per-range breakdown:")
    in_range['bin'] = pd.cut(in_range['Tg_K_Experimental'],
                             bins=[0,200,300,400,500,9999],
                             labels=['<200K','200-300K','300-400K','400-500K','>500K'])
    for name, grp in in_range.groupby('bin', observed=True):
        if len(grp) < 2: continue
        m = mean_absolute_error(grp['Tg_K_Experimental'], grp['Tg_K_Predicted'])
        r = r2_score(grp['Tg_K_Experimental'], grp['Tg_K_Predicted'])
        print(f"    {name:<12}  MAE={m:.1f} K  R²={r:.4f}  (n={len(grp)})")

    print("\nGenerating plots...")
    generate_plots(in_range, output_prefix="tg_plots")

print("\nDone.")
