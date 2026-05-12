import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import compute_sample_weight


# =========================================================
# ZONE DEFINITIONS
# =========================================================
ZONE_NAMES  = {0: '<=300 K', 1: '300-500 K', 2: '>500 K'}
ZONE_BOUNDS = {0: (0, 300), 1: (300, 500), 2: (500, 9999)}

def assign_zone(tg):
    if tg <= 300:   return 0
    elif tg <= 500: return 1
    else:           return 2


# =========================================================
# UTILITY: Clean features
# =========================================================
def clean_features(X, keep_cols=None, fill_median=None, clip_val=1e10):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=-clip_val, upper=clip_val)
    if keep_cols is None:
        X = X.dropna(axis=1, thresh=int(0.5 * len(X)))
        keep_cols   = X.columns.tolist()
        fill_median = X.median(numeric_only=True)
    else:
        X = X.reindex(columns=keep_cols, fill_value=0)
    X = X.fillna(fill_median)
    assert not np.isinf(X.values).any(), "Still has inf!"
    assert not np.isnan(X.values).any(), "Still has nan!"
    return X, keep_cols, fill_median


# =========================================================
# UTILITY: Train + cross-validate one zone model
#          Returns model, cv_mae, train_loss, val_loss
# =========================================================
def train_zone_model(X_zone, y_zone, zone_id, n_cv=5):
    zone_params = {
        0: dict(learning_rate=0.02, max_depth=4, subsample=0.8,
                colsample_bytree=0.6, reg_lambda=0.3, reg_alpha=0.1,
                min_child_weight=2, gamma=0.1),
        1: dict(learning_rate=0.03, max_depth=6, subsample=0.8,
                colsample_bytree=0.7, reg_lambda=0.5, reg_alpha=0.05,
                min_child_weight=3, gamma=0.0),
        2: dict(learning_rate=0.03, max_depth=6, subsample=0.8,
                colsample_bytree=0.7, reg_lambda=0.3, reg_alpha=0.05,
                min_child_weight=2, gamma=0.0),
    }
    params = zone_params[zone_id]
    kf     = KFold(n_splits=n_cv, shuffle=True, random_state=42)
    cv_maes        = []
    best_n         = 1000
    last_train_loss = None
    last_val_loss   = None

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_zone)):
        X_tr, X_va = X_zone[tr_idx], X_zone[va_idx]
        y_tr, y_va = y_zone[tr_idx], y_zone[va_idx]

        m = XGBRegressor(
            n_estimators=2000,
            early_stopping_rounds=50,
            eval_metric='mae',
            random_state=42,
            n_jobs=-1,
            **params
        )
        # Pass both train and val so we capture train loss curve too
        m.fit(X_tr, y_tr,
              eval_set=[(X_tr, y_tr), (X_va, y_va)],
              verbose=False)

        best_n = min(best_n, m.best_iteration + 1)
        cv_maes.append(m.best_score)

        # Keep last fold's curves for plotting
        last_train_loss = m.evals_result()['validation_0']['mae']
        last_val_loss   = m.evals_result()['validation_1']['mae']

    cv_mae = np.mean(cv_maes)
    print(f"    CV MAE: {cv_mae:.1f} K  |  Best n_estimators: {best_n}")

    # Retrain on full zone data
    final_model = XGBRegressor(
        n_estimators=best_n,
        random_state=42,
        n_jobs=-1,
        **params
    )
    final_model.fit(X_zone, y_zone)
    return final_model, cv_mae, last_train_loss, last_val_loss


# =========================================================
# PLOT 1: Training vs Validation Loss (all 3 zones)
# =========================================================
def plot_loss_curves(loss_histories, output_prefix="tg_plots"):
    """
    loss_histories: dict {zone_id: {'train': [...], 'val': [...], 'name': str}}
    """
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    })

    zone_colors = {0: '#378ADD', 1: '#1D9E75', 2: '#E24B4A'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    fig.suptitle('Training vs Validation Loss per Zone (MAE, K)', fontsize=13, y=1.02)

    for zone_id, hist in loss_histories.items():
        ax         = axes[zone_id]
        train_loss = hist['train']
        val_loss   = hist['val']
        zone_name  = hist['name']
        color      = zone_colors[zone_id]
        epochs     = range(1, len(train_loss) + 1)

        ax.plot(epochs, train_loss, color=color,    lw=1.5, label='Train MAE')
        ax.plot(epochs, val_loss,   color=color,    lw=1.5, linestyle='--',
                alpha=0.7, label='Val MAE')

        # Mark best val point
        best_ep  = int(np.argmin(val_loss)) + 1
        best_val = min(val_loss)
        ax.axvline(best_ep, color='gray', lw=0.8, linestyle=':')
        ax.scatter([best_ep], [best_val], color=color, s=50, zorder=5)
        ax.text(best_ep + len(epochs) * 0.02, best_val,
                f"Best: {best_val:.1f} K\n@ ep {best_ep}",
                fontsize=8, color=color, va='center')

        ax.set_title(f'Zone {zone_id}: {zone_name}', fontsize=11)
        ax.set_xlabel('Boosting Round', fontsize=10)
        ax.set_ylabel('MAE (K)', fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{output_prefix}_4_loss_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =========================================================
# PLOT 2: Residual plot (predicted vs residual, per zone)
# =========================================================
def plot_residuals(y_test, all_preds, z_test, output_prefix="tg_plots"):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    })

    residuals   = y_test - all_preds
    zone_colors = {0: '#378ADD', 1: '#1D9E75', 2: '#E24B4A'}

    fig, ax = plt.subplots(figsize=(7, 5))

    for z, name in ZONE_NAMES.items():
        mask = z_test == z
        if mask.sum() == 0: continue
        ax.scatter(all_preds[mask], residuals[mask],
                   alpha=0.5, s=25, color=zone_colors[z],
                   edgecolors='none', label=name)

    ax.axhline(0, color='black', lw=1.5, linestyle='--', label='Zero residual')
    ax.axhline(residuals.mean(), color='#BA7517', lw=1.2, linestyle=':',
               label=f'Mean = {residuals.mean():.1f} K')

    # ±1 std band
    std = residuals.std()
    ax.fill_between([all_preds.min()-10, all_preds.max()+10],
                    [-std, -std], [std, std],
                    alpha=0.07, color='gray', label=f'±1σ ({std:.1f} K)')

    ax.set_xlabel('Predicted Tg (K)', fontsize=12)
    ax.set_ylabel('Residual: Experimental − Predicted (K)', fontsize=12)
    ax.set_title('Residual Plot: Predicted vs Error', fontsize=13, pad=12)

    mae  = mean_absolute_error(y_test, all_preds)
    rmse = np.sqrt(np.mean(residuals ** 2))
    ax.text(0.03, 0.97,
            f"MAE  = {mae:.1f} K\nRMSE = {rmse:.1f} K\nBias = {residuals.mean():.1f} K\nn = {len(y_test)}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    path = f"{output_prefix}_5_residual_plot.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =========================================================
# 1. LOAD + CLEAN
# =========================================================
print("Loading training data...")
train_df = pd.read_csv("Trainfeature.csv")
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  Shape: {train_df.shape}")

X_raw = train_df.drop(columns=['SMILES', 'Tg_K'])
y     = train_df['Tg_K'].values

X, keep_cols, fill_median = clean_features(X_raw)
print(f"  Clean feature shape: {X.shape}")

# =========================================================
# 2. ASSIGN ZONES
# =========================================================
zones = np.array([assign_zone(t) for t in y])
print("\n  Zone distribution (training):")
for z, name in ZONE_NAMES.items():
    print(f"    Zone {z} ({name}): {(zones == z).sum()} samples")

# =========================================================
# 3. STRATIFIED TRAIN / TEST SPLIT
# =========================================================
X_arr = X.values

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
    X_arr, y, zones,
    test_size=0.2,
    random_state=42,
    stratify=zones
)
print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

# =========================================================
# 4. TRAIN ZONE ROUTER
# =========================================================
print("\n--- Training zone router (classifier) ---")
router_weights = compute_sample_weight('balanced', z_train)
router = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=0.5,
    random_state=42, n_jobs=-1, eval_metric='mlogloss'
)
router.fit(X_train, z_train, sample_weight=router_weights)
router_preds = router.predict(X_test)
router_acc   = np.mean(router_preds == z_test)
print(f"  Router accuracy: {router_acc:.4f}")
for z, name in ZONE_NAMES.items():
    mask = z_test == z
    if mask.sum() == 0: continue
    acc = np.mean(router_preds[mask] == z_test[mask])
    print(f"    Zone {z} ({name}): {acc:.4f} accuracy (n={mask.sum()})")

# =========================================================
# 5. TRAIN ONE XGBoost PER ZONE
# =========================================================
print("\n--- Training zone-specific regressors ---")

zone_models   = {}
zone_biases   = {}
loss_histories = {}

for zone_id, zone_name in ZONE_NAMES.items():
    mask_tr = z_train == zone_id
    mask_te = z_test  == zone_id

    X_z = X_train[mask_tr]
    y_z = y_train[mask_tr]

    print(f"\n  Zone {zone_id} ({zone_name}): {mask_tr.sum()} train / {mask_te.sum()} test")

    model, cv_mae, train_loss, val_loss = train_zone_model(X_z, y_z, zone_id)

    # Save loss history for plotting
    loss_histories[zone_id] = {
        'train': train_loss,
        'val':   val_loss,
        'name':  zone_name
    }

    if mask_te.sum() > 1:
        preds     = model.predict(X_test[mask_te])
        residuals = y_test[mask_te] - preds
        bias      = float(np.mean(residuals))
        mae       = mean_absolute_error(y_test[mask_te], preds)
        r2        = r2_score(y_test[mask_te], preds)
        print(f"    Test  MAE: {mae:.1f} K  R²: {r2:.4f}  Bias: {bias:+.1f} K")
    else:
        bias = 0.0

    zone_models[zone_id] = model
    zone_biases[zone_id] = bias

# =========================================================
# 6. OVERALL TEST METRICS
# =========================================================
print("\n=== OVERALL TEST RESULTS (true zone routing) ===")
all_preds = np.zeros(len(y_test))
for z in range(3):
    mask = z_test == z
    if mask.sum() == 0: continue
    preds           = zone_models[z].predict(X_test[mask])
    preds          += zone_biases[z]
    all_preds[mask] = preds

overall_mae  = mean_absolute_error(y_test, all_preds)
overall_r2   = r2_score(y_test, all_preds)
overall_rmse = float(np.sqrt(np.mean((y_test - all_preds) ** 2)))
print(f"  MAE  : {overall_mae:.2f} K")
print(f"  RMSE : {overall_rmse:.2f} K")
print(f"  R²   : {overall_r2:.4f}")

print("\n  Per Tg-range breakdown:")
test_df        = pd.DataFrame({'y': y_test, 'yhat': all_preds})
test_df['bin'] = pd.cut(test_df['y'],
                        bins=[0, 200, 300, 400, 500, 9999],
                        labels=['<200K','200-300K','300-400K','400-500K','>500K'])
for name, grp in test_df.groupby('bin', observed=True):
    if len(grp) < 2: continue
    print(f"    {name:<12}  MAE={mean_absolute_error(grp['y'],grp['yhat']):.1f} K"
          f"  R²={r2_score(grp['y'],grp['yhat']):.4f}  (n={len(grp)})")

# =========================================================
# 7. ROUTER-ROUTED RESULTS
# =========================================================
print("\n=== ROUTER-ROUTED TEST RESULTS ===")
routed_preds = np.zeros(len(y_test))
for z in range(3):
    mask = router_preds == z
    if mask.sum() == 0: continue
    preds              = zone_models[z].predict(X_test[mask])
    preds             += zone_biases[z]
    routed_preds[mask] = preds
routed_mae  = mean_absolute_error(y_test, routed_preds)
routed_r2   = r2_score(y_test, routed_preds)
routed_rmse = float(np.sqrt(np.mean((y_test - routed_preds) ** 2)))
print(f"  MAE  : {routed_mae:.2f} K")
print(f"  RMSE : {routed_rmse:.2f} K")
print(f"  R²   : {routed_r2:.4f}")

# =========================================================
# 8. GENERATE PLOTS
# =========================================================
print("\nGenerating plots...")
plot_loss_curves(loss_histories, output_prefix="tg_plots")
plot_residuals(y_test, all_preds, z_test, output_prefix="tg_plots")
print("  Plots saved: tg_plots_4_loss_curves.png, tg_plots_5_residual_plot.png")

# =========================================================
# 9. SAVE ALL ARTIFACTS
# =========================================================
joblib.dump(zone_models,  "zone_models.pkl")
joblib.dump(zone_biases,  "zone_biases.pkl")
joblib.dump(router,       "zone_router.pkl")
joblib.dump(keep_cols,    "feature_columns.pkl")
joblib.dump(fill_median,  "fill_median.pkl")

print("\nSaved:")
print("  zone_models.pkl     — 3 zone XGBoost regressors")
print("  zone_biases.pkl     — per-zone bias corrections")
print("  zone_router.pkl     — XGBoost zone classifier")
print("  feature_columns.pkl")
print("  fill_median.pkl")
print("\nTraining complete. Run predict.py to generate predictions.")
