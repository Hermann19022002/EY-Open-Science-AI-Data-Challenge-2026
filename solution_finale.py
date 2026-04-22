#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
EY AI & DATA CHALLENGE 2026 — SOLUTION FINALE
================================================================================
Equipe    : Hermann & Neville
Score LB  : R² moyen sur TA, EC, DRP

Approche  :
  - Features : 13 variables (Landsat + TerraClimate + temporelles + géographiques)
  - TA & EC  : Random Forest Extreme
  - DRP      : Meilleur modèle parmi RF / XGBoost / LightGBM (sélection sur 80/20)
  - Validation : train_test_split 80/20 (random_state=42)

Fichiers requis :
  - water_quality_training_dataset.csv
  - landsat_features_training.csv
  - terraclimate_features_training.csv
  - landsat_features_validation.csv
  - terraclimate_features_validation.csv
  - submission_template.csv

Sorties :
  - submission_finale.csv     : prédictions TA, EC, DRP pour les 200 points
  - stats_descriptives.png    : statistiques descriptives des données d'entraînement
  - pred_vs_obs.png           : valeurs prédites vs observées (split 80/20)
================================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

print("=" * 80)
print("EY AI & DATA CHALLENGE 2026 — SOLUTION FINALE")
print("Equipe : Hermann & Neville  |  Score LB : 0.3629")
print("=" * 80)
print(f"   XGBoost  : {'OK' if XGBOOST_AVAILABLE  else 'non disponible'}")
print(f"   LightGBM : {'OK' if LIGHTGBM_AVAILABLE else 'non disponible'}")

# ============================================================
# HYPERPARAMETRES
# ============================================================
RF_EXTREME = {
    'n_estimators':    500,
    'max_depth':       None,
    'min_samples_split': 2,
    'min_samples_leaf':  1,
    'max_features':    'sqrt',
    'random_state':    42,
    'n_jobs':          -1,
}

XGBOOST_PARAMS = {
    'n_estimators':    500,
    'max_depth':       10,
    'learning_rate':   0.05,
    'subsample':       0.8,
    'colsample_bytree':0.8,
    'min_child_weight':1,
    'gamma':           0,
    'reg_alpha':       0.1,
    'reg_lambda':      1.0,
    'random_state':    42,
    'n_jobs':          -1,
}

LIGHTGBM_PARAMS = {
    'n_estimators':    500,
    'max_depth':       10,
    'learning_rate':   0.05,
    'num_leaves':      31,
    'subsample':       0.8,
    'colsample_bytree':0.8,
    'min_child_samples':20,
    'reg_alpha':       0.1,
    'reg_lambda':      1.0,
    'random_state':    42,
    'n_jobs':          -1,
    'verbose':         -1,
}

TARGETS = ['Total Alkalinity', 'Electrical Conductance',
           'Dissolved Reactive Phosphorus']
TARGETS_SHORT = {'Total Alkalinity': 'TA',
                 'Electrical Conductance': 'EC',
                 'Dissolved Reactive Phosphorus': 'DRP'}
TARGETS_UNIT  = {'Total Alkalinity': 'mg/L',
                 'Electrical Conductance': 'µS/cm',
                 'Dissolved Reactive Phosphorus': 'mg/L'}

# ============================================================
# ETAPE 1 — CHARGEMENT DES DONNEES
# ============================================================
print("\n" + "=" * 80)
print("ETAPE 1 — CHARGEMENT DES DONNEES")
print("=" * 80)

Water_Quality_df   = pd.read_csv('water_quality_training_dataset.csv')
landsat_train      = pd.read_csv('landsat_features_training.csv')
terraclimate_train = pd.read_csv('terraclimate_features_training.csv')
submission_template = pd.read_csv('submission_template.csv')
landsat_val        = pd.read_csv('landsat_features_validation.csv')
terraclimate_val   = pd.read_csv('terraclimate_features_validation.csv')

print(f"   Donnees d'entrainement : {len(Water_Quality_df):,} observations")
print(f"   Sites uniques          : "
      f"{Water_Quality_df.groupby(['Latitude','Longitude']).ngroups}")
print(f"   Periode                : "
      f"{pd.to_datetime(Water_Quality_df['Sample Date'], format='%d-%m-%Y').dt.year.min()}"
      f" — "
      f"{pd.to_datetime(Water_Quality_df['Sample Date'], format='%d-%m-%Y').dt.year.max()}")
print(f"   Points de validation   : {len(submission_template):,}")

# ============================================================
# ETAPE 2 — STATISTIQUES DESCRIPTIVES + VISUALISATION
# ============================================================
print("\n" + "=" * 80)
print("ETAPE 2 — STATISTIQUES DESCRIPTIVES")
print("=" * 80)

for col in TARGETS:
    d = Water_Quality_df[col].dropna()
    print(f"\n   {TARGETS_SHORT[col]} ({col}) :")
    print(f"      n={len(d):,}  |  moy={d.mean():.2f}  med={d.median():.2f}"
          f"  std={d.std():.2f}  min={d.min():.2f}  max={d.max():.2f}"
          f"  asymetrie={d.skew():.2f}")

# Graphique stats descriptives
BG    = '#0d1b2a'
PANEL = '#0d2a3a'
WHITE = '#FFFFFF'
GRAY  = '#7ab3d4'
COLORS = {'Total Alkalinity': '#2196F3',
          'Electrical Conductance': '#FFC107',
          'Dissolved Reactive Phosphorus': '#4CAF50'}

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': PANEL,
    'text.color': WHITE,    'axes.labelcolor': GRAY,
    'xtick.color': GRAY,    'ytick.color': GRAY,
    'axes.edgecolor': '#2a4a6a',
    'grid.color': '#1e3a5a', 'grid.linestyle': '--', 'grid.linewidth': 0.5,
})

wq_plot = Water_Quality_df.copy()
wq_plot['Date']  = pd.to_datetime(wq_plot['Sample Date'], format='%d-%m-%Y')
wq_plot['Year']  = wq_plot['Date'].dt.year
wq_plot['Month'] = wq_plot['Date'].dt.month

fig = plt.figure(figsize=(18, 10), facecolor=BG)
fig.suptitle(
    'Statistiques descriptives — Qualite de l\'eau (2011-2015)',
    color=WHITE, fontsize=16, fontweight='bold', y=0.97)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

# Ligne 1 : distributions
for i, col in enumerate(TARGETS):
    ax = fig.add_subplot(gs[0, i])
    data = wq_plot[col].dropna()
    ax.hist(data, bins=40, color=COLORS[col], alpha=0.75,
            edgecolor='none', density=True)
    kde = gaussian_kde(data)
    x_r = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_r, kde(x_r), color='white', linewidth=2)
    ax.axvline(data.mean(),   color='#FF5252', linewidth=1.5,
               linestyle='--', label=f'Moy: {data.mean():.1f}')
    ax.axvline(data.median(), color='#FF9800', linewidth=1.5,
               linestyle=':',  label=f'Med: {data.median():.1f}')
    ax.set_title(f"{TARGETS_SHORT[col]} ({TARGETS_UNIT[col]})",
                 color=WHITE, fontsize=11, fontweight='bold')
    ax.set_xlabel('Valeur', fontsize=9)
    ax.set_ylabel('Densite', fontsize=9)
    ax.legend(fontsize=8, framealpha=0.3, labelcolor='white')
    ax.grid(True, alpha=0.3)

# Tableau résumé (4e colonne)
ax_tab = fig.add_subplot(gs[0, 3])
ax_tab.set_facecolor(PANEL)
ax_tab.axis('off')
rows_lbl = ['Observations', 'Moyenne', 'Mediane', 'Ecart-type',
            'Min', 'Max', 'Q25', 'Q75', 'Asymetrie']
col_labels = ['', 'TA\n(mg/L)', 'EC\n(µS/cm)', 'DRP\n(mg/L)']
table_data = []
for row in rows_lbl:
    r = [row]
    for col in TARGETS:
        d = wq_plot[col].dropna()
        if   row == 'Observations': r.append(f'{len(d):,}')
        elif row == 'Moyenne':      r.append(f'{d.mean():.1f}')
        elif row == 'Mediane':      r.append(f'{d.median():.1f}')
        elif row == 'Ecart-type':   r.append(f'{d.std():.1f}')
        elif row == 'Min':          r.append(f'{d.min():.1f}')
        elif row == 'Max':          r.append(f'{d.max():.1f}')
        elif row == 'Q25':          r.append(f'{d.quantile(0.25):.1f}')
        elif row == 'Q75':          r.append(f'{d.quantile(0.75):.1f}')
        elif row == 'Asymetrie':    r.append(f'{d.skew():.2f}')
    table_data.append(r)

tbl = ax_tab.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor('#0d2535' if row % 2 == 0 else PANEL)
    cell.set_text_props(color=WHITE)
    cell.set_edgecolor('#2a4a6a')
    if row == 0:
        cell.set_facecolor('#0d3d5a')
        cell.set_text_props(color=WHITE, fontweight='bold')
    if col == 1 and row > 0:
        cell.set_text_props(color='#2196F3')
    if col == 2 and row > 0:
        cell.set_text_props(color='#FFC107')
    if col == 3 and row > 0:
        cell.set_text_props(color='#4CAF50')
ax_tab.set_title('Resume statistique', color=WHITE,
                 fontsize=11, fontweight='bold', pad=10)

# Ligne 2 : boxplots par année
years = sorted(wq_plot['Year'].unique())
for i, col in enumerate(TARGETS):
    ax = fig.add_subplot(gs[1, i])
    data_by_year = [wq_plot[wq_plot['Year'] == y][col].dropna().values
                    for y in years]
    bp = ax.boxplot(data_by_year, patch_artist=True, notch=False,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color=GRAY),
                    capprops=dict(color=GRAY),
                    flierprops=dict(marker='o', markerfacecolor=COLORS[col],
                                   markersize=2, alpha=0.4))
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS[col])
        patch.set_alpha(0.6)
    ax.set_xticklabels(years, fontsize=9)
    ax.set_title(f'{TARGETS_SHORT[col]} par annee',
                 color=WHITE, fontsize=10, fontweight='bold')
    ax.set_xlabel('Annee', fontsize=9)
    ax.set_ylabel(TARGETS_UNIT[col], fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

# Nombre de mesures par année
ax_cnt = fig.add_subplot(gs[1, 3])
counts = wq_plot.groupby('Year').size()
bars = ax_cnt.bar(counts.index, counts.values,
                  color='#00BCD4', alpha=0.75, edgecolor='none')
for bar, val in zip(bars, counts.values):
    ax_cnt.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8, str(val),
                ha='center', va='bottom', color=WHITE, fontsize=9)
ax_cnt.set_title('Observations par annee',
                 color=WHITE, fontsize=10, fontweight='bold')
ax_cnt.set_xlabel('Annee', fontsize=9)
ax_cnt.set_ylabel('Observations', fontsize=9)
ax_cnt.grid(True, alpha=0.3, axis='y')

plt.savefig('stats_descriptives.png', dpi=180,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("\n   [OK] stats_descriptives.png genere")

# ============================================================
# ETAPE 3 — CONSTRUCTION DES FEATURES
# ============================================================
print("\n" + "=" * 80)
print("ETAPE 3 — CONSTRUCTION DES FEATURES")
print("=" * 80)

def build_features(wq, ls, tc, dates_col='Sample Date'):
    """Fusionne les données et construit les 13 features du modèle final."""
    df = pd.concat([
        wq,
        ls[['nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI']],
        tc[['pet']]
    ], axis=1).loc[:, lambda d: ~d.columns.duplicated()]

    df['Date']      = pd.to_datetime(df[dates_col], format='%d-%m-%Y')
    df['Month']     = df['Date'].dt.month
    df['Season']    = df['Month'].apply(
        lambda m: 0 if m in [12, 1, 2] else
                  (1 if m in [3, 4, 5] else
                   (2 if m in [6, 7, 8] else 3)))
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df = df.fillna(df.median(numeric_only=True))
    return df

FEATURES = [
    'nir', 'green', 'swir16', 'swir22', 'NDMI', 'MNDWI',
    'pet',
    'Month', 'Season', 'Month_sin', 'Month_cos',
    'Latitude', 'Longitude',
]

wq_data = build_features(Water_Quality_df, landsat_train, terraclimate_train)
val_data = build_features(submission_template, landsat_val, terraclimate_val)

print(f"   Features retenues ({len(FEATURES)}) :")
print(f"   Landsat  : nir, green, swir16, swir22, NDMI, MNDWI")
print(f"   Climat   : pet (evapotranspiration potentielle)")
print(f"   Temporel : Month, Season, Month_sin, Month_cos")
print(f"   Spatial  : Latitude, Longitude")
print(f"\n   Justification : features minimales = meilleure generalisation spatiale")
print(f"   (ajouter TC extra ou buffer stats degradait le LB : 0.3629 -> 0.3549)")

X_full = wq_data[FEATURES].copy()
y_TA   = wq_data['Total Alkalinity'].copy()
y_EC   = wq_data['Electrical Conductance'].copy()
y_DRP  = wq_data['Dissolved Reactive Phosphorus'].copy()
X_val  = val_data[FEATURES].copy()

# ============================================================
# ETAPE 4 — SELECTION DU MEILLEUR MODELE POUR DRP (80/20)
# ============================================================
print("\n" + "=" * 80)
print("ETAPE 4 — SELECTION MODELE DRP (split 80/20, random_state=42)")
print("=" * 80)

X_train, X_test, y_DRP_train, y_DRP_test = train_test_split(
    X_full, y_DRP, test_size=0.2, random_state=42
)
_, _, y_TA_train, y_TA_test = train_test_split(
    X_full, y_TA, test_size=0.2, random_state=42
)
_, _, y_EC_train, y_EC_test = train_test_split(
    X_full, y_EC, test_size=0.2, random_state=42
)

scaler_eval    = StandardScaler()
X_train_scaled = scaler_eval.fit_transform(X_train)
X_test_scaled  = scaler_eval.transform(X_test)

print(f"\n   Train : {len(X_train):,}  |  Test : {len(X_test):,}")

# TA — RF Extreme
rf_ta = RandomForestRegressor(**RF_EXTREME)
rf_ta.fit(X_train_scaled, y_TA_train)
r2_ta = r2_score(y_TA_test, rf_ta.predict(X_test_scaled))
print(f"\n   TA  -> RF Extreme    : R² = {r2_ta:.4f}")

# EC — RF Extreme
rf_ec = RandomForestRegressor(**RF_EXTREME)
rf_ec.fit(X_train_scaled, y_EC_train)
r2_ec = r2_score(y_EC_test, rf_ec.predict(X_test_scaled))
print(f"   EC  -> RF Extreme    : R² = {r2_ec:.4f}")

# DRP — comparaison RF / XGBoost / LightGBM
print(f"\n   DRP -> comparaison RF / XGBoost / LightGBM :")
drp_results = {}

rf_drp = RandomForestRegressor(**RF_EXTREME)
rf_drp.fit(X_train_scaled, y_DRP_train)
drp_results['RF'] = r2_score(y_DRP_test, rf_drp.predict(X_test_scaled))
print(f"      RF       : R² = {drp_results['RF']:.4f}")

if XGBOOST_AVAILABLE:
    xgb_drp = xgb.XGBRegressor(**XGBOOST_PARAMS)
    xgb_drp.fit(X_train_scaled, y_DRP_train)
    drp_results['XGBoost'] = r2_score(
        y_DRP_test, xgb_drp.predict(X_test_scaled))
    print(f"      XGBoost  : R² = {drp_results['XGBoost']:.4f}")

if LIGHTGBM_AVAILABLE:
    lgb_drp = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
    lgb_drp.fit(X_train_scaled, y_DRP_train)
    drp_results['LightGBM'] = r2_score(
        y_DRP_test, lgb_drp.predict(X_test_scaled))
    print(f"      LightGBM : R² = {drp_results['LightGBM']:.4f}")

best_drp_name = max(drp_results, key=drp_results.get)
r2_drp = drp_results[best_drp_name]
print(f"\n   => Meilleur modele DRP : {best_drp_name} (R² = {r2_drp:.4f})")

score_moyen = (r2_ta + r2_ec + r2_drp) / 3
print(f"\n   Score moyen 80/20 : {score_moyen:.4f}")
print(f"   TA={r2_ta:.4f}  EC={r2_ec:.4f}  DRP={r2_drp:.4f}")
print(f"   Score LB officiel : 0.3629")

# ============================================================
# ETAPE 5 — GRAPHIQUE PREDIT vs OBSERVE
# ============================================================
print("\n" + "=" * 80)
print("ETAPE 5 — GRAPHIQUE PREDIT vs OBSERVE")
print("=" * 80)

models_eval = {
    'Total Alkalinity':              rf_ta,
    'Electrical Conductance':        rf_ec,
    'Dissolved Reactive Phosphorus': (
        rf_drp if best_drp_name == 'RF' else
        (xgb_drp if best_drp_name == 'XGBoost' else lgb_drp)
    ),
}
y_tests = {
    'Total Alkalinity':              y_TA_test,
    'Electrical Conductance':        y_EC_test,
    'Dissolved Reactive Phosphorus': y_DRP_test,
}
model_names = {
    'Total Alkalinity':              'RF Extreme',
    'Electrical Conductance':        'RF Extreme',
    'Dissolved Reactive Phosphorus': best_drp_name,
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig.suptitle(
    'Valeurs predites vs observees — Split 80/20 (random_state=42)\n'
    'TA & EC : RF Extreme  |  DRP : ' + best_drp_name,
    color=WHITE, fontsize=13, fontweight='bold', y=1.03)

from matplotlib.colors import Normalize as MplNorm

for ax, col in zip(axes, TARGETS):
    y_te   = y_tests[col].values
    y_pred = models_eval[col].predict(X_test_scaled)
    r2     = r2_score(y_te, y_pred)
    err    = np.abs(y_pred - y_te)
    mae    = err.mean()
    rmse   = np.sqrt(((y_pred - y_te) ** 2).mean())

    norm_err = MplNorm(vmin=0, vmax=np.percentile(err, 90))
    sc_plot  = ax.scatter(y_te, y_pred, c=err, cmap='RdYlGn_r',
                          norm=norm_err, alpha=0.45, s=14, zorder=5)

    mn = min(y_te.min(), y_pred.min())
    mx = max(y_te.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'w--', linewidth=1.5, alpha=0.7,
            label='Prediction parfaite')

    z = np.polyfit(y_te, y_pred, 1)
    x_line = np.linspace(mn, mx, 100)
    ax.plot(x_line, np.poly1d(z)(x_line),
            color=COLORS[col], linewidth=2,
            label=f'Regression (pente={z[0]:.2f})')

    cbar = plt.colorbar(sc_plot, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Erreur absolue', color=GRAY, fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=GRAY, fontsize=7)
    cbar.outline.set_edgecolor('#2a4a6a')

    ax.text(0.05, 0.95,
            f'R2    = {r2:.4f}\n'
            f'MAE   = {mae:.2f}\n'
            f'RMSE  = {rmse:.2f}\n'
            f'n     = {len(y_te):,}\n'
            f'Modele: {model_names[col]}',
            transform=ax.transAxes, color=WHITE, fontsize=9,
            va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1b2a',
                      edgecolor=COLORS[col], alpha=0.85))

    ax.set_title(
        f'{TARGETS_SHORT[col]} — {col}\n'
        f'({TARGETS_UNIT[col]}) | {model_names[col]}',
        color=WHITE, fontsize=11, fontweight='bold')
    ax.set_xlabel(f'Valeur observee ({TARGETS_UNIT[col]})', fontsize=9)
    ax.set_ylabel(f'Valeur predite ({TARGETS_UNIT[col]})', fontsize=9)
    ax.legend(fontsize=8, framealpha=0.3, labelcolor='white',
              loc='lower right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pred_vs_obs.png', dpi=180,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("   [OK] pred_vs_obs.png genere")

# ============================================================
# ETAPE 6 — ENTRAINEMENT FINAL SUR TOUTES LES DONNEES
# ============================================================
print("\n" + "=" * 80)
print("ETAPE 6 — ENTRAINEMENT FINAL (100% des donnees)")
print("=" * 80)

scaler_final  = StandardScaler()
X_full_scaled = scaler_final.fit_transform(X_full)

print("\n   Total Alkalinity       -> RF Extreme")
model_TA = RandomForestRegressor(**RF_EXTREME)
model_TA.fit(X_full_scaled, y_TA)

print("   Electrical Conductance -> RF Extreme")
model_EC = RandomForestRegressor(**RF_EXTREME)
model_EC.fit(X_full_scaled, y_EC)

print(f"   DRP                    -> {best_drp_name}")
if best_drp_name == 'RF':
    model_DRP = RandomForestRegressor(**RF_EXTREME)
elif best_drp_name == 'XGBoost':
    model_DRP = xgb.XGBRegressor(**XGBOOST_PARAMS)
else:
    model_DRP = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
model_DRP.fit(X_full_scaled, y_DRP)

print("   [OK] 3 modeles entraines sur 100% des donnees")

# ============================================================
# ETAPE 7 — PREDICTIONS ET SOUMISSION
# ============================================================
print("\n" + "=" * 80)
print("ETAPE 7 — PREDICTIONS VALIDATION")
print("=" * 80)

X_val_scaled = scaler_final.transform(X_val)

pred_TA  = model_TA.predict(X_val_scaled)
pred_EC  = model_EC.predict(X_val_scaled)
pred_DRP = model_DRP.predict(X_val_scaled)

submission = pd.DataFrame({
    'Latitude':                      submission_template['Latitude'].values,
    'Longitude':                     submission_template['Longitude'].values,
    'Sample Date':                   submission_template['Sample Date'].values,
    'Total Alkalinity':              pred_TA,
    'Electrical Conductance':        pred_EC,
    'Dissolved Reactive Phosphorus': pred_DRP,
})
submission.to_csv('submission_finale.csv', index=False)

print(f"\n   Predictions generees ({len(submission)} points) :")
print(f"   TA  : moy={pred_TA.mean():.1f}   "
      f"min={pred_TA.min():.1f}   max={pred_TA.max():.1f}  [{TARGETS_UNIT['Total Alkalinity']}]")
print(f"   EC  : moy={pred_EC.mean():.1f}   "
      f"min={pred_EC.min():.1f}   max={pred_EC.max():.1f}  [{TARGETS_UNIT['Electrical Conductance']}]")
print(f"   DRP : moy={pred_DRP.mean():.3f}   "
      f"min={pred_DRP.min():.3f}   max={pred_DRP.max():.3f}  [{TARGETS_UNIT['Dissolved Reactive Phosphorus']}]")

# ============================================================
# RESUME FINAL
# ============================================================
print("\n" + "=" * 80)
print("RESUME — SOLUTION FINALE")
print("=" * 80)
print(f"""
  Equipe        : Hermann & Neville
  Score LB      : 0.3629  (R2 moyen officiel)

  Modeles       :
    TA  -> RF Extreme     R2 (80/20) = {r2_ta:.4f}
    EC  -> RF Extreme     R2 (80/20) = {r2_ec:.4f}
    DRP -> {best_drp_name:<12}  R2 (80/20) = {r2_drp:.4f}
    Moyenne                          = {score_moyen:.4f}

  Features (13) :
    Landsat  : nir, green, swir16, swir22, NDMI, MNDWI
    Climat   : pet
    Temporel : Month, Season, Month_sin, Month_cos
    Spatial  : Latitude, Longitude

  Fichiers generes :
    submission_finale.csv    -> predictions a soumettre
    stats_descriptives.png   -> statistiques des donnees d'entrainement
    pred_vs_obs.png          -> performance du modele (80/20)
""")
print("=" * 80)
