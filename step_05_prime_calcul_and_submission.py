# Fichier python pour l'etape 5 : Calcul de la prime pure, audit de rentabilite et soumission Kaggle

# Importation des bibliothèques necessaires
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error  # evaluation de la RMSE globale sur la prime pure
from sklearn.model_selection import train_test_split  # Separation etanche train/validation

# Importation des fonctions modulaires creees aux etapes 2, 3 et 4
from step_02_data_preprocessing import (
    COLS_TO_DROP,
    build_preprocessor,
    load_data,
    prepare_test_features,
)
from step_03_classification_model import (
    optimize_frequency_blending,
    predict_frequency_ensemble,
    train_frequency_models,
    tune_xgboost_frequency,
)
from step_04_regression_model import (
    optimize_severity_blending,
    predict_severity_ensemble,
    train_severity_models,
    tune_xgboost_severity,
)

# Configuration d'execution
# Passer à True pour relancer la recherche Optuna (30 essais par modèle)
# Passer à False pour utiliser directement les hyperparamètres optimises par recherche bayesienne
RUN_BAYESIAN_OPTIMIZATION = True

# 1. Creation du dossier d'export local pour l'etape 5
output_dir = 'resultats_etape5'
os.makedirs(output_dir, exist_ok=True)

# 2. Chargement des donnees d'entraînement complètes
# load_data applique automatiquement le nettoyage et le feature engineering metier
df_train = load_data('train.csv')
print(f"Jeu d'entraînement charge : {df_train.shape[0]} contrats, {df_train.shape[1]} colonnes.")


# PARTIE A : AUDIT DE RENTABILITe ET RMSE GLOBALE SUR LA VALIDATION (20 %)
print("PARTIE A : AUDIT FINANCIER & OPTIMISATION DE LA PRIME SUR VALIDATION")

# Decoupage stratifie au niveau du DataFrame pour aligner classification et regression sur les mêmes clients
train_df, val_df = train_test_split(
    df_train,
    test_size=0.2,
    random_state=42,
    stratify=(df_train['nombre_sinistres'] > 0),
)

# --- A.1 Modelisation de la Frequence (Classification) sur train_df ---
X_train_cls = train_df.drop(columns=COLS_TO_DROP, errors='ignore')
y_train_cls = (train_df['nombre_sinistres'] > 0).astype(int)

X_val = val_df.drop(columns=COLS_TO_DROP, errors='ignore')
y_val_cls = (val_df['nombre_sinistres'] > 0).astype(int)

preprocessor_cls = build_preprocessor(X_train_cls)
X_train_cls_prep = preprocessor_cls.fit_transform(X_train_cls, y_train_cls)
X_val_cls_prep = preprocessor_cls.transform(X_val)

# Hyperparamètres XGBoost Frequence (issus de l'optimisation bayesienne Optuna)
if RUN_BAYESIAN_OPTIMIZATION:
    best_xgb_cls_params = tune_xgboost_frequency(X_train_cls_prep, y_train_cls, n_trials=30)
else:
    best_xgb_cls_params = {
        'n_estimators': 300,
        'learning_rate': 0.028,
        'max_depth': 3,
        'min_child_weight': 12,
        'subsample': 0.82,
        'colsample_bytree': 0.68,
        'reg_alpha': 1.45,
        'reg_lambda': 6.20,
    }

# Entraînement de la frequence avec les hyperparamètres bayesiens
model_logreg, model_xgb_cls = train_frequency_models(
    X_train_cls_prep, y_train_cls, xgb_params=best_xgb_cls_params
)

# Optimisation du blending frequence sur la validation
p_lr_val = model_logreg.predict_proba(X_val_cls_prep)[:, 1]
p_xgb_val = model_xgb_cls.predict_proba(X_val_cls_prep)[:, 1]
w_cls_opt, _ = optimize_frequency_blending(p_lr_val, p_xgb_val, y_val_cls)

p_val = predict_frequency_ensemble(
    model_logreg, model_xgb_cls, X_val_cls_prep, weight_logreg=w_cls_opt
)

# --- A.2 Modelisation de la Gravite (Regression) restreinte aux sinistres ---
train_sinistres = train_df[train_df['nombre_sinistres'] > 0].copy()
X_train_sev = train_sinistres.drop(columns=COLS_TO_DROP, errors='ignore')
y_train_sev = train_sinistres['montant_sinistre']

preprocessor_sev = build_preprocessor(X_train_sev)
X_train_sev_prep = preprocessor_sev.fit_transform(X_train_sev, y_train_sev)
X_val_sev_prep = preprocessor_sev.transform(X_val)

# Hyperparamètres XGBoost Gravite (issus de l'optimisation bayesienne Optuna)
if RUN_BAYESIAN_OPTIMIZATION:
    best_xgb_sev_params = tune_xgboost_severity(X_train_sev_prep, y_train_sev, n_trials=30)
else:
    best_xgb_sev_params = {
        'n_estimators': 150,
        'learning_rate': 0.032,
        'max_depth': 3,
        'min_child_weight': 10,
        'subsample': 0.78,
        'colsample_bytree': 0.65,
        'reg_alpha': 1.10,
        'reg_lambda': 5.80,
    }

# Entraînement de la gravite avec les hyperparamètres bayesiens et realignement
severity_models_val = train_severity_models(
    X_train_sev_prep, y_train_sev, xgb_params=best_xgb_sev_params
)

# Optimisation du blending gravite sur la validation des sinistres
val_sinistres_mask = (val_df['nombre_sinistres'] > 0).values
X_val_sev_only = preprocessor_sev.transform(val_df[val_df['nombre_sinistres'] > 0].drop(columns=COLS_TO_DROP, errors='ignore'))
y_val_sev_only = val_df[val_df['nombre_sinistres'] > 0]['montant_sinistre']

g_ridge_val_sin = np.exp(
    severity_models_val['ridge_log'].predict(X_val_sev_only) + (severity_models_val['sigma2_train'] / 2.0)
) * severity_models_val['facteur_echelle_ridge']
g_xgb_val_sin = severity_models_val['xgb_gamma'].predict(X_val_sev_only) * severity_models_val['facteur_echelle_xgb']
w_sev_opt, _ = optimize_severity_blending(g_ridge_val_sin, g_xgb_val_sin, y_val_sev_only)

# Inference de gravite sur l'ensemble des assures du jeu de validation
g_val = predict_severity_ensemble(
    severity_models_val, X_val_sev_prep, weight_ridge=w_sev_opt
)

# --- A.3 Calcul de la Prime Pure et verification de la rentabilite ---
# Formule actuarielle fondamentale : Prime = p(x) * g(x)
primes_val = p_val * g_val
montants_reels_val = val_df['montant_sinistre'].values

somme_primes_val = float(np.sum(primes_val))
somme_sinistres_val = float(np.sum(montants_reels_val))
solde_technique = somme_primes_val - somme_sinistres_val
loss_ratio = (somme_sinistres_val / somme_primes_val) * 100.0
rmse_prime_validation = root_mean_squared_error(montants_reels_val, primes_val)

print(f"Nombre de contrats evalues        : {len(val_df):,}")
print(f"Somme des primes perçues          : {somme_primes_val:,.2f} €")
print(f"Somme des sinistres indemnises    : {somme_sinistres_val:,.2f} €")
print(f"Solde technique brut (P - S)      : {solde_technique:+,.2f} €")
print(f"Ratio Sinistres / Primes (S/P)    : {loss_ratio:.2f} %")
print(f"Prime pure moyenne par client     : {primes_val.mean():.2f} €")
print(f"Coût reel moyen par client        : {montants_reels_val.mean():.2f} €")
print(f"RMSE Prime Pure sur validation    : {rmse_prime_validation:.2f} €")

if somme_primes_val >= somme_sinistres_val:
    print(">> VERDICT ACTUARIEL : L'assureur est RENTABLE sur son portefeuille (solde positif).")
else:
    print(">> VERDICT ACTUARIEL : Portefeuille en leger deficit technique.")

# Sauvegarde des metriques d'audit
df_audit = pd.DataFrame([
    {'Indicateur': 'Primes Collectees (€)', 'Valeur': round(somme_primes_val, 2)},
    {'Indicateur': 'Sinistres Reels (€)', 'Valeur': round(somme_sinistres_val, 2)},
    {'Indicateur': 'Solde Technique (€)', 'Valeur': round(solde_technique, 2)},
    {'Indicateur': 'Loss Ratio S/P (%)', 'Valeur': round(loss_ratio, 2)},
    {'Indicateur': 'Prime Pure Moyenne (€)', 'Valeur': round(float(primes_val.mean()), 2)},
    {'Indicateur': 'Coût Moyen Reel (€)', 'Valeur': round(float(montants_reels_val.mean()), 2)},
    {'Indicateur': 'RMSE Prime Pure (€)', 'Valeur': round(float(rmse_prime_validation), 2)},
])
df_audit.to_csv(os.path.join(output_dir, 'audit_rentabilite_validation.csv'), index=False)

# Graphique de diagnostic financier
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(primes_val, bins=50, range=(0, 400), color='royalblue', edgecolor='black', alpha=0.7)
axes[0].axvline(primes_val.mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne : {primes_val.mean():.2f} €')
axes[0].set_title('Distribution des primes pures individuelles (Validation)')
axes[0].set_xlabel('Prime pure predite (€)')
axes[0].set_ylabel("Nombre d'assures")
axes[0].legend()

bars = axes[1].bar(
    ['Primes perçues', 'Sinistres reels'],
    [somme_primes_val, somme_sinistres_val],
    color=['forestgreen', 'firebrick'],
    alpha=0.8,
    edgecolor='black',
)
axes[1].set_title('equilibre financier global du portefeuille')
axes[1].set_ylabel('Montant total cumule (€)')
for bar in bars:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval * 0.92, f'{yval:,.0f} €', ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'audit_financier_validation.png'), dpi=300)
plt.show()


# PARTIE B : ReENTRAÎNEMENT GLOBAL (100 %) & SOUMISSION KAGGLE
print("PARTIE B : ReENTRAÎNEMENT COMPLET (100 % DU TRAIN) & TEST.CSV")

# --- B.1 Reentraînement Frequence sur 100 % des donnees (50 000 contrats) ---
print("1/3 Reentraînement Frequence sur 50 000 contrats...")
X_full_cls = df_train.drop(columns=COLS_TO_DROP, errors='ignore')
y_full_cls = (df_train['nombre_sinistres'] > 0).astype(int)

preprocessor_cls_full = build_preprocessor(X_full_cls)
X_full_cls_prep = preprocessor_cls_full.fit_transform(X_full_cls, y_full_cls)

full_logreg, full_xgb_cls = train_frequency_models(
    X_full_cls_prep, y_full_cls, xgb_params=best_xgb_cls_params
)

# --- B.2 Reentraînement Gravite sur 100 % des sinistres (2 917 dossiers) ---
print("2/3 Reentraînement Gravite sur l'ensemble des sinistres...")
df_sinistres_full = df_train[df_train['nombre_sinistres'] > 0].copy()
X_full_sev = df_sinistres_full.drop(columns=COLS_TO_DROP, errors='ignore')
y_full_sev = df_sinistres_full['montant_sinistre']

preprocessor_sev_full = build_preprocessor(X_full_sev)
X_full_sev_prep = preprocessor_sev_full.fit_transform(X_full_sev, y_full_sev)

full_severity_models = train_severity_models(
    X_full_sev_prep, y_full_sev, xgb_params=best_xgb_sev_params
)

# --- B.3 Inference sur test.csv ---
print("3/3 Inference sur le fichier test.csv...")
df_test = load_data('test.csv')
X_test = prepare_test_features(df_test)

# Inference Frequence
X_test_cls_prep = preprocessor_cls_full.transform(X_test)
p_test = predict_frequency_ensemble(
    full_logreg, full_xgb_cls, X_test_cls_prep, weight_logreg=w_cls_opt
)

# Inference Gravite
X_test_sev_prep = preprocessor_sev_full.transform(X_test)
g_test = predict_severity_ensemble(
    full_severity_models, X_test_sev_prep, weight_ridge=w_sev_opt
)

# Calcul de la prime pure finale sur le jeu de test Kaggle
prime_pure_test = p_test * g_test

# --- B.4 Creation du fichier de soumission conforme à prime_pred_sandbox.csv ---
submission_df = pd.DataFrame({
    'index': df_test['index'],
    'pred': prime_pure_test,
})

# Verifications d'integrite strictes
assert submission_df.shape == (50000, 2), "Erreur : dimensions incorrectes !"
assert not submission_df.isna().any().any(), "Erreur : presence de NaN !"
assert (submission_df['pred'] > 0).all(), "Erreur : presences de primes non strictement positives !"

nom_fichier_soumission = 'submission_kaggle.csv'
submission_df.to_csv(nom_fichier_soumission, index=False)
submission_df.to_csv(os.path.join(output_dir, nom_fichier_soumission), index=False)

print(f"\nFichier de soumission genere avec succès : '{nom_fichier_soumission}'")
print("\nAperçu des 5 premières lignes :")
print(submission_df.head().to_string(index=False))

print("\nStatistiques descriptives de vos primes predites sur le jeu de test :")
print(submission_df['pred'].describe().to_frame().T.round(2).to_string())