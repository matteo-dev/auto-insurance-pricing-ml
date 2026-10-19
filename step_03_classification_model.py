# Fichier python pour l'entraînement et l'évaluation des modèles de classification (Fréquence des sinistres)

# Importation des bibliothèques nécessaires
import os
import matplotlib.pyplot as plt
import numpy as np
import optuna  # Bibliothèque pour l'optimisation bayésienne des hyperparamètres (TPE)
import pandas as pd
from scipy.optimize import minimize_scalar  # Bibliothèque pour optimiser les poids du blending mathématiquement
from sklearn.calibration import CalibrationDisplay  # Bibliothèque pour visualiser la calibration des probabilités
from sklearn.dummy import DummyClassifier  # Bibliothèque pour le modèle baseline de fréquence (fréquence moyenne observée)
from sklearn.linear_model import LogisticRegression  # Bibliothèque pour le modèle de régression logistique
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score, roc_curve  # Bibliothèques pour évaluer les performances des modèles de classification
from sklearn.model_selection import StratifiedKFold  # Validation croisée stratifiée étanche
from xgboost import XGBClassifier  # Bibliothèque pour le modèle XGBoost de classification

# On importe uniquement nos fonctions encapsulées depuis data_preprocessing.py
from step_02_data_preprocessing import (
    build_preprocessor,
    load_data,
    prepare_classification_split,
)

# Désactivation des messages verbeux d'Optuna pour préserver la lisibilité de la console
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- FONCTIONS MODULAIRES POUR LA FRÉQUENCE ---

# Fonction pour optimiser les hyperparamètres de XGBoost Classification par optimisation bayésienne (Optuna)
def tune_xgboost_frequency(X_train_prep, y_train, n_trials=30):
    """Recherche bayésienne des hyperparamètres optimaux pour XGBoost Classification.

    Maximise le score ROC-AUC moyen par validation croisée stratifiée (3 plis).
    """
    print(f"\n[Optuna] Lancement de l'optimisation bayesienne sur XGBoost Frequence ({n_trials} essais)...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 150, 400, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 25),
            'subsample': trial.suggest_float('subsample', 0.65, 0.90),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.50, 0.85),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': 42,
        }

        scores = []
        for train_idx, val_idx in cv.split(X_train_prep, y_train):
            X_tr, X_va = X_train_prep[train_idx], X_train_prep[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, preds))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[Optuna] Meilleur ROC-AUC CV Frequence : {study.best_value:.4f}")
    print("[Optuna] Meilleurs hyperparametres trouves :", study.best_params)
    return study.best_params

# Fonction pour instancier les modèles de fréquence (Régression Logistique et XGBoost) avec leurs hyperparamètres optimisés
def get_frequency_models(xgb_params=None):
    """Instancie et retourne les modèles de classification (fréquence).

    Permet d'injecter des hyperparamètres optimisés par Optuna ou d'utiliser les valeurs robustes par défaut.
    """
    log_reg = LogisticRegression(max_iter=200, random_state=42)

    base_params = {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'max_depth': 3,
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42,
    }
    if xgb_params is not None:
        base_params.update(xgb_params)

    xgb_model = XGBClassifier(**base_params)
    return log_reg, xgb_model

# Fonction pour entraîner les modèles de fréquence sur les données prétraitées
def train_frequency_models(X_train_prep, y_train, xgb_params=None):
    """Entraîne la régression logistique et XGBoost sur les données d'entraînement prétraitées."""
    log_reg, xgb_model = get_frequency_models(xgb_params=xgb_params)
    log_reg.fit(X_train_prep, y_train)
    xgb_model.fit(X_train_prep, y_train)
    return log_reg, xgb_model

# Fonction pour optimiser le poids du blending entre Régression Logistique et XGBoost par maximisation du ROC-AUC
def optimize_frequency_blending(p_logreg, p_xgb, y_val):
    """Calcule le poids optimal w dans [0, 1] maximisant le ROC-AUC de la combinaison convexe w*p_lr + (1-w)*p_xgb."""
    def objective_auc(weight_logreg: float) -> float:
        p_blend = weight_logreg * p_logreg + (1.0 - weight_logreg) * p_xgb
        return -roc_auc_score(y_val, p_blend)

    res_opt = minimize_scalar(objective_auc, bounds=(0.0, 1.0), method='bounded')
    optimal_weight_logreg = float(res_opt.x)
    optimal_weight_xgb = 1.0 - optimal_weight_logreg
    return optimal_weight_logreg, optimal_weight_xgb

# Fonction pour calculer la prédiction combinée de fréquence (Ensemble Blending) à partir des modèles entraînés
def predict_frequency_ensemble(log_reg, xgb_model, X_prep, weight_logreg=0.45):
    """Calcule la probabilité de sinistre combinée à partir des deux modèles entraînés."""
    p_logreg = log_reg.predict_proba(X_prep)[:, 1]
    p_xgb = xgb_model.predict_proba(X_prep)[:, 1]
    return weight_logreg * p_logreg + (1.0 - weight_logreg) * p_xgb


# --- SCRIPT D'ÉVALUATION ET DE VISUALISATION (ÉTAPE 3) ---
if __name__ == '__main__':
    # Création du dossier de sauvegarde sur ton ordi
    output_dir = 'resultats_etape3'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Chargement et séparation des données
    df_train = load_data('train.csv')

    # Séparation train/validation stratifiée
    X_train, X_val, y_train, y_val = prepare_classification_split(
        df_train, test_size=0.2, random_state=42
    )

    # 2. Pipeline de transformation
    preprocessor = build_preprocessor(X_train)

    # Fit strictly on train, transform on both
    X_train_prep = preprocessor.fit_transform(X_train, y_train)
    X_val_prep = preprocessor.transform(X_val)

    feature_names = preprocessor.get_feature_names_out()

    # OPTIONNEL : Passer execute_optuna à True pour lancer la recherche bayésienne sur le train
    execute_optuna = True
    best_xgb_params = None
    if execute_optuna:
        best_xgb_params = tune_xgboost_frequency(X_train_prep, y_train, n_trials=30)

    # 3. Entraînement des modèles 
    baseline = DummyClassifier(strategy='prior')
    baseline.fit(X_train_prep, y_train)

    # Entraînement des modèles via la fonction modulaire avec les paramètres optimisés
    log_reg, xgb_model = train_frequency_models(X_train_prep, y_train, xgb_params=best_xgb_params)

    # 4. Prédictions individuelles
    p_baseline = baseline.predict_proba(X_val_prep)[:, 1]
    p_logreg = log_reg.predict_proba(X_val_prep)[:, 1]
    p_xgb = xgb_model.predict_proba(X_val_prep)[:, 1]

    # 5. Optimisation mathématique du Blending via SciPy
    optimal_weight_logreg, optimal_weight_xgb = optimize_frequency_blending(p_logreg, p_xgb, y_val)

    print(f"\n--- Optimisation du Blending (SciPy) ---")
    print(f"Poids optimal RegLog : {optimal_weight_logreg:.4f}")
    print(f"Poids optimal XGBoost : {optimal_weight_xgb:.4f}")

    # Calcul des probabilités combinées avec les poids optimaux via la fonction modulaire
    p_ensemble = predict_frequency_ensemble(
        log_reg, xgb_model, X_val_prep, weight_logreg=optimal_weight_logreg
    )

    # 6. Évaluation, Graphiques et Sauvegardes
    predictions = {
        'Baseline': p_baseline,
        'RegLog': p_logreg,
        'XGBoost': p_xgb,
        f'Ensemble Optimise ({optimal_weight_logreg:.2f}L + {optimal_weight_xgb:.2f}X)': p_ensemble
    }

    resultats = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for nom, y_pred_proba in predictions.items():
        auc = roc_auc_score(y_val, y_pred_proba)
        loss = log_loss(y_val, y_pred_proba)
        resultats.append({'Modele': nom, 'ROC-AUC': auc, 'Log-Loss': loss})

        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        axes[0].plot(fpr, tpr, label=f'{nom} (AUC = {auc:.3f})')

        CalibrationDisplay.from_predictions(y_val, y_pred_proba, n_bins=10, ax=axes[1], name=nom)

    axes[0].plot([0, 1], [0, 1], 'k--', label='Aléatoire')
    axes[0].set_xlabel('Taux de faux positifs')
    axes[0].set_ylabel('Taux de vrais positifs')
    axes[0].set_title('Courbes ROC')
    axes[0].legend()

    axes[1].set_title('Calibration des probabilités')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'ROC_and_Calibration.png'), dpi=300)
    print(f"\nGraphique sauvegardé dans {output_dir}/ROC_and_Calibration.png")
    plt.show()

    df_resultats = pd.DataFrame(resultats)
    print('\n--- Performances sur le jeu de validation ---')
    print(df_resultats.to_string(index=False))
    df_resultats.to_csv(os.path.join(output_dir, 'performances_modeles.csv'), index=False)

    seuil_metier = 0.06 
    y_pred_ensemble_seuil = (p_ensemble >= seuil_metier).astype(int)
    confusion_matrix_ens = confusion_matrix(y_val, y_pred_ensemble_seuil)
    confusion_matrix_ens_csv = pd.DataFrame(
        confusion_matrix_ens, 
        index=['Reel Non Sinistré', 'Reel Sinistré'], 
        columns=['Prédit Non Sinistré', 'Prédit Sinistré']
    )
    confusion_matrix_ens_csv.to_csv(os.path.join(output_dir, 'matrice_confusion_ensemble.csv'), index=True)
    print(f"Matrice de confusion sauvegardée dans {output_dir}/matrice_confusion_ensemble.csv")

    df_importance = pd.DataFrame(
        {'Variable': feature_names, 'Importance': xgb_model.feature_importances_}
    ).sort_values(by='Importance', ascending=False)
    df_importance.to_csv(os.path.join(output_dir, 'importance_variables_frequence.csv'), index=False)
    print('\n--- Top 10 des variables les plus importantes (XGBoost) ---')
    print(df_importance.head(10).to_string(index=False))