# Fichier python pour la création et l'entraînement des modèles de régression pour le projet d'assurance automobile

# Importation des bibliothèques nécessaires
import os
import matplotlib.pyplot as plt
import numpy as np
import optuna  # Bibliothèque pour l'optimisation bayésienne des hyperparamètres (TPE)
import pandas as pd
from scipy.optimize import minimize_scalar  # Bibliothèque pour optimiser les poids du blending mathématiquement
from sklearn.linear_model import Ridge  # Bibliothèque pour la régression Ridge (L2)
from sklearn.metrics import mean_absolute_error, root_mean_squared_error  # Bibliothèques pour évaluer les performances des modèles de régression
from sklearn.model_selection import KFold  # Validation croisée K-Fold pour la régression
from xgboost import XGBRegressor  # Bibliothèque pour le modèle XGBoost de régression

# Importation des fonctions encapsulées depuis data_preprocessing.py pour le prétraitement des données
from step_02_data_preprocessing import (
    build_preprocessor,
    load_data,
    prepare_regression_split,
)

# Désactivation des logs verbeux d'Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- FONCTIONS MODULAIRES POUR LA GRAVITÉ ---

# Fonction pour optimiser les hyperparamètres de XGBoost Régression Gamma par optimisation bayésienne (Optuna)
def tune_xgboost_severity(X_train_prep, y_train, n_trials=30):
    """Recherche bayésienne des hyperparamètres optimaux pour XGBoost Régression Gamma.

    Minimise la RMSE moyenne réalignée par validation croisée (3 plis) sur les sinistrés.
    """
    print(f"\n[Optuna] Lancement de l'optimisation bayesienne sur XGBoost Gravite ({n_trials} essais)...")
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'objective': 'reg:gamma',
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
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

        rmse_scores = []
        for train_idx, val_idx in cv.split(X_train_prep, y_train):
            X_tr, X_va = X_train_prep[train_idx], X_train_prep[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)

            # Prédiction avec réalignement d'échelle sur le pli d'entraînement
            pred_tr = model.predict(X_tr)
            scale = float(y_tr.mean() / pred_tr.mean())
            pred_va = model.predict(X_va) * scale

            rmse_scores.append(root_mean_squared_error(y_va, pred_va))

        return np.mean(rmse_scores)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[Optuna] Meilleure RMSE CV Gravite : {study.best_value:.2f} €")
    print("[Optuna] Meilleurs hyperparametres trouves :", study.best_params)
    return study.best_params

# Fonction pour instancier les modèles de gravité (Ridge Log et XGBoost Gamma) avec leurs hyperparamètres optimisés
def get_severity_models(xgb_params=None):
    """Instancie et retourne les modèles de régression (gravité).

    Centralise les hyperparamètres Ridge et XGBoost Gamma pour l'ensemble du projet.
    """
    ridge_log = Ridge(alpha=50.0, random_state=42)

    base_params = {
        'objective': 'reg:gamma',
        'n_estimators': 150,
        'learning_rate': 0.03,
        'max_depth': 3,
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

    xgb_gamma = XGBRegressor(**base_params)
    return ridge_log, xgb_gamma

# Fonction pour entraîner les modèles de gravité et calculer les paramètres d'inversion log-normale et de réalignement d'échelle
def train_severity_models(X_train_prep, y_train, xgb_params=None):
    """Entraîne les modèles de gravité et calcule la variance des résidus sigma^2 ainsi que les facteurs d'échelle pour l'équilibre budgétaire."""
    ridge_log, xgb_gamma = get_severity_models(xgb_params=xgb_params)

    # Entraînement Ridge sur log(montant)
    y_train_log = np.log(y_train)
    ridge_log.fit(X_train_prep, y_train_log)
    pred_log_train = ridge_log.predict(X_train_prep)
    sigma2_train = float(np.var(y_train_log - pred_log_train))
    pred_ridge_corr_train = np.exp(pred_log_train + (sigma2_train / 2.0))
    facteur_echelle_ridge = float(y_train.mean() / pred_ridge_corr_train.mean())

    # Entraînement XGBoost Gamma
    xgb_gamma.fit(X_train_prep, y_train)
    pred_xgb_train = xgb_gamma.predict(X_train_prep)
    facteur_echelle_xgb = float(y_train.mean() / pred_xgb_train.mean())

    return {
        'ridge_log': ridge_log,
        'xgb_gamma': xgb_gamma,
        'sigma2_train': sigma2_train,
        'facteur_echelle_ridge': facteur_echelle_ridge,
        'facteur_echelle_xgb': facteur_echelle_xgb,
    }

# Fonction pour prédire la gravité avec Ridge Log corrigé et réaligné
def predict_ridge_severity(ridge_model, X_prep, sigma2, facteur_echelle):
    """Calcule l'espérance en euros prédite par Ridge avec correction d'inégalité de Jensen et facteur d'échelle."""
    pred_log = ridge_model.predict(X_prep)
    return np.exp(pred_log + (sigma2 / 2.0)) * facteur_echelle

# Fonction pour prédire la gravité avec XGBoost Gamma réaligné
def predict_xgb_severity(xgb_model, X_prep, facteur_echelle):
    """Calcule le montant en euros prédit par XGBoost Gamma avec facteur de réalignement d'échelle."""
    return xgb_model.predict(X_prep) * facteur_echelle

# Fonction pour optimiser le poids du blending entre Ridge et XGBoost par minimisation de la RMSE sur validation
def optimize_severity_blending(pred_ridge, pred_xgb, y_val):
    """Calcule le poids optimal w dans [0, 1] minimisant la RMSE de la combinaison w*g_ridge + (1-w)*g_xgb."""
    def objective_rmse(weight_ridge: float) -> float:
        pred_blend_temp = weight_ridge * pred_ridge + (1.0 - weight_ridge) * pred_xgb
        return root_mean_squared_error(y_val, pred_blend_temp)

    res_opt = minimize_scalar(objective_rmse, bounds=(0.0, 1.0), method='bounded')
    optimal_weight_ridge = float(res_opt.x)
    optimal_weight_xgb = 1.0 - optimal_weight_ridge
    return optimal_weight_ridge, optimal_weight_xgb

# Fonction pour prédire la gravité combinée (Ensemble Blending) à partir du dictionnaire de modèles
def predict_severity_ensemble(models_dict, X_prep, weight_ridge=0.70):
    """Calcule la prédiction combinée de gravité en appliquant les facteurs d'échelle et la pondération du blending."""
    g_ridge = predict_ridge_severity(
        models_dict['ridge_log'],
        X_prep,
        models_dict['sigma2_train'],
        models_dict['facteur_echelle_ridge'],
    )
    g_xgb = predict_xgb_severity(
        models_dict['xgb_gamma'],
        X_prep,
        models_dict['facteur_echelle_xgb'],
    )
    return weight_ridge * g_ridge + (1.0 - weight_ridge) * g_xgb


# --- SCRIPT D'ÉVALUATION ET DE VISUALISATION (ÉTAPE 4) ---
if __name__ == '__main__':
    # 1. Création du dossier d'export local pour l'étape 4
    output_dir = 'resultats_etape4'
    os.makedirs(output_dir, exist_ok=True)

    # 2. Chargement et séparation des données restreintes aux sinistrés
    df_raw = load_data('train.csv')

    X_train, X_val, y_train, y_val = prepare_regression_split(
        df_raw, test_size=0.2, random_state=42
    )

    print(f"Nombre d'observations d'entraînement (sinistrés) : {len(X_train)}")
    print(f"Nombre d'observations de validation (sinistrés)    : {len(X_val)}")

    # 3. Pipeline de transformation
    preprocessor = build_preprocessor(X_train)

    X_train_prep = preprocessor.fit_transform(X_train, y_train)
    X_val_prep = preprocessor.transform(X_val)

    feature_names = preprocessor.get_feature_names_out()

    # OPTIONNEL : Passer execute_optuna à True pour lancer la recherche bayésienne sur la gravité
    execute_optuna = True
    best_xgb_sev_params = None
    if execute_optuna:
        best_xgb_sev_params = tune_xgboost_severity(X_train_prep, y_train, n_trials=30)

    # 4. Modèle 1 : Baselines empiriques
    mean_train = float(y_train.mean())
    pred_baseline_mean = np.full(shape=len(y_val), fill_value=mean_train)

    median_train = float(y_train.median())
    pred_baseline_median = np.full(shape=len(y_val), fill_value=median_train)

    # 5. Entraînement des modèles de gravité via la fonction modulaire
    severity_models = train_severity_models(X_train_prep, y_train, xgb_params=best_xgb_sev_params)

    pred_ridge_final = predict_ridge_severity(
        severity_models['ridge_log'],
        X_val_prep,
        severity_models['sigma2_train'],
        severity_models['facteur_echelle_ridge'],
    )
    pred_xgb_final = predict_xgb_severity(
        severity_models['xgb_gamma'],
        X_val_prep,
        severity_models['facteur_echelle_xgb'],
    )

    # 6. Optimisation mathématique du Blending via SciPy
    optimal_weight_ridge, optimal_weight_xgb = optimize_severity_blending(
        pred_ridge_final, pred_xgb_final, y_val
    )

    print(f"\n--- Optimisation du Blending Gravité (SciPy) ---")
    print(f"Poids optimal Ridge Log : {optimal_weight_ridge:.4f}")
    print(f"Poids optimal XGBoost   : {optimal_weight_xgb:.4f}")

    pred_ensemble_final = predict_severity_ensemble(
        severity_models, X_val_prep, weight_ridge=optimal_weight_ridge
    )

    # 7. Évaluation comparative des performances
    modeles_preds = {
        'Baseline (Moyenne)': pred_baseline_mean,
        'Baseline (Médiane)': pred_baseline_median,
        'Ridge Log (Corrigé + Réaligné)': pred_ridge_final,
        'XGBoost Gamma (Réaligné)': pred_xgb_final,
        f'Ensemble ({optimal_weight_ridge:.2f}R + {optimal_weight_xgb:.2f}X)': pred_ensemble_final,
    }

    resultats = []
    for nom, preds in modeles_preds.items():
        rmse = root_mean_squared_error(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        moyenne_predite = float(np.mean(preds))
        resultats.append(
            {
                'Modèle': nom,
                'RMSE (€)': round(rmse, 2),
                'MAE (€)': round(mae, 2),
                'Moyenne prédite (€)': round(moyenne_predite, 2),
            }
        )

    df_resultats = pd.DataFrame(resultats)
    print(f"\nMoyenne réelle observée sur la validation : {y_val.mean():.2f} €")
    print('\n--- Performances des modèles de gravité sur la validation ---')
    print(df_resultats.to_string(index=False))

    df_resultats.to_csv(
        os.path.join(output_dir, 'performances_gravite.csv'), index=False
    )

    # 8. Visualisations diagnostiques
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].hist(
        y_val,
        bins=40,
        range=(0, 6000),
        alpha=0.4,
        density=True,
        label='Réel (Validation)',
        color='black',
    )
    axes[0].hist(
        pred_ensemble_final,
        bins=40,
        range=(0, 6000),
        alpha=0.5,
        density=True,
        label='Ensemble Optimisé',
        color='crimson',
    )
    axes[0].hist(
        pred_xgb_final,
        bins=40,
        range=(0, 6000),
        alpha=0.4,
        density=True,
        label='XGBoost Gamma Réaligné',
        color='forestgreen',
    )
    axes[0].set_title('Distribution des montants (zoom < 6 000 €)')
    axes[0].set_xlabel('Montant du sinistre (€)')
    axes[0].set_ylabel('Densité')
    axes[0].legend()

    axes[1].scatter(y_val, pred_ensemble_final, alpha=0.4, color='crimson', s=20)
    axes[1].plot(
        [y_val.min(), y_val.max()],
        [y_val.min(), y_val.max()],
        'r--',
        label='Prédiction parfaite (y = x)',
    )
    axes[1].set_title("Réel vs Prédit (Ensemble Optimisé)")
    axes[1].set_xlabel('Montant réel (€)')
    axes[1].set_ylabel('Montant prédit (€)')
    axes[1].set_xlim(0, 10000)
    axes[1].set_ylim(0, 5000)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagnostic_gravite.png'), dpi=300)
    print(f"\nGraphique sauvegarde dans {output_dir}/diagnostic_gravite.png")
    plt.show()

    df_val_preds = pd.DataFrame(
        {
            'montant_reel': y_val,
            'montant_predit_ensemble': pred_ensemble_final,
        }
    )
    df_val_preds.to_csv(
        os.path.join(output_dir, 'predictions_gravite_validation.csv'), index=False
    )

    df_importance = pd.DataFrame(
        {
            'Variable': feature_names,
            'Importance': severity_models['xgb_gamma'].feature_importances_,
        }
    ).sort_values(by='Importance', ascending=False)
    df_importance.to_csv(
        os.path.join(output_dir, 'importance_variables_gravite.csv'), index=False
    )

    print('\n--- Top 10 des variables les plus importantes (Gravité) ---')
    print(df_importance.head(10).to_string(index=False))