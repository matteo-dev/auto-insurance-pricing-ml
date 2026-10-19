# Tarification d'Assurance Automobile par Machine Learning

Ce projet implémente une pipeline complète de Machine Learning pour le calcul de la **prime pure** en assurance automobile. Il repose sur l'approche actuarielle standard séparant le risque en deux dimensions : la **fréquence** des sinistres et leur **gravité**.

## Contexte et Objectifs
L'objectif est d'évaluer le coût moyen attendu des sinistres d'un client. La prime pure se modélise mathématiquement par :
`Prime Pure = P(Fréquence > 0) × E[Gravité | Fréquence > 0]`

Pour résoudre ce problème, deux modèles ont été développés :
1. **Modèle de Fréquence (Classification) :** Prédire la probabilité qu'un client déclare au moins un sinistre (gestion d'un fort déséquilibre de classes à ~5.8%).
2. **Modèle de Gravité (Régression) :** Prédire le coût du sinistre s'il survient (gestion d'une distribution financière fortement asymétrique avec transformation log-normale et distribution Gamma).

## Stack Technique
* **Langage :** Python
* **Modélisation :** Scikit-Learn, XGBoost
* **Optimisation :** Optuna (Optimisation Bayésienne), SciPy (Blending mathématique)
* **Dashboard :** Streamlit, Matplotlib, Seaborn

## Structure du Projet
Le projet est découpé en 5 étapes modulaires pour éviter toute fuite de données (data leakage) et assurer la reproductibilité :

* `step_01_EDA.py` : Analyse exploratoire (distributions cibles, multicolinéarité).
* `step_02_data_preprocessing.py` : Feature engineering métier et pipelines Scikit-Learn (Target Encoding pour les départements, imputations robustes).
* `step_03_classification_model.py` : Entraînement, calibration des probabilités et blending (Régression Logistique + XGBoost).
* `step_04_regression_model.py` : Modélisation des coûts, réalignement d'échelle et correction de l'inégalité de Jensen.
* `step_05_prime_calcul_and_submission.py` : Audit de rentabilité (Loss Ratio) et prédictions finales.
* `step_06_dashboard.py` : Interface Streamlit interactive présentant les résultats.

## Installation et Utilisation
```bash
# Cloner le dépôt
git clone [https://github.com/ton-profil/auto-insurance-pricing-ml.git](https://github.com/ton-profil/auto-insurance-pricing-ml.git)

# Installer les dépendances
pip install pandas numpy scikit-learn xgboost optuna streamlit matplotlib seaborn scipy

# Lancer le dashboard interactif
streamlit run app.py
```

## ENGLISH 

# Auto Insurance Pricing using Machine Learning

This repository contains an end-to-end Machine Learning pipeline for calculating **pure premiums** in auto insurance. It follows the industry-standard actuarial approach by decoupling the risk into two dimensions: claim **frequency** and claim **severity**.

## Context and Objectives
The goal is to estimate the expected cost of claims for a given policyholder. The pure premium is mathematically modeled as:
`Pure Premium = P(Claim > 0) × E[Amount | Claim > 0]`

To achieve this, two distinct models were developed:
1. **Frequency Model (Classification):** Predicts the probability of a client filing at least one claim (handling extreme class imbalance at ~5.8%).
2. **Severity Model (Regression):** Predicts the expected cost of the claim if an accident occurs (handling highly right-skewed financial distributions using Log-normal corrections and Gamma objectives).

## Tech Stack
* **Language:** Python
* **Modeling:** Scikit-Learn, XGBoost
* **Optimization:** Optuna (Bayesian Optimization), SciPy (Mathematical Blending)
* **Dashboard:** Streamlit, Matplotlib, Seaborn

## Project Structure
The pipeline is modularized into 5 steps to strictly prevent data leakage and ensure reproducibility:

* `step_01_EDA.py`: Exploratory Data Analysis (target distributions, multicollinearity).
* `step_02_data_preprocessing.py`: Domain-specific feature engineering and Scikit-Learn pipelines (Target Encoding for Zip Codes, robust imputation).
* `step_03_classification_model.py`: Model training, probability calibration, and blending (Logistic Regression + XGBoost).
* `step_04_regression_model.py`: Cost modeling, scale realignment, and Jensen's inequality correction.
* `step_05_prime_calcul_and_submission.py`: Actuarial profitability audit (Loss Ratio) and final predictions.
* `step_06_dashboard.py`: Interactive Streamlit dashboard showcasing the methodology and results.

## How to Run
```bash
# Clone the repository
git clone [https://github.com/your-profile/auto-insurance-pricing-ml.git](https://github.com/your-profile/auto-insurance-pricing-ml.git)

# Install dependencies
pip install pandas numpy scikit-learn xgboost optuna streamlit matplotlib seaborn scipy

# Launch the interactive dashboard
streamlit run app.py
