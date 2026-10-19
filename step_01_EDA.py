# Fichier python pour l'analyse exploratoire des données (EDA) du projet d'assurance automobile

# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration esthétique pour les graphiques
sns.set_theme(style="whitegrid")

# 1. Chargement des données d'entraînement
# Règle d'or : on n'explore JAMAIS le jeu de test pour éviter le data leakage
df = pd.read_csv('train.csv')

print("--- Apercu du dataset ---")
print((df.head()))

print("\n--- Informations Generales ---")
df.info()

# 2. Analyse des valeurs manquantes
print("\n--- Valeurs Manquantes par colonne ---")
missing_data = df.isnull().sum()[df.isnull().sum() > 0]
if not missing_data.empty:
    print(missing_data.sort_values(ascending=False))
else:
    print("Aucune valeur manquante detectee.")

# Sauvegarde des valeurs manquantes dans un fichier CSV
missing_data.to_csv('missing_values.csv')

# 3. Analyse des variables cibles
# Création d'une variable binaire pour modéliser la fréquence (classification)
df['is_sinistre'] = (df['nombre_sinistres'] > 0).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Cible 1 : Fréquence (Déséquilibre des classes)
sns.countplot(data=df, x='is_sinistre', ax=axes[0], palette='viridis')
axes[0].set_title('Distribution de la cible Classification (Sinistré vs Non Sinistré)')
prop_sinistre = df['is_sinistre'].mean() * 100
axes[0].set_xlabel(f'0 = Non Sinistré, 1 = Sinistré ({prop_sinistre:.2f}%)')

# Cible 2 : Gravité (Asymétrie des montants)
# On filtre logiquement sur les clients ayant eu un sinistre
df_sinistres = df[df['is_sinistre'] == 1]
sns.histplot(df_sinistres['montant_sinistre'], bins=50, kde=True, ax=axes[1], color='coral')
axes[1].set_title('Distribution des montants de sinistres (Gravité)')
axes[1].set_xlabel('Montant du sinistre')

plt.tight_layout()
plt.savefig('target_analysis.png')

# 4. Analyse des variables explicatives
# Identification des types de variables (hors cibles)
cibles = ['nombre_sinistres', 'montant_sinistre', 'is_sinistre']
var_num = df.select_dtypes(include=['int64', 'float64']).columns.drop(cibles, errors='ignore')
var_cat = df.select_dtypes(include=['object', 'category']).columns

print("\n--- Statistiques descriptives (Variables Numeriques) ---")
print((df[var_num].describe()))

# Sauvegarde des statistiques descriptives dans un fichier CSV
df[var_num].describe().to_csv('descriptive_statistics_numerical.csv')

# Vérification de l'asymétrie (skewness) sur les features numériques
print("\n--- Asymetrie (Skewness) des variables numeriques ---")

# Une valeur > 1 ou < -1 indique une forte asymétrie
print(df[var_num].skew().sort_values(ascending=False))

# Sauvegarde de l'asymétrie dans un fichier CSV
df[var_num].skew().sort_values(ascending=False).to_csv('skewness_numerical.csv')

# Détection des modalités rares dans les variables catégorielles
print("\n--- Analyse des modalites rares (Variables Categorielles) ---")
for col in var_cat:
    freq = df[col].value_counts(normalize=True)
    rares = freq[freq < 0.05] # Seuil fixé arbitrairement à 5%
    if not rares.empty:
        print(f"\n{col} : Modalites representant moins de 5% des donnees :\n{rares}")

# Sauvegarde des modalités rares dans un fichier CSV
rare_modalities = {}
for col in var_cat:
    freq = df[col].value_counts(normalize=True)
    rares = freq[freq < 0.05]
    if not rares.empty:
        rare_modalities[col] = rares
rare_modalities_df = pd.DataFrame({k: v for k, v in rare_modalities.items()})
rare_modalities_df.to_csv('rare_modalities_categorical.csv')

# 5. Corrélations
# On calcule la matrice de corrélation sur les variables numériques incluant les cibles
corr_cols = list(var_num) + ['is_sinistre', 'montant_sinistre']
corr_matrix = df[corr_cols].corr()

# Sauvegarde de cette matrice en CSV
corr_matrix.to_csv('correlation_matrix.csv')

# Visualisation de la matrice de corrélation
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matrice de correlation (Variables Numeriques & Cibles)')
plt.savefig('correlation_matrix.png')

# Je veux voir la ligne ou j'ai une missing value de la colonne anciennete_vehicule
missing_values_anciennete = df[df['anciennete_vehicule'].isnull()]
print("\n--- Lignes avec valeurs manquantes pour 'anciennete_vehicule' ---")
print(missing_values_anciennete)

# Sauvegarde des lignes avec valeurs manquantes pour 'anciennete_vehicule' dans un fichier CSV
missing_values_anciennete.to_csv('missing_values_anciennete_vehicule.csv', index=False)