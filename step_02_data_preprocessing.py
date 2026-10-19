# Fichier python pour le prétraitement des données du projet d'assurance automobile

# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # Bibliothèque pour transformer les colonnes de manière spécifique
from sklearn.impute import SimpleImputer # Bibliothèque pour gérer les valeurs manquantes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline # Bibliothèque pour créer des pipelines de transformation et de modélisation
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder # Bibliothèques pour encoder les variables catégorielles (ajout du TargetEncoder pour la composante géographique) et standardiser les variables numériques

# Colonnes identifiantes ou cibles à exclure systématiquement des features
COLS_TO_DROP = [
    'nombre_sinistres',
    'montant_sinistre',
    'is_sinistre',
    'id_client',
    'id_vehicule',
    'id_contrat',
    'index',
    'modele_vehicule', # On exclut le modèle du véhicule car il possède plus de 800 modalités, ce qui fait exploser la dimensionnalité et entraîne un sur-apprentissage massif sur des profils rares.
    'code_postal', # On exclut le code postal brut car avec ~12000 valeurs, il génère trop de bruit. On le remplace par le 'departement' créé plus bas.
    # Suppression des variables trop corrélées : debut_vente_vehicule et fin_vente_vehicule présentent une corrélation > 0.94 avec anciennete_vehicule et entre elles (évite la redondance et la multicolinéarité)
    'debut_vente_vehicule',
    'fin_vente_vehicule'
]

# Fonction dédiée à la création de nouvelles variables (Feature Engineering)
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Crée de nouvelles variables métiers pertinentes pour l'assurance auto afin d'aider les modèles basés sur les arbres."""
    df_eng = df.copy()
    
    # Création de la variable 'age_obtention_permis1' : Un jeune permis de 40 ans n'est pas le même risque qu'un conducteur de 40 ans ayant 20 ans de permis.
    if 'age_conducteur1' in df_eng.columns and 'anciennete_permis1' in df_eng.columns:
        df_eng['age_obtention_permis1'] = df_eng['age_conducteur1'] - df_eng['anciennete_permis1']
    
    # Création de la variable 'ratio_puissance_poids' : Permet d'identifier les véhicules à fort caractère sportif. Le +1 évite l'erreur mathématique de division par zéro.
    if 'din_vehicule' in df_eng.columns and 'poids_vehicule' in df_eng.columns:
        df_eng['ratio_puissance_poids'] = df_eng['din_vehicule'] / (df_eng['poids_vehicule'] + 1)
        
    # Regroupement géographique propre. On extrait les 2 premiers caractères du code postal pour obtenir le département (en s'assurant du format sur 5 caractères).
    if 'code_postal' in df_eng.columns:
        df_eng['departement'] = df_eng['code_postal'].astype(str).str.zfill(5).str[:2]
        
    # Création de l'indicateur binaire 'is_jeune_permis' : conducteur novice avec moins de 3 ans de permis, profil à sinistralité statistiquement plus élevée
    if 'anciennete_permis1' in df_eng.columns:
        df_eng['is_jeune_permis'] = (df_eng['anciennete_permis1'] < 3).astype(int)
        
    # Création du flag 'conducteur2_jeune' : conducteur secondaire de moins de 25 ans déclaré sur le véhicule (cas classique de l'enfant ajouté sur le contrat des parents)
    if 'age_conducteur2' in df_eng.columns:
        df_eng['conducteur2_jeune'] = (
            (df_eng['age_conducteur2'] > 0) & (df_eng['age_conducteur2'] < 25)
        ).astype(int)
        
    # Feature engineering orienté gravité (coût des réparations) :
    # Ratio valeur / ancienneté : approximation de la valeur résiduelle du véhicule (pièces neuves et véhicules récents plus onéreux)
    if 'prix_vehicule' in df_eng.columns and 'anciennete_vehicule' in df_eng.columns:
        df_eng['ratio_valeur_anciennete'] = df_eng['prix_vehicule'] / (df_eng['anciennete_vehicule'] + 1)
        
    # Ratio puissance / cylindrée : rendement mécanique et complexité technologique du moteur (pièces détachées plus chères)
    if 'din_vehicule' in df_eng.columns and 'cylindre_vehicule' in df_eng.columns:
        df_eng['ratio_puissance_cylindree'] = df_eng['din_vehicule'] / (df_eng['cylindre_vehicule'] + 1)
        
    return df_eng

# Fonction pour nettoyer les valeurs aberrantes connues dans le dataset brut
def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    cols_permis = ['anciennete_permis1', 'anciennete_permis2']
    for col in cols_permis:
        if col in df_clean.columns:
            # Remplacement du code d'erreur 111 par un vrai NaN (vide) mathématiquement compréhensible par le SimpleImputer
            df_clean[col] = df_clean[col].replace(111, np.nan)
            
    # On applique la création de variables (Feature Engineering) directement lors du nettoyage initial
    df_clean = feature_engineering(df_clean)
    return df_clean

# Fonction pour charger le fichier CSV et appliquer le nettoyage initial
def load_data(filepath: str = 'train.csv') -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return clean_raw_data(df)

# Fonction pour préparer les splits d'entraînement et de validation pour la classification (fréquence) avec découpage stratifié
def prepare_classification_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    df_proc = df.copy()
    df_proc['is_sinistre'] = (df_proc['nombre_sinistres'] > 0).astype(int)

    X = df_proc.drop(columns=COLS_TO_DROP, errors='ignore')
    y = df_proc['is_sinistre']

    # La stratification est essentielle ici car nous n'avons que ~5.8% de sinistrés. Cela garantit la même proportion stricte de sinistrés dans le Train et le Validation.
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

# Fonction pour préparer les splits d'entraînement et de validation pour la régression (gravité) restreinte aux sinistrés
def prepare_regression_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    # On filtre uniquement sur les clients ayant eu un sinistre car le montant (gravité) ne se modélise que sur la sous-population accidentée.
    df_sinistres = df[df['nombre_sinistres'] > 0].copy()

    X = df_sinistres.drop(columns=COLS_TO_DROP, errors='ignore')
    y = df_sinistres['montant_sinistre']

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

# Fonction pour préparer les features du jeu de test pour la soumission Kaggle
def prepare_test_features(df_test: pd.DataFrame) -> pd.DataFrame:
    # Aligne parfaitement les colonnes du jeu de test de soumission avec celles attendues par la pipeline.
    return df_test.drop(columns=COLS_TO_DROP, errors='ignore')

# Fonction pour construire le préprocesseur (ColumnTransformer) adapté aux types de colonnes du jeu d'entraînement
def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    var_num = X_train.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()
    
    # On retire 'departement' des variables catégorielles standard pour ne pas faire un One-Hot Encoding dessus
    var_cat = X_train.select_dtypes(
        include=['object', 'category']
    ).columns.drop('departement', errors='ignore').tolist()
    
    # On isole la variable departement pour la passer au Target Encoder
    var_geo = ['departement'] if 'departement' in X_train.columns else []

    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')), # La médiane est privilégiée face à la forte asymétrie des données financières
            ('scaler', StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                'imputer',
                SimpleImputer(strategy='constant', fill_value='Inconnu'),
            ),
            (
                'onehot',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            ),
        ]
    )
    
    # Pipeline exclusive pour la géographie (le département). 
    # Le TargetEncoder va remplacer l'identifiant brut du département par la probabilité historique de sinistre moyenne constatée pour ce même département.
    geo_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('target_enc', TargetEncoder(smooth='auto')) # Le paramètre smooth empêche de donner un poids extrême à un département où il n'y a eu qu'un ou deux clients dans l'historique.
    ])

    # On assemble les trois pipelines distincts (Numériques, Catégorielles, et Géographique)
    transformers = [
        ('num', numeric_transformer, var_num),
        ('cat', categorical_transformer, var_cat)
    ]
    
    # On ajoute la pipeline geo (TargetEncoder) seulement si le département existe dans le dataframe d'entraînement
    if var_geo:
        transformers.append(('geo', geo_transformer, var_geo))

    return ColumnTransformer(transformers=transformers)