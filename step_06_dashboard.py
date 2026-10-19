# Fichier python pour l'interface streamlit 

# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import os
from PIL import Image

 
# CONFIGURATION DE LA PAGE
 
st.set_page_config(
    page_title="Auto Insurance Pricing Portfolio",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application d'un style CSS personnalisé pour le portfolio
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: bold; }
    .sub-header { font-size: 1.5rem; color: #3B82F6; font-weight: 600; margin-top: 2rem;}
    .metric-card { background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

 
# FONCTIONS UTILITAIRES
 
@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    """Charge un fichier CSV en cache pour optimiser les performances."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()

def display_image(image_path: str, caption: str, width=None):
    """Affiche une image en toute sécurité si elle existe dans le répertoire."""

    # On gère les extensions potentiellement différentes (.png ou .jpg)
    alt_path = image_path.replace('.png', '.jpg') if image_path.endswith('.png') else image_path.replace('.jpg', '.png')
    
    if os.path.exists(image_path):
        img = Image.open(image_path)
        st.image(img, caption=caption, width=width, use_container_width=True)
    elif os.path.exists(alt_path):
        img = Image.open(alt_path)
        st.image(img, caption=caption, width=width, use_container_width=True)
    else:
        st.warning(f"Visualization missing: {image_path}. Please run the corresponding pipeline step to generate it.")

 
# BARRE DE NAVIGATION LATÉRALE

st.sidebar.title("Project Navigation")
st.sidebar.markdown("**Machine Learning Pipeline**")

step = st.sidebar.radio(
    "",
    [
        "0. Project Overview",
        "1. Exploratory Data Analysis",
        "2. Feature Engineering",
        "3. Claim Frequency Model",
        "4. Claim Severity Model",
        "5. Financial Audit & Premium"
    ]
)

 
# VUE 0 : VUE D'ENSEMBLE DU PROJET
 
if step == "0. Project Overview":
    st.markdown('<p class="main-header">Auto Insurance Pricing using Machine Learning</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Business Context
    In auto insurance, the premium is the price paid by the policyholder to be covered in the event of a claim. The goal of this project is to calculate the **Pure Premium**, which represents the expected cost of claims for a specific client. 
    
    To achieve extreme precision, the actuarial standard approaches this problem by splitting the risk into two distinct models:
    """)
    
    st.latex(r"\text{Pure Premium} = \mathbb{P}(\text{Claim Frequency}) \times \mathbb{E}[\text{Claim Severity} \mid \text{Claim} > 0]")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**1. Frequency Model (Classification)**\n\nPredicts the probability of a client having at least one claim during the coverage period. It deals with highly imbalanced classes.")
    with col2:
        st.success("**2. Severity Model (Regression)**\n\nPredicts the expected cost of the claim, assuming an accident has occurred. It deals with highly right-skewed financial distributions.")
        
    st.markdown("### Technical Highlights")
    st.markdown("""
    - **Target Encoding** for high-cardinality geographical features (zip codes).
    - **Bayesian Optimization** with Optuna to tune XGBoost hyperparameters.
    - **Mathematical Blending** combining robust linear models (Logistic/Ridge) with gradient boosting.
    - **Probability Calibration** ensuring that predicted probabilities correctly scale into financial metrics.
    """)

 
# VUE 1 : ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
 
elif step == "1. Exploratory Data Analysis":
    st.markdown('<p class="main-header">Step 1: Exploratory Data Analysis (EDA)</p>', unsafe_allow_html=True)
    st.markdown("Understanding the fundamental structure of the data and the target distributions before modeling.")
    
    st.markdown('<p class="sub-header">Target Variables Distribution</p>', unsafe_allow_html=True)
    st.markdown("""
    The targets exhibit two classic challenges in actuarial data science:
    1. **Extreme Class Imbalance**: Only **5.83%** of the dataset represents policyholders who filed a claim. Accuracy cannot be used as an evaluation metric here.
    2. **Right-Skewed Severity**: Claim amounts follow a long-tailed distribution (often modeled with Gamma or Tweedie distributions).
    """)
    
    display_image('target_analysis.png', "Figure 1: Target distributions (Frequency & Severity)")
    
    st.markdown("---")
    st.markdown('<p class="sub-header">Feature Correlation & Multicollinearity</p>', unsafe_allow_html=True)
    st.markdown("""
    Analyzing the correlation matrix reveals several structural data properties:
    - **High Multicollinearity**: Features such as `debut_vente_vehicule`, `fin_vente_vehicule`, and `anciennete_vehicule` have a correlation $> 0.95$. Redundant features were dropped to prevent instability in linear models.
    - **Target Leakage Prevention**: We meticulously removed variables that would only be known *after* a claim occurs to prevent data leakage during training.
    """)
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        display_image('correlation_matrix.png', "Figure 2: Pearson Correlation Matrix for Numerical Features")

 
# VUE 2 : PRÉPARATION DES DONNÉES ET PIPELINE
 
elif step == "2. Feature Engineering":
    st.markdown('<p class="main-header">Step 2: Robust Data Preprocessing</p>', unsafe_allow_html=True)
    st.markdown("Building a scikit-learn compatible pipeline to avoid data leakage between Train and Validation sets.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Feature Engineering Strategy")
        st.markdown("""
        To help tree-based models capture non-linear relationships, several domain-specific features were engineered:
        - **`age_obtention_permis1`**: Age of the driver when they obtained their license. Separates young people from older inexperienced drivers.
        - **`ratio_puissance_poids`**: Power-to-weight ratio to identify sports cars.
        - **`departement`**: Extracted the first 2 digits of the Zip Code to reduce cardinality from ~12,000 to ~100.
        """)
        
        st.markdown("### The Preprocessing Pipeline")
        st.code("""
        # Pipeline architecture
        transformers = [
            ('num', numeric_transformer, var_num), # Median Imputer + StandardScaler
            ('cat', categorical_transformer, var_cat), # Constant Imputer + OneHotEncoder
            ('geo', geo_transformer, var_geo) # Most Frequent + TargetEncoder (Smooth)
        ]
        preprocessor = ColumnTransformer(transformers=transformers)
        """, language="python")
    
    with col2:
        st.info("💡 **Target Encoding for Geography**\n\nInstead of One-Hot Encoding 100+ departments (which creates a massive sparse matrix), we used `TargetEncoder`. It replaces each department with its historical claim rate. We used Laplace smoothing to handle departments with very few policyholders.")

 
# VUE 3 : MODÈLE DE FRÉQUENCE
 
elif step == "3. Claim Frequency Model":
    st.markdown('<p class="main-header">Step 3: Claim Frequency Model (Classification)</p>', unsafe_allow_html=True)
    st.markdown("Predicting $\mathbb{P}(\text{Claim} > 0)$. Since the prediction will be multiplied by a monetary value, **probability calibration** is just as important as the ROC-AUC score.")
    
    st.markdown('<p class="sub-header">Model Architecture: Optuna + Blending</p>', unsafe_allow_html=True)
    st.markdown("""
    The final frequency model is an optimized ensemble:
    1. **Logistic Regression**: Provides a stable, highly calibrated baseline.
    2. **XGBoost Classifier**: Tuned via Bayesian Optimization (Optuna) to capture complex non-linear relationships.
    3. **Mathematical Blending**: Weights between Logistic Regression and XGBoost were optimized using `scipy.optimize` to maximize the validation ROC-AUC.
    """)
    
    # Emplacements prévus pour les résultats de resultats_etape3/
    path_roc_cal = 'ROC_and_Calibration.png' 
    # Fallback sur le dossier racine si besoin
    if not os.path.exists(path_roc_cal): path_roc_cal = 'ROC_and_Calibration.jpg'
    
    display_image(path_roc_cal, "Figure 3: ROC Curve and Probability Calibration plot on the Validation set")

 
# VUE 4 : MODÈLE DE GRAVITÉ
 
elif step == "4. Claim Severity Model":
    st.markdown('<p class="main-header">Step 4: Claim Severity Model (Regression)</p>', unsafe_allow_html=True)
    st.markdown("Predicting $\mathbb{E}[\text{Amount} \mid \text{Claim} > 0]$. Trained **exclusively** on policyholders who filed a claim.")
    
    st.markdown('<p class="sub-header">Handling the Skewed Distribution</p>', unsafe_allow_html=True)
    st.markdown("""
    To model the heavy right tail of claim amounts, two approaches were ensembled:
    - **Ridge Regression on Log(Target)**: The target is logged to achieve normality. *Crucial Actuarial Step:* When converting back via $\exp()$, Jensen's inequality states that $\mathbb{E}[\exp(X)] \neq \exp(\mathbb{E}[X])$. We must apply the variance correction.
    - **XGBoost with Gamma Objective**: Natively handles right-skewed distributions (`objective: reg:gamma`).
    
    Both models underwent a **Scale Realignment** post-prediction to ensure that $\sum \text{Predicted Severity} = \sum \text{Actual Severity}$ on the training fold.
    """)
    
    path_diag_grav = 'diagnostic_gravite.png'
    if not os.path.exists(path_diag_grav): path_diag_grav = 'diagnostic_gravite.jpg'
        
    display_image(path_diag_grav, "Figure 4: Severity distribution and Actual vs. Predicted scatter plot")

 
# VUE 5 : AUDIT FINANCIER ET SOUMISSION
 
elif step == "5. Financial Audit & Premium":
    st.markdown('<p class="main-header">Step 5: Pure Premium & Financial Audit</p>', unsafe_allow_html=True)
    st.markdown("The moment of truth: combining both models to assess portfolio profitability.")
    
    st.latex(r"\hat{\text{Premium}}_i = \hat{\mathbb{P}}_i(\text{Frequency}) \times \hat{\mathbb{E}}_i(\text{Severity})")
    
    st.markdown('<p class="sub-header">Validation Set Actuarial Audit</p>', unsafe_allow_html=True)
    st.markdown("""
    Before deploying the model to the test set, we simulate the financial reality on our validation set (20% of data). 
    A model is only viable if the **Sum of Collected Premiums $\ge$ Sum of Incurred Claims** (Loss Ratio $\le 100\%$).
    """)
    
    path_audit = 'audit_financier_validation.png'
    if not os.path.exists(path_audit): path_audit = 'audit_financier_validation.png'
        
    display_image(path_audit, "Figure 5: Predicted Premiums Distribution and Global Financial Balance")
    
    st.markdown("---")
    st.markdown("### 🚀 Final Kaggle Submission")
    st.markdown("""
    After validating the financial soundness of the approach:
    1. Both frequency and severity pipelines were retrained on **100% of the training data**.
    2. The models evaluated the 50,000 blind profiles from `test.csv`.
    3. The resulting pure premiums were exported as `submission_kaggle.csv` for scoring on the Kaggle CY Tech 2027 leaderboard.
    """)
    
    st.success("Project Successfully Deployed. 📈")
