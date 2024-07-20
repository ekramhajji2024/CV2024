import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Charger les données
data = pd.read_csv('Heart_Disease_Prediction.csv')
csv_path = 'Heart_Disease_Prediction.csv'

# Chemins vers les fichiers du modèle et du scaler
model_path = 'heart_disease_model.pkl'
scaler_path = 'scaler.pkl'
scaler = joblib.load(scaler.pkl)
model = joblib.load(heart_disease_model.pkl)
       
# corr
scaler = None
model = None

try:
    scaler = joblib.load(scaler_path)
    st.write("Scaler loaded successfully.")
except Exception as e:
    st.error(f"Error loading scaler: {e}")

try:
    model = joblib.load(model_path)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
# Obtenir le répertoire du script
script_dir = os.path.dirname(__file__)
# Fonction pour les prédictions
def predict_risk(features):
    try:
        # Vérifiez si le scaler et le modèle sont chargés
        if 'scaler' in locals() and 'model' in locals():
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            prediction_prob = model.predict_proba(features_scaled)[:, 1]
            return prediction[0], prediction_prob[0]
        else:
            st.error("Scaler ou modèle non chargé.")
            return None, None
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None, None

# Titre de l'application
st.title("Prédiction du Risque Cardiovasculaire")


# Fonction pour les prédictions
def predict_risk(features):
    try:
        if 'scaler' in locals() and 'model' in locals():
            # Vérifiez que le nombre de caractéristiques correspond à celui attendu par le scaler
            if len(features) == scaler.n_features_in_:
                features_scaled = scaler.transform([features])
                prediction = model.predict(features_scaled)
                prediction_prob = model.predict_proba(features_scaled)[:, 1]
                return prediction[0], prediction_prob[0]
            else:
                st.error(f"Le nombre de caractéristiques ne correspond pas. Attendu: {scaler.n_features_in_}, Reçu: {len(features)}")
                return None, None
        else:
            st.error("Scaler ou modèle non chargé.")
            return None, None
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None, None
# Créer une interface pour l'utilisateur
st.header("Entrée des caractéristiques")

age = st.number_input("Âge", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sexe", options=[0, 1], format_func=lambda x: "Homme" if x == 1 else "Femme")
cp = st.number_input("Type de douleur thoracique (cp)", min_value=0, max_value=3, value=0)
trestbps = st.number_input("Tension artérielle au repos (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholestérol sérique (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Glycémie à jeun > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
restecg = st.number_input("Résultats électrocardiographiques au repos (restecg)", min_value=0, max_value=2, value=0)
thalach = st.number_input("Fréquence cardiaque maximale atteinte (thalach)", min_value=70, max_value=210, value=150)
exang = st.selectbox("Angine induite par l'exercice (exang)", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
oldpeak = st.number_input("Dépression ST induite par l'exercice par rapport au repos (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.number_input("Pente du segment ST (slope)", min_value=0, max_value=2, value=1)
ca = st.number_input("Nombre de vaisseaux principaux colorés par fluoroscopie (ca)", min_value=0, max_value=4, value=0)
thal = st.number_input("Thalassémie (thal)", min_value=0, max_value=3, value=2)

# Mettre toutes les caractéristiques dans une liste
features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Prédire le risque lorsque l'utilisateur clique sur le bouton
if st.button("Prédire le Risque"):
    prediction, prediction_prob = predict_risk(features)
    st.write(f"Prédiction : {'Présence de Maladie' if prediction == 1 else 'Absence de Maladie'}")
    st.write(f"Probabilité de Maladie : {prediction_prob:.2f}")

# Exécuter le script : streamlit run streamlit_app.py
