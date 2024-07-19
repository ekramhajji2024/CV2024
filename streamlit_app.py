import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os
# Charger les données
data = pd.read_csv('data/gdp_data.csv')
csv_path = 'Heart_Disease_Prediction.csv'
#scal
scaler_path = 'scaler.pkl'

try:
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        st.write("Scaler loaded successfully.")
    else:
        st.error(f"Scaler file '{scaler_path}' not found. Please check the path and try again.")
except FileNotFoundError:
    st.error(f"File '{scaler_path}' not found. Please ensure the file exists and the path is correct.")
except Exception as e:
    st.error(f"An error occurred while loading the scaler: {e}")

# Titre de l'application
st.title("Prédiction du Risque Cardiovasculaire")

# Fonction pour les prédictions
def predict_risk(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    prediction_prob = model.predict_proba(features_scaled)[:, 1]
    return prediction[0], prediction_prob[0]
    #
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(logreg, 'heart_disease_model.pkl')

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
