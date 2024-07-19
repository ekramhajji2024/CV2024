
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os
# Charger les données
data = pd.read_csv('data/gdp_data.csv')
csv_path = 'Heart_Disease_Prediction.csv'

# Titre de l'application
st.title("Prédiction du Risque Cardiovasculaire")

le = LabelEncoder()
data['Heart Disease'] = le.fit_transform(data['Heart Disease'])

# Define features and target variable
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)

# Train the model with the best parameters
best_logreg = grid_search.best_estimator_
best_logreg.fit(X_train_scaled, y_train)

y_train_pred = best_logreg.predict(X_train_scaled)
y_test_pred = best_logreg.predict(X_test_scaled)

y_train_pred_prob = best_logreg.predict_proba(X_train_scaled)[:, 1]
y_test_pred_prob = best_logreg.predict_proba(X_test_scaled)[:, 1]

# Evaluate on training set
print("Training Set Evaluation with Tuned Model")
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_test_pred_prob))
print("Accuracy:", accuracy_score(y_test, y_test_pred))

# Fonction pour les prédictions
def predict_risk(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    prediction_prob = model.predict_proba(features_scaled)[:, 1]
    return prediction[0], prediction_prob[0]
    #


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
