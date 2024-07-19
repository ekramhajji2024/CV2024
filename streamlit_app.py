import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, kstest, anderson
import matplotlib.pyplot as plt
# Charger le dataset
data = pd.read_csv('data/gdp_data.csv')

# Visualisation
plt.hist(data['age'], bins=30, edgecolor='black')
plt.title('Histogramme de l\'âge')
plt.xlabel('Âge')
plt.ylabel('Fréquence')
plt.show()

stats.probplot(data['age'], dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()

# Tests statistiques
stat, p = shapiro(data['age'])
print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Les données suivent une distribution normale (ne rejette pas H0)')
else:
    print('Les données ne suivent pas une distribution normale (rejette H0)')

stat, p = kstest(data['age'], 'norm')
print('Kolmogorov-Smirnov Test: Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Les données suivent une distribution normale (ne rejette pas H0)')
else:
    print('Les données ne suivent pas une distribution normale (rejette H0)')

result = anderson(data['age'], dist='norm')
print('Anderson-Darling Test: Statistic: %.3f' % result.statistic)
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
        print('A la significativité %.3f, on ne rejette pas H0 (Les données suivent une distribution normale)' % sl)
    else:
        print('A la significativité %.3f, on rejette H0 (Les données ne suivent pas une distribution normale)' % sl)
        import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Charger le modèle et le scaler
model = joblib.load('heart_disease_model.pkl')
scaler = StandardScaler()

# Charger les données
data = pd.read_csv('Heart_Disease_Prediction.csv')

# Afficher les noms des colonnes pour vérification
st.write("Colonnes du fichier CSV:", data.columns)

# Titre de l'application
st.title("Prédiction du Risque Cardiovasculaire")

# Fonction pour les prédictions
def predict_risk(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    prediction_prob = model.predict_proba(features_scaled)[:, 1]
    return prediction[0], prediction_prob[0]

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
    st.write(f"Prédiction : {'Présence de Maladie' si prediction == 1 else 'Absence de Maladie'}")
    st.write(f"Probabilité de Maladie : {prediction_prob:.2f}")

# Tracer un histogramme de l'âge
if 'Age' in data.columns:
    plt.hist(data['Age'], bins=30, edgecolor='black')
    st.pyplot(plt.gcf())
else:
    st.write("La colonne 'Age' n'existe pas dans le fichier CSV.")

