# 1️⃣ Importer les bibliothèques
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib


# 2️⃣ Charger les données (chemin dynamique)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "sonar.all-data.csv")

columns = [f'F{i}' for i in range(1, 61)] + ['Objet']
df = pd.read_csv(DATA_PATH, names=columns)


# 3️⃣ Transformer la cible (M=1, R=0)
df['Objet'] = df['Objet'].map({'M': 1, 'R': 0})


# 4️⃣ Séparer X et y
X = df.drop('Objet', axis=1)
y = df['Objet']


# 5️⃣ Train / Test split (avec stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y   # IMPORTANT pour garder proportion M/R
)


# 6️⃣ Standardisation (IMPORTANT pour KNN)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)  # fit seulement sur train
X_test = scaler.transform(X_test)        # transform test (pas fit !)


# 7️⃣ Entraîner le modèle
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print("✅ Modèle entraîné avec succès !")


# 8️⃣ Évaluation
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("🎯 Accuracy :", round(accuracy, 4))


# 9️⃣ Sauvegarder le modèle et le scaler
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "knn_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("💾 Modèle et scaler sauvegardés dans /models")