# ============================================================
#  predict.py — Prédiction avec le modèle KNN Sonar
# ============================================================

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Chemins dynamiques
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "knn_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "data", "raw", "sonar.all-data.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Charger le modèle et le scaler
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Modèle et scaler chargés avec succès !")


def predict_single(features):
    if len(features) != 60:
        raise ValueError(f"❌ Attendu 60 features, reçu {len(features)}")
    X = np.array(features).reshape(1, -1)
    X = scaler.transform(X)
    prediction = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    label      = 'Mine (M)' if prediction == 1 else 'Roche (R)'
    print(f"\n🔍 Résultat de la prédiction :")
    print(f"   → Classe prédite  : {label}")
    print(f"   → Proba Roche     : {proba[0]*100:.2f}%")
    print(f"   → Proba Mine      : {proba[1]*100:.2f}%")
    return label


if __name__ == "__main__":

    print("\n📂 Chargement des données sonar...")
    columns = [f'F{i}' for i in range(1, 61)] + ['Objet']
    df = pd.read_csv(DATA_PATH, names=columns)

    y_true = df['Objet'].map({'M': 1, 'R': 0})
    X_all  = df.drop('Objet', axis=1)

    X_scaled    = scaler.transform(X_all)
    predictions = model.predict(X_scaled)

    # Accuracy
    acc = accuracy_score(y_true, predictions)
    print(f"\n🎯 Accuracy globale : {round(acc * 100, 2)}%")

    # Rapport
    print("\n📊 Rapport de classification :")
    print(classification_report(y_true, predictions, target_names=['Roche (R)', 'Mine (M)']))

    # Matrice de confusion (texte)
    cm = confusion_matrix(y_true, predictions)
    print("🔲 Matrice de confusion :")
    print(f"                Prédit Roche   Prédit Mine")
    print(f"  Réel Roche  :     {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  Réel Mine   :     {cm[1][0]:>5}          {cm[1][1]:>5}")

    # Matrice de confusion (graphique)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Roche (R)', 'Mine (M)'],
                yticklabels=['Roche (R)', 'Mine (M)'],
                linewidths=0.5, linecolor='gray')
    plt.title('Matrice de Confusion — KNN Sonar', fontsize=13, fontweight='bold')
    plt.xlabel('Classe Predite', fontsize=11)
    plt.ylabel('Classe Reelle', fontsize=11)
    plt.tight_layout()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=150)
    print("\n💾 Matrice sauvegardée dans reports/confusion_matrix.png")
    plt.show()

    # Test échantillon fictif
    print("\n🧪 Test avec un échantillon fictif...")
    predict_single([0.5] * 60)