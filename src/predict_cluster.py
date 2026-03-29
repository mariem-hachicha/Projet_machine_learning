# ============================================================
# 🧹 PREPROCESSING - NETTOYAGE ET PRÉPARATION DES DONNÉES
# Projet : Analyse Comportementale Clientèle Retail
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

# ============================================================
# 📥 CHARGEMENT DES DONNÉES
# ============================================================

print("=" * 60)
print("🧹 PREPROCESSING - NETTOYAGE DES DONNÉES")
print("=" * 60)

# ✅ chemin corrigé (TRÈS IMPORTANT)
df = pd.read_csv("../data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")

print(f"\n📊 Dataset brut chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# ============================================================
# 🔍 1. RÉSUMÉ INITIAL
# ============================================================

print("\n📋 Aperçu des données :")
print(df.head(3))

print(f"\n⚠️ Valeurs manquantes : {df.isnull().sum().sum()}")
print(f"🔁 Doublons : {df.duplicated().sum()}")

# ============================================================
# 🗑️ 2. SUPPRESSION DES COLONNES INUTILES
# ============================================================

cols_to_drop = []

if "NewsletterSubscribed" in df.columns and df["NewsletterSubscribed"].nunique() == 1:
    cols_to_drop.append("NewsletterSubscribed")

if "CustomerID" in df.columns:
    cols_to_drop.append("CustomerID")

if "LastLoginIP" in df.columns:
    cols_to_drop.append("LastLoginIP")

df = df.drop(columns=cols_to_drop, errors='ignore')
print(f"\n✅ Colonnes restantes : {df.shape[1]}")

# ============================================================
# 📅 3. TRAITEMENT DATE
# ============================================================

if "RegistrationDate" in df.columns:
    df["RegistrationDate"] = pd.to_datetime(df["RegistrationDate"], dayfirst=True, errors="coerce")

    df["RegYear"] = df["RegistrationDate"].dt.year
    df["RegMonth"] = df["RegistrationDate"].dt.month
    df["RegWeekday"] = df["RegistrationDate"].dt.weekday

    df.drop(columns=["RegistrationDate"], inplace=True)
    print("✅ Date transformée")

# ============================================================
# 🔧 4. VALEURS ABERRANTES
# ============================================================

if "SupportTicketsCount" in df.columns:
    df["SupportTicketsCount"].replace([-1, 999], np.nan, inplace=True)

if "SatisfactionScore" in df.columns:
    df["SatisfactionScore"].replace([-1, 0, 99], np.nan, inplace=True)

print("✅ Valeurs aberrantes corrigées")

# ============================================================
# 🩹 5. IMPUTATION
# ============================================================

num_cols = df.select_dtypes(include=np.number).columns.tolist()
if "Churn" in num_cols:
    num_cols.remove("Churn")

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

cat_cols = df.select_dtypes(include="object").columns.tolist()

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("✅ Imputation terminée")

# ============================================================
# 🏷️ 6. ENCODAGE
# ============================================================

ordinal_mappings = {
    "AgeCategory": {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65+": 5},
    "SpendingCategory": {"Low": 0, "Medium": 1, "High": 2, "VIP": 3},
    "LoyaltyLevel": {"Nouveau": 0, "Jeune": 1, "Établi": 2, "Ancien": 3},
}

for col, mapping in ordinal_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping).fillna(-1)

nominal_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

print("✅ Encodage terminé")

# ============================================================
# 💡 7. FEATURE ENGINEERING
# ============================================================

if "MonetaryTotal" in df.columns and "Recency" in df.columns:
    df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)

if "Frequency" in df.columns and "MonetaryTotal" in df.columns:
    df["AvgBasketValue"] = df["MonetaryTotal"] / (df["Frequency"] + 1)

print("✅ Features créées")

# ============================================================
# ✂️ 8. TRAIN / TEST
# ============================================================

if "Churn" not in df.columns:
    raise ValueError("❌ Colonne Churn introuvable")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}")
print(f"Test : {X_test.shape}")

# ============================================================
# 💾 9. SAUVEGARDE
# ============================================================

os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../data/train_test", exist_ok=True)

df.to_csv("../data/processed/cleaned.csv", index=False)

X_train.to_csv("../data/train_test/X_train.csv", index=False)
X_test.to_csv("../data/train_test/X_test.csv", index=False)
y_train.to_csv("../data/train_test/y_train.csv", index=False)
y_test.to_csv("../data/train_test/y_test.csv", index=False)

print("\n✅ Données sauvegardées")
print("🚀 PREPROCESSING TERMINÉ AVEC SUCCÈS !")