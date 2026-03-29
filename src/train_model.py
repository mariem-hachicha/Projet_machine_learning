import pandas as pd
import numpy as np
import os
import joblib
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, mean_absolute_error, r2_score
)

print("TRAIN_MODEL")

# CHARGEMENT
X_train = pd.read_csv("../data/train_test/X_train.csv")
y_train = pd.read_csv("../data/train_test/y_train.csv").squeeze()
df = X_train.copy()
df["Churn"] = y_train.values

# SUPPRIMER COLONNES LEAKAGE
cols_leakage = [c for c in df.columns if
                'ChurnRisk'          in c or
                'RFMSegment'         in c or
                'CustomerType_Perdu' in c or
                'SatisfactionScore'  in c or
                'LoyaltyLevel'       in c or
                'AccountStatus'      in c or
                'Recency'            == c or
                'FirstPurchaseDaysAgo' == c]

df = df.drop(columns=cols_leakage, errors='ignore')
print(f"Colonnes leakage supprimees : {len(cols_leakage)}")
print(f"Dataset : {df.shape[0]} lignes x {df.shape[1]} colonnes")

# CLUSTERING
print("CLUSTERING...")
features_cluster = [
    "Frequency", "MonetaryTotal",
    "CustomerTenureDays", "AvgDaysBetweenPurchases", "TotalTransactions"
]
df_cluster = df[features_cluster].copy().fillna(df[features_cluster].mean())
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(df_cluster)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

best_k, best_score = 2, -1
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    score = silhouette_score(X_pca, km.fit_predict(X_pca))
    print(f"k={k} silhouette={score:.4f}")
    if score > best_score:
        best_score, best_k = score, k

print(f"Meilleur K = {best_k}")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_pca)

# CLASSIFICATION
print("CLASSIFICATION...")
df_clf = df.drop("Cluster", axis=1)
df_clf = pd.get_dummies(df_clf, drop_first=True)
X = df_clf.drop("Churn", axis=1)
y = df_clf["Churn"]
scaler_clf = StandardScaler()
X_scaled_clf = scaler_clf.fit_transform(X)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled_clf, y, test_size=0.2, random_state=42, stratify=y
)

def objective_clf(trial):
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 300),
        max_depth=trial.suggest_int("max_depth", 5, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        class_weight="balanced",
        random_state=42
    )
    return cross_val_score(model, X_train_c, y_train_c, cv=5, scoring="roc_auc").mean()

study_clf = optuna.create_study(direction="maximize")
study_clf.optimize(objective_clf, n_trials=20)
clf = RandomForestClassifier(**study_clf.best_params, class_weight="balanced", random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred = clf.predict(X_test_c)
roc = roc_auc_score(y_test_c, clf.predict_proba(X_test_c)[:, 1])
print(classification_report(y_test_c, y_pred, target_names=["Fidele", "Churne"]))
print(f"ROC AUC : {roc:.4f}")
print(confusion_matrix(y_test_c, y_pred))

# REGRESSION
print("REGRESSION...")
df_reg = df.drop("Cluster", axis=1)
df_reg = pd.get_dummies(df_reg, drop_first=True)
X_reg = df_reg.drop("MonetaryTotal", axis=1)
y_reg = df_reg["MonetaryTotal"]
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

def objective_reg(trial):
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 50, 300),
        max_depth=trial.suggest_int("max_depth", 5, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        random_state=42
    )
    return -cross_val_score(model, X_train_r, y_train_r, cv=5,
                            scoring="neg_mean_absolute_error").mean()

study_reg = optuna.create_study(direction="minimize")
study_reg.optimize(objective_reg, n_trials=20)
reg = RandomForestRegressor(**study_reg.best_params, random_state=42)
reg.fit(X_train_r, y_train_r)
mae = mean_absolute_error(y_test_r, reg.predict(X_test_r))
r2  = r2_score(y_test_r, reg.predict(X_test_r))
print(f"MAE : {mae:.2f}")
print(f"R2  : {r2:.4f}")

# SAUVEGARDE
print("Sauvegarde...")
os.makedirs("../models", exist_ok=True)
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../reports", exist_ok=True)

joblib.dump(kmeans,           "../models/kmeans.pkl")
joblib.dump(pca,              "../models/pca.pkl")
joblib.dump(scaler_cluster,   "../models/scaler_cluster.pkl")
joblib.dump(features_cluster, "../models/cluster_features.pkl")
joblib.dump(clf,              "../models/churn_model.pkl")
joblib.dump(scaler_clf,       "../models/scaler_clf.pkl")
joblib.dump(X.columns,        "../models/churn_columns.pkl")
joblib.dump(reg,              "../models/regression_model.pkl")
joblib.dump(scaler_reg,       "../models/scaler_reg.pkl")
joblib.dump(X_reg.columns,    "../models/reg_columns.pkl")

df.to_csv("../data/processed/customers_segmented.csv", index=False)
print("TRAINING TERMINE !")