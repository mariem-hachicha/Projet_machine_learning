import pandas as pd
import joblib

# CHARGEMENT DES MODELES
kmeans           = joblib.load("../models/kmeans.pkl")
pca              = joblib.load("../models/pca.pkl")
scaler_cluster   = joblib.load("../models/scaler_cluster.pkl")
cluster_features = joblib.load("../models/cluster_features.pkl")

clf          = joblib.load("../models/churn_model.pkl")
clf_columns  = joblib.load("../models/churn_columns.pkl")
scaler_clf   = joblib.load("../models/scaler_clf.pkl")

reg          = joblib.load("../models/regression_model.pkl")
reg_columns  = joblib.load("../models/reg_columns.pkl")
scaler_reg   = joblib.load("../models/scaler_reg.pkl")

print("Modeles charges")

# NOUVEAU CLIENT
new_client = pd.DataFrame([{
    "Frequency":                25,
    "MonetaryTotal":           500,
    "CustomerTenureDays":      300,
    "AvgDaysBetweenPurchases":  20,
    "TotalTransactions":        15
}])

print("Client teste :")
print(new_client)

# CLUSTERING
df_cluster = new_client.reindex(columns=cluster_features, fill_value=0).astype(float)
X_scaled   = scaler_cluster.transform(df_cluster)
X_pca      = pca.transform(X_scaled)
cluster    = kmeans.predict(X_pca)[0]

labels = {0: "Clients stables", 1: "Clients a risque"}
print(f"\nCluster predit  : {cluster}")
print(f"Interpretation  : {labels.get(cluster, 'Segment ' + str(cluster))}")

# CHURN
df_clf = new_client.copy()
df_clf["Cluster"] = cluster
df_clf = pd.get_dummies(df_clf)
df_clf = df_clf.reindex(columns=clf_columns, fill_value=0)
X_clf_scaled  = scaler_clf.transform(df_clf)
churn_pred    = clf.predict(X_clf_scaled)[0]
churn_proba   = clf.predict_proba(X_clf_scaled)[0][1]

print(f"\nChurn           : {churn_pred} ({'Risque' if churn_pred == 1 else 'Stable'})")
print(f"Probabilite     : {churn_proba:.2%}")

# REVENUE
df_reg = new_client.copy()
df_reg["Cluster"] = cluster
df_reg = pd.get_dummies(df_reg)
df_reg = df_reg.reindex(columns=reg_columns, fill_value=0)
X_reg_scaled = scaler_reg.transform(df_reg)
revenue      = reg.predict(X_reg_scaled)[0]

print(f"\nRevenu estime   : {revenue:.2f} GBP")
print("\nPrediction terminee !")