import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc

os.makedirs("../reports", exist_ok=True)

# CHARGEMENT
X_train = pd.read_csv("../data/train_test/X_train.csv")
y_train = pd.read_csv("../data/train_test/y_train.csv").squeeze()
X_test  = pd.read_csv("../data/train_test/X_test.csv")
y_test  = pd.read_csv("../data/train_test/y_test.csv").squeeze()

df = pd.read_csv("../data/processed/customers_segmented.csv")

kmeans       = joblib.load("../models/kmeans.pkl")
pca          = joblib.load("../models/pca.pkl")
scaler       = joblib.load("../models/scaler_cluster.pkl")
clf          = joblib.load("../models/churn_model.pkl")
scaler_clf   = joblib.load("../models/scaler_clf.pkl")
clf_columns  = joblib.load("../models/churn_columns.pkl")
reg          = joblib.load("../models/regression_model.pkl")
scaler_reg   = joblib.load("../models/scaler_reg.pkl")
reg_columns  = joblib.load("../models/reg_columns.pkl")
features_cluster = joblib.load("../models/cluster_features.pkl")

print("Modeles charges")

# ============================================================
# 1. BOXPLOTS OUTLIERS
# ============================================================
cols_plot = ["MonetaryTotal","Frequency","CustomerTenureDays",
             "TotalTransactions","AvgDaysBetweenPurchases"]
cols_plot = [c for c in cols_plot if c in X_train.columns]

fig, axes = plt.subplots(1, len(cols_plot), figsize=(16, 5))
for i, col in enumerate(cols_plot):
    axes[i].boxplot(X_train[col].dropna())
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel("")
plt.suptitle("Boxplots - Detection des valeurs aberrantes", fontsize=13)
plt.tight_layout()
plt.savefig("../reports/boxplots_outliers.png", dpi=150, bbox_inches="tight")
plt.close()
print("boxplots_outliers.png sauvegarde")

# ============================================================
# 2. PCA VARIANCE
# ============================================================
to_drop = [c for c in X_train.columns if any(x in c for x in
           ['ChurnRisk','RFMSegment','Perdu','Satisfaction','Loyalty','AccountStatus','Recency','FirstPurchase'])]
X_clean = X_train.drop(columns=to_drop, errors='ignore')
X_clean = X_clean.fillna(X_clean.mean())
X_clean = pd.get_dummies(X_clean, drop_first=True)

scaler_pca = StandardScaler()
X_pca_all  = scaler_pca.fit_transform(X_clean)

pca_full = PCA()
pca_full.fit(X_pca_all)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_opt  = np.argmax(cumvar >= 0.90) + 1

plt.figure(figsize=(12, 5))
plt.plot(range(1, len(cumvar)+1), cumvar, 'b.-')
plt.axhline(0.90, color='red',    linestyle='--', label='90% variance cible')
plt.axvline(n_opt, color='orange', linestyle=':', label=f'n_components optimal = {n_opt}')
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance cumulee expliquee")
plt.title("ACP - Variance expliquee cumulee")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/pca_variance.png", dpi=150, bbox_inches="tight")
plt.close()
print("pca_variance.png sauvegarde")

# ============================================================
# 3. CLUSTERS 2D
# ============================================================
df_cluster = df[features_cluster].fillna(0)
X_sc  = scaler.transform(df_cluster)
X_pca2 = pca.transform(X_sc)

plt.figure(figsize=(12, 7))
scatter = plt.scatter(X_pca2[:, 0], X_pca2[:, 1],
                      c=df["Cluster"], cmap="tab10", alpha=0.6, s=20)
plt.colorbar(scatter, label="Cluster")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Clusters KMeans (k={df['Cluster'].nunique()}) - Projection ACP")
plt.tight_layout()
plt.savefig("../reports/clusters_2d.png", dpi=150, bbox_inches="tight")
plt.close()
print("clusters_2d.png sauvegarde")

# ============================================================
# 4. CONFUSION MATRIX
# ============================================================
X_test_clean = X_test.copy()
X_test_clean["Churn"] = y_test.values
to_drop2 = [c for c in X_test_clean.columns if any(x in c for x in
            ['ChurnRisk','RFMSegment','Perdu','Satisfaction','Loyalty','AccountStatus','Recency','FirstPurchase'])]
X_test_clean = X_test_clean.drop(columns=to_drop2, errors='ignore')
X_test_clean = pd.get_dummies(X_test_clean, drop_first=True)

y_test2 = X_test_clean["Churn"]
X_test2 = X_test_clean.drop("Churn", axis=1)
X_test2 = X_test2.reindex(columns=clf_columns, fill_value=0)
X_test2_sc = scaler_clf.transform(X_test2)

y_pred  = clf.predict(X_test2_sc)
y_proba = clf.predict_proba(X_test2_sc)[:, 1]

cm = confusion_matrix(y_test2, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fidele","Churne"],
            yticklabels=["Fidele","Churne"])
plt.title("Confusion Matrix - RandomForest Churn")
plt.ylabel("Reel")
plt.xlabel("Predit")
plt.tight_layout()
plt.savefig("../reports/confusion_RandomForest_Churn.png", dpi=150, bbox_inches="tight")
plt.close()
print("confusion_RandomForest_Churn.png sauvegarde")

# ============================================================
# 5. COURBE ROC
# ============================================================
fpr, tpr, _ = roc_curve(y_test2, y_proba)
roc_auc     = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='steelblue', lw=2,
         label=f"RandomForest Churn (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC - RandomForest Churn")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/roc_RandomForest_Churn.png", dpi=150, bbox_inches="tight")
plt.close()
print("roc_RandomForest_Churn.png sauvegarde")

# ============================================================
# 6. REGRESSION REEL VS PREDIT
# ============================================================
X_test_reg = X_test.copy()
X_test_reg = X_test_reg.drop(columns=to_drop2, errors='ignore')
X_test_reg["Churn"] = y_test.values
X_test_reg = pd.get_dummies(X_test_reg, drop_first=True)

if "MonetaryTotal" in X_test_reg.columns:
    y_test_reg = X_test_reg["MonetaryTotal"]
    X_test_reg = X_test_reg.drop("MonetaryTotal", axis=1)
    X_test_reg = X_test_reg.reindex(columns=reg_columns, fill_value=0)
    X_test_reg_sc = scaler_reg.transform(X_test_reg)
    y_pred_reg = reg.predict(X_test_reg_sc)

    plt.figure(figsize=(8, 7))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.5, color='green', s=20)
    max_val = max(y_test_reg.max(), y_pred_reg.max())
    plt.plot([0, max_val],[0, max_val], 'r--', label='Parfait')
    plt.xlabel("Reel (GBP)")
    plt.ylabel("Predit (GBP)")
    plt.title("Regression - Reel vs Predit (MonetaryTotal)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../reports/regression_reel_vs_predit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("regression_reel_vs_predit.png sauvegarde")

print("\nTous les graphiques generes dans reports/")