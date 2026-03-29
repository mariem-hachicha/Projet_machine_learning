# ============================================================
# 🌐 APP.PY - APPLICATION FLASK
# Interface web pour prédire le segment, churn et revenu d'un client
# ============================================================

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ============================================================
# 📦 CHARGEMENT DES MODÈLES
# On charge tout au démarrage pour ne pas recharger à chaque requête
# ============================================================

# 🔵 Clustering
kmeans         = joblib.load("../models/kmeans.pkl")
pca            = joblib.load("../models/pca.pkl")
scaler_cluster = joblib.load("../models/scaler_cluster.pkl")
cluster_features = joblib.load("../models/cluster_features.pkl")

# 🟢 Classification (Churn)
clf          = joblib.load("../models/churn_model.pkl")
clf_columns  = joblib.load("../models/churn_columns.pkl")
scaler_clf   = joblib.load("../models/scaler_clf.pkl")

# 🟣 Régression (Revenue)
reg          = joblib.load("../models/regression_model.pkl")
reg_columns  = joblib.load("../models/reg_columns.pkl")
scaler_reg   = joblib.load("../models/scaler_reg.pkl")

print("✅ Tous les modèles chargés avec succès")

# ============================================================
# 🧠 INTERPRÉTATION DES CLUSTERS
# Dictionnaire de labels humains pour chaque numéro de cluster
# ============================================================

def interpret_cluster(cluster_id):
    """
    Transforme un numéro de cluster en description humaine.
    Ces labels doivent être mis à jour en fonction des résultats
    réels de votre clustering (cf. cluster_analysis.csv).
    """
    descriptions = {
        0: "🟡 Clients occasionnels",
        1: "🔴 Clients à risque",
        2: "💰 Clients VIP",
        3: "🟡 Clients peu actifs",
        4: "💰 Gros acheteurs",
        5: "🟢 Clients fidèles",
        6: "🟢 Clients réguliers"
    }
    return descriptions.get(cluster_id, f"Segment {cluster_id}")

# ============================================================
# 🏠 ROUTE PRINCIPALE
# GET  → affiche le formulaire vide
# POST → reçoit les données et retourne les prédictions
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        try:
            # =================================================
            # 📥 RÉCUPÉRATION DES DONNÉES DU FORMULAIRE
            # =================================================
            data = {
                "Recency":                  float(request.form["Recency"]),
                "Frequency":                float(request.form["Frequency"]),
                "MonetaryTotal":            float(request.form["MonetaryTotal"]),
                "CustomerTenureDays":       float(request.form["CustomerTenure"]),
                "AvgDaysBetweenPurchases":  float(request.form["AvgDaysBetween"]),
                "TotalTransactions":        float(request.form["TotalTrans"])
            }

            df = pd.DataFrame([data])

            # =================================================
            # 🔵 ÉTAPE 1 : CLUSTERING
            # Aligner les colonnes exactement comme lors de l'entraînement
            # =================================================
            df_cluster = df.reindex(columns=cluster_features, fill_value=0).astype(float)

            # Appliquer le même scaler et PCA qu'à l'entraînement
            X_scaled = scaler_cluster.transform(df_cluster)
            X_pca    = pca.transform(X_scaled)
            cluster  = kmeans.predict(X_pca)[0]

            interpretation = interpret_cluster(cluster)

            # =================================================
            # 🟢 ÉTAPE 2 : PRÉDICTION DU CHURN
            # On ajoute le cluster comme feature supplémentaire
            # puis on aligne sur les colonnes du modèle
            # =================================================
            df_clf = df.copy()
            df_clf["Cluster"] = cluster

            df_clf = pd.get_dummies(df_clf)
            df_clf = df_clf.reindex(columns=clf_columns, fill_value=0)

            # ⚠️ SCALING OBLIGATOIRE (même scaler qu'à l'entraînement)
            X_clf_scaled = scaler_clf.transform(df_clf)

            churn_pred  = clf.predict(X_clf_scaled)[0]
            churn_proba = clf.predict_proba(X_clf_scaled)[0][1]

            # =================================================
            # 🟣 ÉTAPE 3 : PRÉDICTION DU REVENU
            # Même logique : cluster + dummies + reindex + scaling
            # =================================================
            df_reg = df.copy()
            df_reg["Cluster"] = cluster

            df_reg = pd.get_dummies(df_reg)
            df_reg = df_reg.reindex(columns=reg_columns, fill_value=0)

            # ⚠️ SCALING OBLIGATOIRE
            X_reg_scaled = scaler_reg.transform(df_reg)

            revenue = reg.predict(X_reg_scaled)[0]

            # =================================================
            # 📤 ENVOI DES RÉSULTATS AU TEMPLATE HTML
            # =================================================
            return render_template(
                "index.html",
                interpretation = interpretation,
                churn          = int(churn_pred),
                proba          = round(float(churn_proba), 2),
                revenue        = round(float(revenue), 2)
            )

        except Exception as e:
            # En cas d'erreur, afficher le message dans la page
            return render_template("index.html", error=str(e))

    # GET : page vide avec formulaire
    return render_template("index.html")

# ============================================================
# ▶️ LANCEMENT DU SERVEUR
# debug=True → recharge automatiquement en cas de modification
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)