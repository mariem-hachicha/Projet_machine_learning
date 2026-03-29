import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def resume_dataset(df):
    print("=" * 50)
    print("RESUME DATASET")
    print("=" * 50)
    print(f"Lignes   : {df.shape[0]:,}")
    print(f"Colonnes : {df.shape[1]}")
    print(f"Doublons : {df.duplicated().sum()}")
    print(f"NaN      : {df.isnull().sum().sum():,}")
    print(f"Memoire  : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("=" * 50)

def afficher_valeurs_manquantes(df):
    manquantes  = df.isnull().sum()
    pourcentage = (manquantes / len(df)) * 100
    resume = pd.DataFrame({
        "Valeurs_manquantes": manquantes,
        "Pourcentage":        pourcentage.round(2)
    })
    resume = resume[resume["Valeurs_manquantes"] > 0]
    resume = resume.sort_values("Pourcentage", ascending=False)
    if resume.empty:
        print("Aucune valeur manquante.")
    else:
        print(f"{len(resume)} colonnes avec NaN :")
        print(resume.to_string())
    return resume

def detecter_outliers_iqr(df, colonne):
    Q1  = df[colonne].quantile(0.25)
    Q3  = df[colonne].quantile(0.75)
    IQR = Q3 - Q1
    lb  = Q1 - 1.5 * IQR
    ub  = Q3 + 1.5 * IQR
    n   = ((df[colonne] < lb) | (df[colonne] > ub)).sum()
    print(f"{colonne} : Q1={Q1:.2f} Q3={Q3:.2f} IQR={IQR:.2f}")
    print(f"  Bornes [{lb:.2f}, {ub:.2f}] | Outliers : {n} ({n/len(df)*100:.2f}%)")
    return n, lb, ub

def detecter_outliers_toutes_colonnes(df):
    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    resultats = []
    for col in num_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lb  = Q1 - 1.5 * IQR
        ub  = Q3 + 1.5 * IQR
        n   = ((df[col] < lb) | (df[col] > ub)).sum()
        resultats.append({
            "feature":      col,
            "borne_basse":  round(lb, 2),
            "borne_haute":  round(ub, 2),
            "n_outliers":   n,
            "pct_outliers": round(n / len(df) * 100, 2)
        })
    return pd.DataFrame(resultats).sort_values("n_outliers", ascending=False)

def afficher_distribution(df, colonne, bins=30, output_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[colonne].dropna(), bins=bins, kde=True, ax=axes[0], color="#1e88e5")
    axes[0].set_title(f"Distribution : {colonne}")
    sns.boxplot(x=df[colonne].dropna(), ax=axes[1], color="#43a047")
    axes[1].set_title(f"Boxplot : {colonne}")
    plt.tight_layout()
    if output_dir:
        sauvegarder_graphique(f"dist_{colonne}.png", output_dir)
    plt.show()

def afficher_correlation(df, seuil=0.8, output_dir=None):
    df_num = df.select_dtypes(include=np.number)
    corr   = df_num.corr().abs()
    plt.figure(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Matrice de correlation")
    plt.tight_layout()
    if output_dir:
        sauvegarder_graphique("correlation_heatmap.png", output_dir)
    plt.show()
    paires = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] >= seuil:
                paires.append({
                    "Feature_1":   corr.columns[i],
                    "Feature_2":   corr.columns[j],
                    "Correlation": round(corr.iloc[i, j], 3)
                })
    if paires:
        result = pd.DataFrame(paires).sort_values("Correlation", ascending=False)
        print(f"{len(result)} paires avec correlation >= {seuil}")
        print(result.to_string(index=False))
        return result
    return None

def sauvegarder_graphique(nom_fichier, dossier="../reports"):
    os.makedirs(dossier, exist_ok=True)
    chemin = os.path.join(dossier, nom_fichier)
    plt.savefig(chemin, bbox_inches="tight", dpi=150)
    print(f"Graphique sauvegarde : {chemin}")

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

def evaluer_classificateur(model, X_test, y_test, nom_modele="Modele", output_dir=None):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    print(f"\n{nom_modele}")
    print(classification_report(y_test, y_pred, target_names=["Fidele", "Churne"]))
    if y_prob is not None:
        print(f"AUC-ROC : {roc_auc_score(y_test, y_prob):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fidele", "Churne"],
                yticklabels=["Fidele", "Churne"])
    plt.title(f"Confusion Matrix : {nom_modele}")
    plt.tight_layout()
    if output_dir:
        sauvegarder_graphique(f"confusion_{nom_modele}.png", output_dir)
    plt.show()
    return y_pred

from sklearn.metrics import mean_absolute_error, r2_score

def evaluer_regression(model, X_test, y_test, nom_modele="Modele", output_dir=None):
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"\n{nom_modele}")
    print(f"MAE : {mae:.2f}")
    print(f"R2  : {r2:.4f}")
    return y_pred