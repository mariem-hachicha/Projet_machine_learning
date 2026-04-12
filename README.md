# Projet Machine Learning - Analyse Comportementale Clientèle Retail

## 📋 Présentation

Ce projet analyse le comportement client pour une entreprise de retail afin de :
- personnaliser les campagnes marketing,
- détecter les clients à risque de churn,
- estimer le revenu futur des clients.

Le jeu de données principal contient environ **4 372 clients** et des variables comportementales, transactionnelles et démographiques.

---

## 📁 Structure du projet

```
Projet_machine_learning/
├── app/
│   ├── app1.py
│   └── templates/
├── data/
│   ├── raw/
│   │   └── retail_customers_COMPLETE_CATEGORICAL.csv
│   ├── processed/
│   │   ├── cleaned.csv
│   │   └── customers_segmented.csv
│   └── train_test/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/
│   ├── churn_columns.pkl
│   ├── churn_model.pkl
│   ├── cluster_features.pkl
│   ├── kmeans.pkl
│   ├── pca.pkl
│   ├── regression_model.pkl
│   ├── reg_columns.pkl
│   ├── scaler_clf.pkl
│   ├── scaler_cluster.pkl
│   └── scaler_reg.pkl
├── notebooks/
├── reports/
├── src/
│   ├── Generate additional plots · PY
│   ├── generate_reports.py
│   ├── predict.py
│   ├── predict_cluster.py
│   ├── train_model.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1. Créer et activer l'environnement virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 🧰 Scripts principaux

### `src/predict_cluster.py`
- prétraitement du dataset brut
- suppression des colonnes inutiles
- imputation des valeurs manquantes
- encodage des variables catégorielles
- création de nouvelles features
- sauvegarde de `data/processed/cleaned.csv`
- génération du split `data/train_test/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

### `src/train_model.py`
- chargement des données d'entraînement
- clustering avec KMeans
- entraînement d'un classifieur `RandomForestClassifier` pour le churn
- entraînement d'un régressseur `RandomForestRegressor` pour le revenu
- sauvegarde des modèles et scalers dans `models/`

### `src/predict.py`
- charge les artefacts sauvegardés
- réalise une prédiction de cluster, churn et revenu pour un client exemple

### `src/generate_reports.py`
- génère des graphiques et rapports
- sauvegarde les images dans `reports/`

### `src/utils.py`
- outils d'analyse exploratoire
- détection des outliers
- visualisation des distributions et corrélations
- métriques de classification

### `app/app1.py`
- application Flask
- interface web pour prédire le segment client, le churn et le revenu estimé

---

## 📦 Données

### Données brutes
- `data/raw/retail_customers_COMPLETE_CATEGORICAL.csv`

### Données transformées
- `data/processed/cleaned.csv`
- `data/processed/customers_segmented.csv`

### Split train/test
- `data/train_test/X_train.csv`
- `data/train_test/X_test.csv`
- `data/train_test/y_train.csv`
- `data/train_test/y_test.csv`

---

## 🔧 Exécution

### 1. Prétraiter les données

```bash
cd src
python predict_cluster.py
```

### 2. Entraîner les modèles

```bash
cd src
python train_model.py
```

### 3. Tester une prédiction locale

```bash
cd src
python predict.py
```

### 4. Générer les rapports

```bash
cd src
python generate_reports.py
```

### 5. Lancer l'application Flask

```bash
cd app
python app1.py
```

---

## 💡 Remarques

- Les scripts `src/train_model.py`, `src/predict.py` et `src/generate_reports.py` doivent être exécutés depuis `src/`.
- L'application Flask attend les modèles dans `../models/` et les templates dans `app/templates/`.
- Après modification des dépendances, régénérer `requirements.txt` avec `pip freeze > requirements.txt`.
