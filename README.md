# Projet Machine Learning - Analyse Comportementale Clientèle Retail

## 📋 Description

Ce projet vise à analyser le comportement des clients d'une entreprise e-commerce de cadeaux pour :
- **Personnaliser** les stratégies marketing
- **Réduire** le taux de départ des clients (churn)
- **Optimiser** le chiffre d'affaires

Le dataset contient **4 372 clients** avec **52 features** issues de transactions réelles.

---

## 📁 Structure du Projet

```
projet_ml_retail/
├── data/
│   ├── raw/                          # Données brutes originales
│   │   └── retail_customers_COMPLETE_CATEGORICAL.csv
│   ├── processed/                    # Données nettoyées
│   │   └── retail_customers_CLEANED.csv
│   └── train_test/                   # Données splitées
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── notebooks/                        # Notebooks Jupyter (prototypage)
├── src/                              # Scripts Python (production)
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
├── models/                           # Modèles sauvegardés (.pkl, .joblib)
├── app/                              # Application web (Flask)
├── reports/                          # Rapports et visualisations
├── requirements.txt                  # Dépendances
├── README.md                         # Documentation
└── .gitignore
```

---

## 🚀 Installation

### 1. Créer l'environnement virtuel

```bash
# Création
python -m venv venv

# Activation (Windows)
venv\Scripts\activate

# Activation (Mac/Linux)
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Générer le fichier requirements.txt (si modifications)

```bash
pip freeze > requirements.txt
```

---

## 📊 Données

### Données Originales
- **Fichier** : `retail_customers_COMPLETE_CATEGORICAL.csv`
- **Lignes** : 4 372 clients
- **Colonnes** : 52 features + 1 target (Churn)

### Données Nettoyées
- **Fichier** : `retail_customers_CLEANED.csv`
- **Lignes** : 4 372 clients
- **Colonnes** : 105 features + 1 target (Churn)
- **Split** : 80% Train (3 497) / 20% Test (875)

---

## 🔧 Pipeline de Nettoyage

### Étape 1 : Imputation des Valeurs Manquantes

| Feature | Méthode | Valeur | Description |
|---------|---------|--------|-------------|
| `Age` | Médiane | 49.0 | 1 311 valeurs manquantes |
| `AvgDaysBetweenPurchases` | Médiane | 1.15 | 79 valeurs manquantes |
| `SupportTicketsCount` | Médiane | 2.0 | Outliers -1, 999 → NaN |
| `SatisfactionScore` | Mode | 5.0 | Outliers -1, 99 → NaN |

### Étape 2 : Parsing des Dates

- **Feature** : `RegistrationDate`
- **Formats traités** : `"17/07/10"`, `"2010-10-04"`, `"10/18/2010"`
- **Nouvelles features créées** :
  - `RegYear` : Année d'inscription
  - `RegMonth` : Mois d'inscription
  - `RegDay` : Jour d'inscription
  - `RegWeekday` : Jour de la semaine (0=Lundi, 6=Dimanche)

### Étape 3 : Suppression des Features Inutiles

| Feature | Raison |
|---------|--------|
| `NewsletterSubscribed` | Valeur constante (toujours "Yes") |
| `LastLoginIP` | Non pertinent pour le ML |
| `CustomerID` | Identifiant unique |

### Étape 4 : Traitement des Outliers (Méthode IQR)

Les outliers ont été limités aux bornes IQR (Q1 - 1.5×IQR, Q3 + 1.5×IQR) pour :
- `MonetaryTotal`, `MonetaryAvg`, `MonetaryStd`
- `MonetaryMin`, `MonetaryMax`
- `TotalQuantity`, `AvgQuantityPerTransaction`
- `Age`

### Étape 5 : Gestion de la Multicolinéarité

| Feature Supprimée | Corrélée avec | Corrélation |
|-------------------|---------------|-------------|
| `MonetaryAvg` | `MonetaryTotal` | Élevée |
| `UniqueInvoices` | `Frequency` | 1.0 |
| `UniqueDescriptions` | `UniqueProducts` | 1.0 |
| `MinQuantity` | `MaxQuantity` | 0.961 |
| `AvgProductsPerTransaction` | `AvgLinesPerInvoice` | 0.963 |
| `NegativeQuantityCount` | `CancelledTransactions` | 1.0 |

### Étape 6 : Encodage des Variables Catégorielles

#### Encodage Ordinal (5 variables)

| Feature | Mapping |
|---------|---------|
| `SpendingCategory` | Low=0, Medium=1, High=2, VIP=3 |
| `LoyaltyLevel` | Nouveau=0, Jeune=1, Établi=2, Ancien=3 |
| `ChurnRiskCategory` | Faible=0, Moyen=1, Élevé=2, Critique=3 |
| `AgeCategory` | Inconnu=0, 18-24=1, ..., 65+=6 |
| `BasketSizeCategory` | Petit=0, Moyen=1, Grand=2 |

#### One-Hot Encoding (10 variables → 70 colonnes)

- `RFMSegment` (3 colonnes)
- `CustomerType` (4 colonnes)
- `FavoriteSeason` (3 colonnes)
- `PreferredTimeOfDay` (3 colonnes)
- `Region` (12 colonnes)
- `WeekendPreference` (2 colonnes)
- `ProductDiversity` (2 colonnes)
- `Gender` (2 colonnes)
- `AccountStatus` (3 colonnes)
- `Country` (36 colonnes)

### Étape 7 : Scaling

- **Méthode** : StandardScaler
- **Features** : 35 variables numériques
- **Transformation** : Moyenne = 0, Écart-type = 1

### Étape 8 : Train/Test Split

- **Ratio** : 80% Train / 20% Test
- **Méthode** : Stratifié sur la target `Churn`
- **Random State** : 42 (reproductibilité)

---

## 📈 Distribution de la Target

| Classe | Description | Proportion |
|--------|-------------|------------|
| 0 | Client fidèle | 66.7% |
| 1 | Client parti (Churn) | 33.3% |

---

## 💻 Utilisation

### Charger les données nettoyées

```python
import pandas as pd

# Dataset complet
 df = pd.read_csv('data/processed/retail_customers_CLEANED.csv')

# Train/Test split
X_train = pd.read_csv('data/train_test/X_train.csv')
X_test = pd.read_csv('data/train_test/X_test.csv')
y_train = pd.read_csv('data/train_test/y_train.csv')
y_test = pd.read_csv('data/train_test/y_test.csv')
```

### Entraîner un modèle

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Initialiser le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner
model.fit(X_train, y_train)

# Prédire
y_pred = model.predict(X_test)

# Évaluer
print(classification_report(y_test, y_pred))
```

---

## 🔬 Features Principales

### Features Numériques (35)
- `Recency` : Jours depuis le dernier achat
- `Frequency` : Nombre de commandes
- `MonetaryTotal` : Somme totale dépensée
- `TotalQuantity` : Quantité totale d'articles
- `Age` : Âge estimé du client
- `CustomerTenureDays` : Durée de la relation client
- ... et 29 autres

### Features Catégorielles Encodées (70)
- Segments RFM, Types de clients, Saisons préférées
- Régions, Pays, Statut du compte
- ... et d'autres

---

## 📚 Dépendances Principales

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## 🎯 Objectifs Pédagogiques

Ce projet couvre l'ensemble de la chaîne de traitement en data science :

| Compétence | Description |
|------------|-------------|
| **Exploration** | Analyser la qualité et la structure des données |
| **Préparation** | Nettoyer, encoder et normaliser les features |
| **Transformation** | Réduire la dimension via ACP (optionnel) |
| **Modélisation** | Appliquer clustering, classification et régression |
| **Évaluation** | Interpréter les résultats et proposer des recommandations |
| **Déploiement** | Créer une interface utilisateur avec Flask |

---

## 📝 Notes Importantes

1. **Data Leakage** : Le scaling est appliqué APRÈS le split (fit sur train, transform sur test)
2. **Imputation** : Utilisation de la médiane/mode (pas de KNN comme demandé)
3. **Reproductibilité** : Random state fixé à 42 pour tous les modèles
4. **Stratification** : Le split préserve la distribution de la target Churn

---

## 👤 Auteur

**Fadoua Drira**  
Module Machine Learning - GI2  
Année Universitaire : 2025-2026

---

## 📄 Licence

Ce projet est destiné à des fins pédagogiques