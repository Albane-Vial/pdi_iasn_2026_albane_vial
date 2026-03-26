#  Pipeline de Détection d'Anomalies Médicales (MIMIC-IV)

Ce projet implémente un pipeline complet de traitement de données et d'entraînement de modèles (DeBERTaV3 et LOF) pour la détection d'erreurs de prescription dans la base de données MIMIC-IV.

## Fonctionnalités
* **Nettoyage de données** : Extraction et filtrage depuis PostgreSQL.
* **Génération de datasets** : Split stratifié (Train/Test) et injection d'anomalies synthétiques à différents niveaux de perturbation ($p \in \{0.2, 0.5, 1, 2\}$).
* **Modélisation Multimodale** : Entraînement et test de modèles Transformer (NLP) et Outlier Detection (Statistique).
* **Explicabilité** : Génération de rapports d'audit sur les erreurs de prédiction.
---

## Installation

1. **Cloner le répertoire** :
   ```bash
   git clone https://github.com/votre-repo/mimic_albanevial.git
   cd mimic_albanevial
   ```

2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

---

## Guide d'utilisation

Le script principal `main.py` est piloté par des arguments en ligne de commande.

### 1. Préparation des données
Avant d'entraîner un modèle, vous devez préparer les fichiers Parquet.

* **Nettoyage complet (SQL -> Parquet)** :
    ```bash
    python main.py --clean
    ```
* **Génération des splits et des erreurs de test** :
    ```bash
    python main.py --generate --test_size 0.1
    ```
    *Note : Cette commande crée un dossier `data/` pour le Train et `data/data_test/` pour les benchmarks.*

### 2. Entraînement des modèles (Train)
Vous pouvez entraîner les modèles individuellement ou tous ensemble.

* **Entraîner DeBERTaV3** :
    ```bash
    python main.py --model deberta --train --output_dir "./outputs"
    ```
* **Entraîner LOF (Local Outlier Factor)** :
    ```bash
    python main.py --model lof --train --output_dir "./outputs"
    ```
* **Entraîner les deux modèles** :
    ```bash
    python main.py --model all --train --output_dir "./outputs"
    ```

### 3. Évaluation et Inférence (Test)
Cette étape évalue le modèle sur les 4 niveaux de perturbation générés précédemment.

```bash
python main.py --model all --test --data_dir "./data"
```

### 4. Audit et Explicabilité
Pour analyser pourquoi le modèle a échoué sur certaines lignes (spécifique à DeBERTa) :

```bash
python main.py --model deberta --explain
```

---

## Architecture du Projet

Le projet suit une architecture modulaire, divisée en scripts de préparation de données et en modules spécifiques pour chaque algorithme d'apprentissage.

```text
.
├── main.py                   # Point d'entrée principal (CLI). Orchestre l'extraction, la génération, l'entraînement, l'inférence et l'explicabilité pour les modèles.
├── nettoyage_donnees.py      # Pipeline d'extraction PostgreSQL, filtrage, jointure (patients, admissions, biologie, prescriptions) et construction de la `phrase_clinique`.
├── creation_dataset_test.py  # Logique de split et d'injection de perturbations (erreurs unitaires et multiples sur les voies, molécules, unités, sous-dosages, sur-dosages).
├── explore_mimic.py          # Script d'exploration de l'API Kaggle pour lister les fichiers et schémas MIMIC-IV sans tout télécharger.
├── data/                     # Stockage des fichiers de données (.parquet)
│   ├── df_train.parquet      # Dataset d'entraînement sain
│   └── data_test/            # Datasets de test incluant des anomalies à différents niveaux de perturbation (0.2, 0.5, 1, 2)
├── modeles/                  # Implémentations des modèles ML/DL
│   ├── evaluation_modele.py  # Fonctions d'évaluation globale (métriques de classification et rapports)
│   ├── deberta/              # Modèle Transformer DeBERTaV3 (NLP)
│   │   ├── modele_debertaV3.py      # Entraînement et inférence
│   │   └── explicabilite_deberta.py # Explicabilité (SHAP, LIME, IG)
│   ├── isolation_forest/     # Modèle classique (Isolation Forest) avec embeddings BERT
│   │   ├── modele_if.py             # Entraînement et inférence
│   │   └── explicabilite_if.py      # Explicabilité (SHAP, LIME)
│   └── lof/                  # Modèle classique (Local Outlier Factor) avec embeddings BERT
│       └── modele_lof.py            # Entraînement, inférence et réduction PCA
└── modeles_sorties/          # (Créé à l'exécution) Sauvegarde des poids, prédictions et rapports d'évaluation
```

### Flux d'Exécution (Data Flow)
1. **Extraction & Nettoyage** (`nettoyage_donnees.py`) : Connexion à PostgreSQL, extraction conditionnelle et formatage des variables structurées en une phrase textuelle (via des balises comme `[DRUG]`, `[DOSE]`, `[ROUTE]`, etc.).
2. **Génération d'Anomalies** (`creation_dataset_test.py`) : Sur la base de données saines, création de fichiers de tests bruités (erreurs simulées de prescriptions : typographie, mauvaise unité, erreurs de dosage).
3. **Apprentissage & Inférence** (`main.py` -> `modeles/`) : Les modèles (DeBERTa, LOF, ou Isolation Forest) apprennent la représentation de prescriptions "normales" pour détecter celles qui sont aberrantes.
4. **Évaluation & Audit** : Comparaison des prédictions aux anomalies injectées, génération de métriques et utilisation de XAI (LIME, SHAP, etc.) pour expliquer les décisions du modèle lorsqu'il se trompe.

---
