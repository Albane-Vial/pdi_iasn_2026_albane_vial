# mimic_albanevial

#  Pipeline de Détection d'Anomalies Médicales (MIMIC-IV)

Ce projet implémente un pipeline complet de traitement de données et d'entraînement de modèles (DeBERTaV3 et LOF) pour la détection d'erreurs de prescription dans la base de données MIMIC-IV.

## Fonctionnalités
* **Nettoyage de données** : Extraction et filtrage depuis PostgreSQL.
* **Génération de datasets** : Split stratifié (Train/Test) et injection d'anomalies synthétiques à différents niveaux de perturbation ($p \in \{0.2, 0.5, 1, 2\}$).
* **Modélisation Multimodale** : Entraînement et test de modèles Transformer (NLP) et Outlier Detection (Statistique).
* **Explicabilité** : Génération de rapports d'audit sur les erreurs de prédiction.
---

## 🛠 Installation

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

```text
.
├── main.py                 # Point d'entrée principal (CLI)
├── creation_dataset_test.py # Logique de split et perturbation
├── nettoyage_donnees.py     # Fonctions de nettoyage et NLP
├── data/                    # Stockage des fichiers .parquet
│   └── data_test/           # Datasets de test (p0_2, p0_5, etc.)
└── modeles/                 # Sorties (Poids du modèle, Logs, Eval)
    ├── deberta/
    └── lof/
```

---

## ⚙️ Paramètres avancés

| Argument | Type | Défaut | Description |
| :--- | :--- | :--- | :--- |
| `--perturb_dose` | float | `0.3` | Écart-type pour les erreurs de dose. |
| `--error_ratio` | float | `0.30` | % d'anomalies dans le set de test. |
| `--output_dir` | str | `modeles_sorties` | Dossier de sauvegarde des modèles. |

---

## ✉️ Contact
**Auteur** : Albane Vial  
**Projet** : Analyse de risque de prescription - MIMIC-IV

---
