import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    classification_report,
    confusion_matrix
)

def evaluer_detection_binaire(df, col_vrai='label_vrai', col_pred='label_pred'):
    """
    Calcule les métriques de détection globale (classification binaire sain vs erroné).

    Évalue la capacité du modèle à identifier la présence d'au moins une erreur, 
    indépendamment de la qualification de cette erreur.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les prédictions et les vérités terrain.
        col_vrai (str, optional): Nom de la colonne contenant le label réel (0 ou 1). Par défaut 'label_vrai'.
        col_pred (str, optional): Nom de la colonne contenant le label prédit (0 ou 1). Par défaut 'label_pred'.

    Returns:
        dict: Dictionnaire contenant les métriques suivantes :
            - Accuracy (float)
            - Precision (float)
            - Recall (float)
            - F1_Score (float)
            - Matrice_Confusion (list of lists)
    """
    y_true = df[col_vrai].astype(int)
    y_pred = df[col_pred].astype(int)
    
    return {
        'Accuracy': (y_true == y_pred).mean(),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1_Score': f1_score(y_true, y_pred, zero_division=0),
        'Matrice_Confusion': confusion_matrix(y_true, y_pred).tolist()
    }

def preparer_labels_multilabel(df, col_vrai, col_pred):
    """
    Transforme les chaînes d'erreurs textuelles en matrices binaires (One-Hot Encoding).

    Convertit un format concaténé (ex: "drug|dosage") en vecteurs numériques denses
    exploitables par les algorithmes de scikit-learn.

    Args:
        df (pd.DataFrame): Le DataFrame source.
        col_vrai (str): Nom de la colonne contenant les types d'erreurs réels.
        col_pred (str): Nom de la colonne contenant les types d'erreurs prédits.

    Returns:
        tuple: Un tuple contenant trois éléments :
            - y_true_bin (np.ndarray): Matrice binaire des vérités terrain.
            - y_pred_bin (np.ndarray): Matrice binaire des prédictions.
            - classes (np.ndarray): Liste des noms des classes détectées par le binarizer.
    """
    y_true_list = df[col_vrai].apply(lambda x: str(x).split('|') if pd.notna(x) and str(x) != "none" else []).tolist()
    y_pred_list = df[col_pred].apply(lambda x: str(x).split('|') if pd.notna(x) and str(x) != "none" else []).tolist()

    mlb = MultiLabelBinarizer()
    mlb.fit(y_true_list + y_pred_list)
    return mlb.transform(y_true_list), mlb.transform(y_pred_list), mlb.classes_

def evaluer_caracterisation_multilabel(df, col_vrai='error_types', col_pred='error_types_pred'):
    """
    Produit un rapport de classification multilabel par catégorie d'erreur.

    Utilise scikit-learn pour générer un rapport standard (Precision, Recall, F1)
    pour chaque classe spécifique d'erreur identifiée.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les prédictions textuelles.
        col_vrai (str, optional): Colonne des erreurs réelles. Par défaut 'error_types'.
        col_pred (str, optional): Colonne des erreurs prédites. Par défaut 'error_types_pred'.

    Returns:
        dict: Dictionnaire contenant la clé 'Rapport_Detaille' avec le texte formaté 
              par scikit-learn, ou une clé 'Erreur' si aucune classe n'est trouvée.
    """
    y_true_bin, y_pred_bin, classes = preparer_labels_multilabel(df, col_vrai, col_pred)

    if len(classes) == 0:
        return {"Erreur": "Aucune classe à évaluer."}
    return {'Rapport_Detaille': classification_report(y_true_bin, y_pred_bin, target_names=classes, zero_division=0)}

def evaluer_qualite_caracterisation_tp(df, col_vrai='error_types', col_pred='error_types_pred'):
    """
    Évalue la précision de caractérisation sur les Vrais Positifs.

    Isole les lignes où le modèle a correctement détecté une erreur (label_pred == 1) 
    et évalue sa capacité à nommer exactement les bonnes catégories d'erreurs, 
    indépendamment de l'ordre d'apparition des termes.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        col_vrai (str, optional): Colonne des erreurs réelles. Par défaut 'error_types'.
        col_pred (str, optional): Colonne des erreurs prédites. Par défaut 'error_types_pred'.

    Returns:
        dict: Dictionnaire stratifié par le nombre d'erreurs initiales.
              Chaque clé (ex: "1_erreur(s)") contient les métriques : Volume, Match_Strict, 
              Match_Partiel, et Rappel_Moyen.
    """
    df_tp = df[(df['nb_errors'] > 0) & (df['label_pred'] == 1)].copy()
    
    if df_tp.empty:
        return {}

    df_tp['set_vrai'] = df_tp[col_vrai].apply(lambda x: set(str(x).split('|')) if pd.notna(x) and str(x) != 'none' else set())
    df_tp['set_pred'] = df_tp[col_pred].apply(lambda x: set(str(x).split('|')) if pd.notna(x) and str(x) != 'none' else set())

    df_tp['match_strict'] = df_tp['set_vrai'] == df_tp['set_pred']
    df_tp['match_partiel'] = df_tp.apply(lambda row: len(row['set_vrai'].intersection(row['set_pred'])) > 0, axis=1)
    df_tp['rappel_ligne'] = df_tp.apply(
        lambda row: len(row['set_vrai'].intersection(row['set_pred'])) / len(row['set_vrai']) if len(row['set_vrai']) > 0 else 0, 
        axis=1
    )

    resultats = {}
    for nb in sorted(df_tp['nb_errors'].unique()):
        df_subset = df_tp[df_tp['nb_errors'] == nb]
        resultats[f"{nb}_erreur(s)"] = {
            'Volume': int(len(df_subset)),
            'Match_Strict': round(float(df_subset['match_strict'].mean()), 4),
            'Match_Partiel': round(float(df_subset['match_partiel'].mean()), 4),
            'Rappel_Moyen': round(float(df_subset['rappel_ligne'].mean()), 4)
        }
    return resultats

def analyser_faiblesses_par_type_erreur(df, col_vrai='error_types', col_pred='error_types_pred', col_nb='nb_errors'):
    """
    Identifie les types d'erreurs les plus difficiles à détecter..

    Décompose les erreurs multiples en lignes distinctes et calcule le taux de 
    détection spécifique pour chaque catégorie d'erreur, stratifié par la charge 
    d'erreurs initiale dans la prescription.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        col_vrai (str, optional): Colonne des erreurs réelles. Par défaut 'error_types'.
        col_pred (str, optional): Colonne des erreurs prédites. Par défaut 'error_types_pred'.
        col_nb (str, optional): Colonne du nombre d'erreurs réelles. Par défaut 'nb_errors'.

    Returns:
        pd.DataFrame: DataFrame contenant les colonnes Nb_Erreurs_Initiales, Type_Erreur, 
                      Total_Cas_Reels, Total_Detectes, et Taux_Detection (%).
                      Trié par difficulté (taux de détection croissant).
    """
    df_err = df[df[col_nb] > 0].copy()
    if df_err.empty:
        return pd.DataFrame()

    df_err['Type_Erreur'] = df_err[col_vrai].apply(lambda x: str(x).split('|') if pd.notna(x) and str(x) != 'none' else [])
    df_exploded = df_err.explode('Type_Erreur')
    df_exploded = df_exploded[df_exploded['Type_Erreur'].notna()]

    df_exploded['Est_Trouve'] = df_exploded.apply(
        lambda row: row['Type_Erreur'] in str(row[col_pred]).split('|'), axis=1
    ).astype(int)

    resume = df_exploded.groupby([col_nb, 'Type_Erreur']).agg(
        Total_Cas_Reels=('Est_Trouve', 'count'),
        Total_Detectes=('Est_Trouve', 'sum')
    ).reset_index()
    
    resume['Taux_Detection (%)'] = (resume['Total_Detectes'] / resume['Total_Cas_Reels'] * 100).round(2)
    return resume.sort_values(by=[col_nb, 'Taux_Detection (%)'])
def analyser_hallucinations_autre(df, col_pred='error_types_pred'):
    """
    Calcule la fréquence à laquelle le modèle hallucine une erreur sur une balise saine.
    
    Puisque la classe 'autre' n'existe jamais dans la vérité terrain, chaque 
    prédiction contenant 'autre' constitue une erreur stricte du modèle (Faux Positif).
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les prédictions.
        col_pred (str): Nom de la colonne contenant les types d'erreurs prédits.
        
    Returns:
        dict: Volume absolu et proportion des prescriptions impactées par cette hallucination.
    """
    if df.empty or col_pred not in df.columns:
        return {"Volume_Hallucinations_Autre": 0, "Proportion_Sur_Total_Test (%)": 0.0}

    masque_autre = df[col_pred].apply(lambda x: 'autre' in str(x).split('|') if pd.notna(x) else False)
    
    volume_autre = masque_autre.sum()
    proportion = (volume_autre / len(df) * 100) if len(df) > 0 else 0.0
    
    return {
        "Volume_Hallucinations_Autre": int(volume_autre),
        "Proportion_Sur_Total_Test (%)": round(float(proportion), 2)
    }

def evaluer_pipeline_complet(df):
    """
    Exécute l'intégralité des modules d'évaluation et consolide les résultats.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les prédictions et vérités.

    Returns:
        dict: Dictionnaire consolidé contenant toutes les métriques d'évaluation.
    """
    return {
        'Detection_Binaire': evaluer_detection_binaire(df),
        'Caracterisation_Multilabel': evaluer_caracterisation_multilabel(df),
        'Caracterisation_TP_Strict': evaluer_qualite_caracterisation_tp(df),
        'Faiblesses_Analyse': analyser_faiblesses_par_type_erreur(df),
        'Analyse_Hallucinations': analyser_hallucinations_autre(df)
    }

def exporter_resultat(nom_modele, pertub_value, df_predictions, rapport_evaluation, dossier_pred, dossier_eval):
    """
    Centralise, formate et archive les résultats complets de l'expérience sur disque.

    Args:
        nom_modele (str): Identifiant du modèle.
        pertub_value (float): Niveau de perturbation évalué.
        df_predictions (pd.DataFrame): Données avec prédictions.
        rapport_evaluation (dict): Rapport des métriques généré.
        dossier_pred (Path ou str): Répertoire de sauvegarde des prédictions.
        dossier_eval (Path ou str): Répertoire de sauvegarde des évaluations.

    Returns:
        None
    """
    suffixe = f"{nom_modele}_p{str(pertub_value).replace('.', '_')}"
    chemin_eval = Path(dossier_eval)
    chemin_pred = Path(dossier_pred)
    chemin_eval.mkdir(parents=True, exist_ok=True)
    chemin_pred.mkdir(parents=True, exist_ok=True)
    
    res_bin = rapport_evaluation['Detection_Binaire']
    res_multi = rapport_evaluation['Caracterisation_Multilabel']
    res_tp = rapport_evaluation['Caracterisation_TP_Strict']
    df_faiblesse = rapport_evaluation['Faiblesses_Analyse']
    res_hallucination = rapport_evaluation['Analyse_Hallucinations']

    contenu = f"=== RAPPORT D'ÉVALUATION : {nom_modele} (Perturbation: {pertub_value}) ===\n\n"
    
    contenu += "1. DÉTECTION BINAIRE GLOBALE (Sain vs Erroné)\n"
    contenu += f"Accuracy  : {res_bin['Accuracy']:.4f}\nPrecision : {res_bin['Precision']:.4f}\n"
    contenu += f"Recall    : {res_bin['Recall']:.4f}\nF1 Score  : {res_bin['F1_Score']:.4f}\n"
    contenu += f"Matrice   : {res_bin['Matrice_Confusion']}\n\n"

    contenu += "2. CARACTÉRISATION MULTILABEL (Rapport Sklearn)\n"
    contenu += res_multi.get('Erreur', res_multi.get('Rapport_Detaille')) + "\n\n"

    contenu += "3. QUALITÉ DE CARACTÉRISATION SUR VRAIS POSITIFS (Sets Strict/Partiel)\n"
    for k, v in res_tp.items():
        contenu += f"- {k} : {v}\n"
    contenu += "\n"

    contenu += "4. ANALYSE DES FAIBLESSES (Taux de détection par classe réelle)\n"
    if isinstance(df_faiblesse, pd.DataFrame) and not df_faiblesse.empty:
        contenu += df_faiblesse.to_string(index=False) + "\n\n"
    else:
        contenu += "Aucune donnée de faiblesse disponible.\n\n"

    contenu += "5. ANALYSE DES HALLUCINATIONS (Erreurs sur contextes cliniques non pertinents)\n"
    contenu += f"Volume de prédictions 'autre' : {res_hallucination['Volume_Hallucinations_Autre']}\n"
    contenu += f"Proportion sur le jeu de test : {res_hallucination['Proportion_Sur_Total_Test (%)']} %\n\n"

    print(contenu)

    fichier_rapport = chemin_eval / f'rapport_evaluation_{suffixe}.txt'
    with open(fichier_rapport, 'w', encoding='utf-8') as f:
        f.write(contenu)
    print(f"-> Rapport sauvegardé : {fichier_rapport}")

    fichier_predictions = chemin_pred / f'predictions_resultats_{suffixe}.parquet'
    df_predictions.to_parquet(fichier_predictions, index=False)
    print(f"-> Prédictions sauvegardées : {fichier_predictions}")