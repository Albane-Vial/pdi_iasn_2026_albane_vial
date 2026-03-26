import pandas as pd
import numpy as np
import torch
import shap
from pathlib import Path
from scipy.special import expit
from lime.lime_text import LimeTextExplainer

from modeles.isolation_forest.modele_if import iterateur_embeddings_tokens

def _pipeline_prediction_boite_noire(textes, modele_if, pca_model, scaler_model, tokenizer, bert_model):
    """
    Fonction interne évaluant un tableau de textes et retournant le score d'anomalie minimal pour chaque phrase.

    Args:
        textes (list ou str): Textes à évaluer.
        modele_if: Modèle Isolation Forest.
        pca_model: Modèle PCA.
        scaler_model: Modèle Scaler.
        tokenizer: Tokenizer BERT.
        bert_model: Modèle BERT.

    Returns:
        np.ndarray: Tableau des scores d'anomalie.
    """
    if isinstance(textes, str):
        textes = [textes]
    elif isinstance(textes, np.ndarray):
        textes = textes.tolist()

    scores_phrases = []
    
    generateur = iterateur_embeddings_tokens(textes, batch_size=32, tokenizer=tokenizer, model=bert_model)
    
    dict_scores = {i: [] for i in range(len(textes))}
    
    for _, batch_emb, batch_ids in generateur:
        if len(batch_emb) == 0:
            continue
            
        emb_scaled = scaler_model.transform(batch_emb)
        emb_reduced = pca_model.transform(emb_scaled)
        scores_tokens = modele_if.decision_function(emb_reduced)
        
        for p_id, score in zip(batch_ids, scores_tokens):
            dict_scores[p_id].append(score)

    for i in range(len(textes)):
        if dict_scores[i]:
            scores_phrases.append(np.min(dict_scores[i]))
        else:
            scores_phrases.append(1.0) 
            
    return np.array(scores_phrases)


def generer_explicabilite_shap_if(df_audit: pd.DataFrame, modele_if, pca_model, scaler_model, tokenizer, bert_model, chemin_export: str):
    """
    Génère les valeurs de Shapley en traitant le pipeline IF comme un modèle agnostique.

    Args:
        df_audit (pd.DataFrame): Données à auditer.
        modele_if: Modèle Isolation Forest.
        pca_model: Modèle PCA.
        scaler_model: Modèle Scaler.
        tokenizer: Tokenizer BERT.
        bert_model: Modèle BERT.
        chemin_export (str): Chemin d'exportation Excel.

    Returns:
        None
    """
    def fonction_prediction_shap(textes):
        return _pipeline_prediction_boite_noire(textes, modele_if, pca_model, scaler_model, tokenizer, bert_model)

    masqueur_texte = shap.maskers.Text(tokenizer)
    explainer_shap = shap.Explainer(fonction_prediction_shap, masqueur_texte)
    
    df_echantillon = df_audit.head(10)
    textes = df_echantillon['phrase_clinique'].astype(str).tolist()
    labels_vrais = df_echantillon['label_vrai'].tolist()
    labels_preds = df_echantillon['label_pred'].tolist()
    
    shap_values = explainer_shap(textes)
    
    df_shap = pd.DataFrame({
        "phrase": textes, 
        "label_vrai": labels_vrais,
        "label_pred": labels_preds,
        "shap_values": [v.values.tolist() for v in shap_values]
    })
    df_shap.to_excel(chemin_export, index=False)


def generer_explicabilite_lime_if(df_audit: pd.DataFrame, modele_if, pca_model, scaler_model, tokenizer, bert_model, chemin_export: str):
    """
    Génère les explications LIME en convertissant les distances en pseudo-probabilités.

    Args:
        df_audit (pd.DataFrame): Données à auditer.
        modele_if: Modèle Isolation Forest.
        pca_model: Modèle PCA.
        scaler_model: Modèle Scaler.
        tokenizer: Tokenizer BERT.
        bert_model: Modèle BERT.
        chemin_export (str): Chemin d'exportation Excel.

    Returns:
        None
    """
    def prediction_probabiliste_lime(textes):
        scores_bruts = _pipeline_prediction_boite_noire(textes, modele_if, pca_model, scaler_model, tokenizer, bert_model)
        

        prob_normal = expit(scores_bruts)
        prob_anomalie = 1.0 - prob_normal
        
        return np.vstack((prob_normal, prob_anomalie)).T

    explainer_lime = LimeTextExplainer(class_names=["Prescription Valide", "Anomalie Détectée"], split_expression=r'\W+')
    
    df_echantillon = df_audit.head(5)
    explications_excel = []

    for idx, row in df_echantillon.iterrows():
        texte = str(row['phrase_clinique'])
        
        exp = explainer_lime.explain_instance(texte, prediction_probabiliste_lime, num_features=10, num_samples=200)
        
        fichier_html = str(chemin_export).replace(".xlsx", f"_{idx}.html")
        exp.save_to_file(fichier_html)
        
        explications_excel.append({
            "phrase": texte,
            "label_vrai": row['label_vrai'],
            "label_pred": row['label_pred'],
            "mots_importants_lime": str(exp.as_list()), 
            "fichier_html_associe": Path(fichier_html).name
        })

    pd.DataFrame(explications_excel).to_excel(chemin_export, index=False)



def extraire_erreurs_pour_explicabilite(chemin_predictions_parquet: str):
    """
    Extrait un échantillon d'erreurs (Faux Positifs et Faux Négatifs) pour l'audit XAI.

    Args:
        chemin_predictions_parquet (str): Chemin vers le fichier des prédictions.

    Returns:
        pd.DataFrame: DataFrame contenant l'échantillon d'audit.
    """
    df = pd.read_parquet(chemin_predictions_parquet, engine='pyarrow')
    
    df_erreurs = df[df['label_vrai'] != df['label_pred']].copy()
    
    df_fp = df_erreurs[df_erreurs['label_pred'] == 1]
    df_fn = df_erreurs[df_erreurs['label_vrai'] == 1]
    
    exemples_fp = df_fp.groupby('error_types').first().reset_index()
    exemples_fn = df_fn.groupby('error_types').first().reset_index()
    
    df_audit = pd.concat([exemples_fp, exemples_fn], ignore_index=True)
    return df_audit