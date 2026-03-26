import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import joblib
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def iterateur_embeddings_tokens(phrases, model_id="bert-base-multilingual-cased", batch_size=16, tokenizer=None, model=None):
    """
    Génère les embeddings contextuels token par token via une approche par lots.

    Args:
        phrases (list of str): Liste des textes cliniques.
        model_id (str): Identifiant du modèle HuggingFace.
        batch_size (int): Nombre de phrases traitées simultanément.
        tokenizer (AutoTokenizer, optional): Tokenizer pré-instancié.
        model (AutoModel, optional): Modèle BERT pré-instancié.

    Yields:
        tuple: (batch_tokens, batch_embeddings, batch_phrase_ids)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    if model is None:
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()

    for i in range(0, len(phrases), batch_size):
        batch = phrases[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        batch_tokens = []
        batch_embeddings = []
        batch_phrase_ids = []
        
        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
            
        for j in range(len(batch)):
            absolute_phrase_index = i + j
            input_ids = inputs['input_ids'][j]
            mask = inputs['attention_mask'][j] == 1
            
            valid_input_ids = input_ids[mask]
            valid_embeddings = token_embeddings[j][mask]
            
            tokens_texte = tokenizer.convert_ids_to_tokens(valid_input_ids)
            special_tokens = tokenizer.all_special_tokens
            
            for mot, emb in zip(tokens_texte, valid_embeddings):
                if mot not in special_tokens:
                    batch_tokens.append(mot)
                    batch_embeddings.append(emb.cpu().numpy())
                    batch_phrase_ids.append(absolute_phrase_index) 
        
        yield batch_tokens, np.vstack(batch_embeddings), np.array(batch_phrase_ids)

def entrainer_pipeline_lof_generateur(phrases_train, n_components=50, n_neighbors=20, batch_size=32, data_dir="temp_data"):
    """
    Entraîne les algorithmes de normalisation, réduction PCA et LOF en optimisant la mémoire.

    Args:
        phrases_train (list of str): Corpus d'entraînement.
        n_components (int): Nombre de dimensions pour l'IPCA.
        n_neighbors (int): Nombre de voisins pour LOF.
        batch_size (int): Taille de lot pour l'inférence.
        data_dir (str): Dossier pour les matrices intermédiaires.

    Returns:
        tuple: (lof, ipca, scaler)
    """
    os.makedirs(data_dir, exist_ok=True)
    chemin_embeddings = os.path.join(data_dir, "embeddings_bruts.npy")
    
    all_embeddings = []
    logger.info("Début de l'extraction GPU vers la RAM temporaire...")
    
    generateur = iterateur_embeddings_tokens(phrases_train, batch_size=batch_size)
    for _, batch_emb, _ in generateur:
        all_embeddings.append(batch_emb)
    
    matrice_brute = np.vstack(all_embeddings)
    np.save(chemin_embeddings, matrice_brute)
    del all_embeddings
    
    data_mmap = np.load(chemin_embeddings, mmap_mode='r')
    logger.info(f"Fichier disque prêt : {data_mmap.shape}. Début de l'entraînement CPU...")
    
    scaler = StandardScaler()
    ipca = IncrementalPCA(n_components=n_components)
    
    chunk_size = 50000
    for i in range(0, data_mmap.shape[0], chunk_size):
        chunk = data_mmap[i:i+chunk_size]
        scaler.partial_fit(chunk)
    
    for i in range(0, data_mmap.shape[0], chunk_size):
        chunk_scaled = scaler.transform(data_mmap[i:i+chunk_size])
        ipca.partial_fit(chunk_scaled)
        
    matrice_reduite = ipca.transform(scaler.transform(data_mmap))
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination='auto')
    lof.fit(matrice_reduite)
    
    return lof, ipca, scaler

def analyser_phrase_par_mot(phrase, lof_model, pca_model, scaler_model, model_id="bert-base-multilingual-cased", tokenizer=None, bert_model=None):
    """
    Analyse unitaire d'une phrase pour déterminer individuellement les mots aberrants.

    Args:
        phrase (str ou list): Phrase à analyser.
        lof_model: Modèle LOF entraîné.
        pca_model: Modèle PCA entraîné.
        scaler_model: Modèle Scaler entraîné.
        model_id (str): Identifiant du modèle linguistique.
        tokenizer: Tokenizer pré-instancié.
        bert_model: Modèle BERT pré-instancié.

    Returns:
        list: Liste de dictionnaires contenant le mot et le statut d'anomalie.
    """
    if isinstance(phrase, str):
        phrase = [phrase]
        
    mots, embeddings, _ = next(iterateur_embeddings_tokens(phrase, model_id=model_id, batch_size=1, tokenizer=tokenizer, model=bert_model))
    
    if len(embeddings) == 0:
        return []
        
    embeddings_scaled = scaler_model.transform(embeddings)
    embeddings_reduced = pca_model.transform(embeddings_scaled)
    
    predictions = lof_model.predict(embeddings_reduced)
    
    resultats = []
    for mot, pred in zip(mots, predictions):
        est_anomalie = 1 if pred == -1 else 0
        resultats.append({"mot": mot, "anomalie": est_anomalie})
        
    return resultats

def sauvegarder_pipeline_lof(model_lof, model_pca, model_scaler, chemin_dossier: str):
    """
    Sérialise le pipeline complet LOF sur le disque.

    Args:
        model_lof: Modèle Local Outlier Factor.
        model_pca: Modèle PCA.
        model_scaler: Modèle Scaler.
        chemin_dossier (str): Dossier de sauvegarde.

    Returns:
        None
    """
    chemin = Path(chemin_dossier)
    chemin.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model_scaler, chemin / 'scaler.joblib')
    joblib.dump(model_pca, chemin / 'pca.joblib')
    joblib.dump(model_lof, chemin / 'lof.joblib')
    
    logger.info(f"Pipeline LOF sauvegardé avec succès dans le répertoire : {chemin_dossier}")

def charger_pipeline_lof(chemin_dossier: str = "modeles_sauvegardes/lof"):
    """
    Désérialise les composants du pipeline LOF depuis le disque.

    Args:
        chemin_dossier (str): Dossier de chargement.

    Returns:
        tuple: (model_lof, model_pca, model_scaler)
    """
    chemin = Path(chemin_dossier)
    
    if not chemin.exists():
        raise FileNotFoundError(f"Le répertoire {chemin_dossier} n'existe pas. Entraînement requis.")
        
    model_scaler = joblib.load(chemin / 'scaler.joblib')
    model_pca = joblib.load(chemin / 'pca.joblib')
    model_lof = joblib.load(chemin / 'lof.joblib')
    
    logger.info("Pipeline LOF chargé avec succès depuis le disque.")
    return model_lof, model_pca, model_scaler

def executer_pipeline_inference_lof(df_test, model_lof, model_pca, model_scaler, batch_size=32, tokenizer=None, model_bert=None):
    """
    Exécute l'inférence vectorisée par lots et agrège les résultats au niveau de la phrase.

    Args:
        df_test (pd.DataFrame): Dataset d'évaluation.
        model_lof: Modèle Local Outlier Factor.
        model_pca: Modèle PCA.
        model_scaler: Modèle Scaler.
        batch_size (int): Taille de lot d'inférence.
        tokenizer: Tokenizer linguistique.
        model_bert: Modèle dialectique BERT.

    Returns:
        pd.DataFrame: DataFrame test enrichi des prédictions (labels binaires et statuts d'erreur).
    """
    phrases_test = df_test['phrase_clinique'].tolist()
    
    all_tokens = []
    all_embeddings_list = []
    phrase_ids = []
    
    generateur = iterateur_embeddings_tokens(
        phrases_test, batch_size=batch_size, tokenizer=tokenizer, model=model_bert
    )
    
    for batch_tok, batch_emb, batch_ids in generateur:
        all_tokens.extend(batch_tok)
        all_embeddings_list.append(batch_emb)
        phrase_ids.extend(batch_ids)
        
    if len(all_embeddings_list) == 0:
        df_resultat = df_test.copy()
        df_resultat['label_pred'] = 0
        df_resultat['label_vrai'] = df_resultat['nb_errors'].apply(lambda x: 1 if x > 0 else 0)
        df_resultat['error_types_pred'] = "none"
        return df_resultat

    embeddings = np.vstack(all_embeddings_list)

    emb_scaled = model_scaler.transform(embeddings)
    emb_reduced = model_pca.transform(emb_scaled)
    preds_mots = model_lof.predict(emb_reduced)

    tag_to_label = {
        "[DRUG]": "drug",
        "[ROUTE]": "route",
        "[UNIT]": "unit_dosage",
        "[DOSE]": "dosage",
    }
    
    y_pred_phrases = np.zeros(len(phrases_test), dtype=int)
    detected_categories_per_phrase = {i: set() for i in range(len(phrases_test))}
    
    current_context = "none"
    current_phrase_id = -1

    for mot, pred, p_id in zip(all_tokens, preds_mots, phrase_ids):
        if p_id != current_phrase_id:
            current_context = "none"
            current_phrase_id = p_id

        if mot in tag_to_label:
            current_context = tag_to_label[mot]
            continue 
            
        if pred == -1:
            y_pred_phrases[p_id] = 1 
            if current_context != "none":
                detected_categories_per_phrase[p_id].add(current_context)

    labels_pred_phrases = []
    for i in range(len(phrases_test)):
        categories = detected_categories_per_phrase[i]
        if len(categories) > 0:
            labels_pred_phrases.append("|".join(list(categories)))
        elif y_pred_phrases[i] == 1:
            labels_pred_phrases.append("erreur_hors_contexte")
        else:
            labels_pred_phrases.append("none")

    df_resultat = df_test.copy()
    df_resultat['label_pred'] = y_pred_phrases
    df_resultat['label_vrai'] = df_resultat['nb_errors'].apply(lambda x: 1 if x > 0 else 0)
    df_resultat['error_types_pred'] = labels_pred_phrases

    return df_resultat